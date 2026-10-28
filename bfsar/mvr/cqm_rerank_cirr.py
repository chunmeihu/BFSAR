#!/usr/bin/env python3
"""CIRR val/test rerank with BLIP-2 QTR (CQM + ITM Reranker)."""

from __future__ import annotations

import logging
import os
import sys
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bfsar.cirr import cirr_paths
from bfsar.mvr.cqm_data import (
    extract_blip2_index_features,
    get_cirr_preprocess,
    import_cirr_dataset,
)
from bfsar.mvr.cvs_infer import CR_SRC, REPO_ROOT, metrics_stage2_pool
from bfsar.mvr.cqm_model import (
    DEFAULT_ITM_ID,
    Blip2QTR,
    build_blip2_qtr,
    load_blip2_qtr_ckpt,
)
from bfsar.rerank.paths import DEFAULT_CIRR_TOPK_VAL

logger = logging.getLogger("cirr_blip2_qtr_rerank")

DEFAULT_TOPK_VAL = DEFAULT_CIRR_TOPK_VAL


@torch.no_grad()
def run_cirr_blip2_qtr_rerank(
    topk_pt: Path,
    *,
    split: str = "val",
    k_rerank: int = 50,
    ckpt_path: Optional[Path] = None,
    model: Optional[Blip2QTR] = None,
    device: Optional[torch.device] = None,
    pretrained_id: str = DEFAULT_ITM_ID,
    local_files_only: bool = False,
    max_queries: Optional[int] = None,
    num_workers: int = 0,
    score_mode: str = "itc",
) -> Dict[str, Any]:
    """Rerank a top-K pool with cosine similarity of the composed query."""
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        stage = "b" if score_mode == "itm" else "a"
        if ckpt_path is not None and Path(ckpt_path).is_file():
            meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            from bfsar.rerank.paths import blip2_pretrained_id

            pretrained_id = blip2_pretrained_id(str(meta.get("pretrained_id", pretrained_id)))
            stage = str(meta.get("stage", stage))
        model = build_blip2_qtr(
            device,
            pretrained_id=pretrained_id,
            local_files_only=local_files_only,
            freeze_vision=True,
            stage=stage,
        )
        if ckpt_path is not None and Path(ckpt_path).is_file():
            load_blip2_qtr_ckpt(model, str(ckpt_path), device)
            logger.info("Loaded QTR ckpt: %s", ckpt_path)
    else:
        if device is None:
            device = next(model.parameters()).device
        model = model.to(device)
    model.eval()

    preprocess = get_cirr_preprocess()
    CIRRDataset, collate_fn = import_cirr_dataset()
    os.chdir(CR_SRC)
    try:
        os.environ["CIR_NUM_WORKERS"] = str(num_workers)
        classic = CIRRDataset(split, "classic", preprocess, load_topk=str(topk_pt), K=k_rerank)
        index_features, index_names = extract_blip2_index_features(classic, model)
        relative = CIRRDataset(split, "relative", preprocess, load_topk=str(topk_pt), K=k_rerank)
    finally:
        os.chdir(REPO_ROOT)

    loader = DataLoader(
        relative,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
    name_to_feat = dict(zip(index_names, index_features))
    k_rerank = relative.K

    pred_logits: List[torch.Tensor] = []
    group_logits: List[torch.Tensor] = []
    target_names: List[str] = []
    reference_names: List[str] = []
    group_members_no_ref: List[List[str]] = []
    k_sorted_names: List[np.ndarray] = []
    labels_topk: List[np.ndarray] = []
    pairids: List[int] = []

    def _score(ref_feat, cand, captions):
        if score_mode == "itc":
            return model.img_txt_fusion_val_itc(ref_feat, cand, captions)
        return model.img_txt_fusion_val(ref_feat, cand, captions)

    n_done = 0
    for batch in tqdm(loader, desc=f"qtr rerank split={split} K={k_rerank} mode={score_mode}"):
        if max_queries is not None and n_done >= max_queries:
            break

        if split == "test1":
            (
                batch_pairids,
                batch_reference_names,
                captions,
                batch_group_members,
                K_sorted_index_names,
            ) = batch
            K_labels = None
            batch_target_names = [""]
            for pid in batch_pairids:
                pairids.append(int(pid.item() if hasattr(pid, "item") else pid))
        else:
            (
                batch_reference_names,
                batch_target_names,
                captions,
                batch_group_members,
                K_sorted_index_names,
                K_labels,
                _K_group_labels,
            ) = batch

        K_sorted_index_names = [x[0] for x in K_sorted_index_names]
        batch_group_members = np.array(batch_group_members).T.tolist()[0]
        k_sorted_names.append(np.array(K_sorted_index_names, dtype=object))
        if K_labels is not None:
            k_lab = K_labels.cpu().numpy() if torch.is_tensor(K_labels) else np.asarray(K_labels)
            labels_topk.append(k_lab.reshape(-1))

        ref_feat = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
        if k_rerank == 1:
            cand = itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
        else:
            cand = torch.stack(itemgetter(*K_sorted_index_names)(name_to_feat))
        logits = _score(ref_feat, cand, captions).unsqueeze(0)

        gm = [x for x in batch_group_members if x not in batch_reference_names]
        g_cand = torch.stack(itemgetter(*gm)(name_to_feat))
        g_logits = _score(ref_feat, g_cand, captions).unsqueeze(0)

        pred_logits.append(logits.cpu())
        group_logits.append(g_logits.cpu())
        target_names.extend(batch_target_names)
        reference_names.extend(batch_reference_names)
        group_members_no_ref.append(gm)
        n_done += 1

    pred_logits_t = torch.cat(pred_logits, dim=0)
    group_logits_t = torch.cat(group_logits, dim=0)
    k_sorted_arr = np.stack(k_sorted_names, axis=0)
    rerank_idx = torch.argsort(pred_logits_t, dim=-1, descending=True).numpy()
    reranked_topk = np.take_along_axis(k_sorted_arr, rerank_idx, axis=1)
    group_rerank_idx = torch.argsort(group_logits_t, dim=-1, descending=True).numpy()
    group_members_arr = np.array(group_members_no_ref, dtype=object)
    reranked_group = np.take_along_axis(group_members_arr, group_rerank_idx, axis=1)

    out: Dict[str, Any] = {
        "reranked_topk": reranked_topk,
        "reranked_group": reranked_group,
        "rerank_idx": rerank_idx,
        "group_rerank_idx": group_rerank_idx,
        "pred_logits": pred_logits_t.numpy(),
        "target_names": target_names,
        "reference_names": reference_names,
        "group_members_no_ref": group_members_no_ref,
        "k_rerank": k_rerank,
        "topk_pt": str(topk_pt),
        "split": split,
        "model": "blip2_qtr",
        "score_mode": score_mode,
        "group_logits": group_logits_t.numpy(),
    }
    if pairids:
        out["pairids"] = pairids
    if labels_topk:
        labels_arr = np.stack(labels_topk, axis=0)
        out["labels_topk"] = labels_arr
        out["pool_metrics"] = metrics_stage2_pool(
            rerank_idx, labels_arr, reranked_group, target_names
        )
    return out
