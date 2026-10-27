#!/usr/bin/env python3
"""FashionIQ reranking with the composed-query cosine score.

Does not load CIRR weights. Does not write under cirr/ paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bfsar.mvr.cqm_model import Blip2QTR
from bfsar.fiq import fiq_paths
from bfsar.mvr.cqm_data_fiq import (
    FIQClassicImages,
    default_topk_pt,
    fiq_targetpad_preprocess,
    load_fiq_val_queries,
)

logger = logging.getLogger("fiq_blip2_qtr_rerank")

_GALLERY_CACHE: Dict[str, Dict[str, Any]] = {}


def _abort_cirr(path: str) -> None:
    if "cirr" in str(path).lower():
        raise ValueError(f"FIQ QTR must not use a cirr path: {path}")


@torch.no_grad()
def extract_gallery_vit(
    names: Sequence[str],
    model: Blip2QTR,
    preprocess,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    cache_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    if cache_key and cache_key in _GALLERY_CACHE:
        cached = _GALLERY_CACHE[cache_key]
        if list(cached["names"]) == list(names):
            return cached["name_to_feat"]
    ds = FIQClassicImages(names, preprocess)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    device = next(model.parameters()).device
    name_to_feat: Dict[str, torch.Tensor] = {}
    model.eval()
    for batch_names, images in tqdm(loader, desc=f"qtr gallery {cache_key or ''}"):
        images = images.to(device, non_blocking=True)
        feats = model.img_embed(images).half().cpu()
        for n, f in zip(batch_names, feats):
            name_to_feat[str(n)] = f
    if cache_key:
        _GALLERY_CACHE[cache_key] = {"names": list(names), "name_to_feat": name_to_feat}
    return name_to_feat


def pool_recalls(labels_reranked: np.ndarray, ks: Sequence[int] = (10, 50, 100, 300)) -> Dict[str, float]:
    out = {}
    n = labels_reranked.shape[0]
    k_max = labels_reranked.shape[1]
    for k in ks:
        kk = min(int(k), k_max)
        out[f"pool_recall_at{k}"] = float(labels_reranked[:, :kk].any(axis=1).mean() * 100.0)
    out["coverage"] = float(labels_reranked.any(axis=1).mean() * 100.0)
    out["n_queries"] = float(n)
    return out


@torch.no_grad()
def rerank_category(
    model: Blip2QTR,
    category: str,
    *,
    topk_pt: Path,
    k_rerank: int = 100,
    score_mode: str = "itc",
    preprocess=None,
    gallery_batch: int = 8,
    num_workers: int = 4,
    max_queries: Optional[int] = None,
) -> Dict[str, Any]:
    _abort_cirr(str(topk_pt))
    payload = torch.load(topk_pt, map_location="cpu", weights_only=False)
    refs_pt = [str(x) for x in payload["reference_names"]]
    tgts_pt = [str(x) for x in payload["target_names"]]
    index_names = [str(x) for x in payload["index_names"]]
    topk = np.array(payload["sorted_index_names"], dtype=object)[:, :k_rerank]
    labels = payload["labels"]
    if torch.is_tensor(labels):
        labels = labels.numpy()
    labels = np.asarray(labels)[:, :k_rerank]

    split = str(payload.get("split") or "val")
    from bfsar.mvr.cqm_data_fiq import load_fiq_queries

    refs, tgts, caps = load_fiq_queries(category, split)
    if refs != refs_pt or tgts != tgts_pt:
        # Align official json to topk payload order.
        key_to_i = {(r, t): i for i, (r, t) in enumerate(zip(refs_pt, tgts_pt))}
        order = []
        for r, t, c in zip(refs, tgts, caps):
            if (r, t) not in key_to_i:
                continue
            order.append((key_to_i[(r, t)], r, t, c))
        order.sort(key=lambda x: x[0])
        idxs = [x[0] for x in order]
        refs = [x[1] for x in order]
        tgts = [x[2] for x in order]
        caps = [x[3] for x in order]
        topk = topk[idxs]
        labels = labels[idxs]

    if max_queries is not None:
        refs, tgts, caps = refs[:max_queries], tgts[:max_queries], caps[:max_queries]
        topk, labels = topk[:max_queries], labels[:max_queries]

    if preprocess is None:
        preprocess = fiq_targetpad_preprocess()
    name_to_feat = extract_gallery_vit(
        index_names, model, preprocess,
        batch_size=gallery_batch, num_workers=num_workers, cache_key=category,
    )
    device = next(model.parameters()).device

    def _score(ref_name: str, cand_names: Sequence[str], caption: str) -> torch.Tensor:
        ref_feat = name_to_feat[str(ref_name)].unsqueeze(0).to(device=device, dtype=torch.float32)
        cand = torch.stack([name_to_feat[str(n)] for n in cand_names]).to(device=device, dtype=torch.float32)
        if score_mode == "itc":
            return model.img_txt_fusion_val_itc(ref_feat, cand, caption)
        return model.img_txt_fusion_val(ref_feat, cand, caption)

    scores = []
    n = len(refs)
    it = tqdm(range(n), desc=f"qtr {category} K={k_rerank} {score_mode}", ncols=88)
    for i in it:
        logits = _score(refs[i], [str(x) for x in topk[i]], caps[i])
        scores.append(logits.detach().cpu())
    scores_t = torch.stack(scores, dim=0)
    pred_logits = scores_t.detach().cpu().numpy().astype(np.float32)
    rerank_idx = torch.argsort(scores_t, dim=-1, descending=True).numpy()
    labels_reranked = np.take_along_axis(labels.astype(bool), rerank_idx, axis=1)
    metrics = pool_recalls(labels_reranked, ks=(10, 50, 100, 300))
    logger.info(
        "FIQ %s %s | pool R@10=%.2f R@50=%.2f R@100=%.2f coverage=%.2f n=%d",
        category, score_mode,
        metrics["pool_recall_at10"], metrics["pool_recall_at50"], metrics["pool_recall_at100"],
        metrics["coverage"], int(metrics["n_queries"]),
    )
    return {
        "category": category,
        "metrics": metrics,
        "rerank_idx": rerank_idx,
        "pred_logits": pred_logits,
        "k_sorted": np.asarray(topk, dtype=object),
        "labels": np.asarray(labels, dtype=bool),
        "reference_names": list(refs),
        "target_names": list(tgts),
        "k_rerank": k_rerank,
        "score_mode": score_mode,
        "topk_pt": str(topk_pt),
    }


@torch.no_grad()
def rerank_all_categories(
    model: Blip2QTR,
    *,
    categories: Sequence[str] = fiq_paths.CATEGORIES,
    k_rerank: int = 100,
    score_mode: str = "itc",
    topk_dir: Optional[Path] = None,
    max_queries: Optional[int] = None,
    gallery_batch: int = 8,
    num_workers: int = 4,
) -> Dict[str, Any]:
    preprocess = fiq_targetpad_preprocess()
    per_cat = {}
    for cat in categories:
        if topk_dir is None:
            pt = default_topk_pt(cat)
        else:
            # Prefer pool size matching k_rerank (e.g. Top-500 compose41), else legacy Top-200.
            td = Path(topk_dir)
            pt = td / f"hbi_top{int(k_rerank)}_val_{cat}.pt"
            if not pt.is_file():
                pt = td / f"hbi_top200_val_{cat}.pt"
        _abort_cirr(str(pt))
        if not pt.is_file():
            raise FileNotFoundError(f"Missing FIQ BFSAR top-k: {pt}")
        per_cat[cat] = rerank_category(
            model, cat, topk_pt=pt, k_rerank=k_rerank, score_mode=score_mode,
            preprocess=preprocess, gallery_batch=gallery_batch, num_workers=num_workers,
            max_queries=max_queries,
        )
    keys = ("pool_recall_at10", "pool_recall_at50", "pool_recall_at100", "coverage")
    mean = {
        k: float(np.mean([per_cat[c]["metrics"][k] for c in categories]))
        for k in keys
    }
    logger.info(
        "FIQ mean %s | pool R@10=%.2f R@50=%.2f R@100=%.2f coverage=%.2f",
        score_mode, mean["pool_recall_at10"], mean["pool_recall_at50"],
        mean["pool_recall_at100"], mean["coverage"],
    )
    return {"mean": mean, "per_category": {c: per_cat[c]["metrics"] for c in categories}}
