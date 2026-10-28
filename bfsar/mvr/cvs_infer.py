"""BFSAR coarse retrieval + Candidate-Reranking-CIR BLIP stage-2 rerank utilities."""

from __future__ import annotations

import json
import logging
import os
import sys
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from bfsar.cirr import cirr_paths
from bfsar.rerank.paths import (
    CR_CONFIGS,
    CR_ROOT,
    CR_SRC,
    DEFAULT_CIRR_STAGE1,
    DEFAULT_CIRR_STAGE2,
)

logger = logging.getLogger("bfsar_cvs")

ASSET_ROOT = Path(cirr_paths.ASSET_ROOT)
REPO_ROOT = Path(cirr_paths.REPO_ROOT)

# CIRR candidate-verification stage-1/2 defaults.
DEFAULT_STAGE1 = DEFAULT_CIRR_STAGE1
DEFAULT_STAGE2 = DEFAULT_CIRR_STAGE2


def _ensure_cirr_image_symlinks() -> None:
    """BLIP opens images under cirr_dataset/{dev,train,test1}.

    The official release already stores them there. If this machine still
    keeps the real files in img_raw, link those three folders across.
    """
    ds_root = Path(cirr_paths.CIRR_DATASET_ROOT)
    img_root = Path(cirr_paths.CIRR_DATA_ROOT) / "img_raw"
    for sub in ("dev", "train", "test1"):
        link = ds_root / sub
        if link.is_symlink() or link.is_dir():
            continue
        target = img_root / sub
        if not target.is_dir():
            raise FileNotFoundError(
                f"Missing CIRR images: {link}. Unzip the official release under {ds_root}."
            )
        link.symlink_to(target, target_is_directory=True)
        logger.info("Symlink %s -> %s", link, target)


def _prefer_offline_hf_bert() -> None:
    """Candidate verification loads bert-base-uncased from weights/bert."""
    local_vocab = REPO_ROOT / "weights" / "bert" / "bert-base-uncased" / "vocab.txt"
    if local_vocab.is_file():
        logger.info("bert-base-uncased: %s", local_vocab.parent)
        return
    snap_root = Path.home() / ".cache/huggingface/hub/models--bert-base-uncased/snapshots"
    if not snap_root.is_dir() or not any(snap_root.iterdir()):
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("HF_ENDPOINT", None)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    logger.info("bert-base-uncased: using local HF cache (offline, no HF_ENDPOINT/proxy)")


def _import_blip_modules():
    _prefer_offline_hf_bert()
    _ensure_cirr_image_symlinks()
    if str(CR_SRC) not in sys.path:
        sys.path.insert(0, str(CR_SRC))
    prev = os.getcwd()
    os.chdir(CR_SRC)
    try:
        import data_utils  # noqa: WPS433

        data_utils.base_path = Path(cirr_paths.CIRR_DATASET_ROOT).parent
        from data_utils import CIRRDataset, targetpad_transform  # noqa: WPS433
        from utils import collate_fn, device, extract_index_features  # noqa: WPS433
        from blip_stage1 import blip_stage1  # noqa: WPS433
        from blip_stage2 import blip_stage2  # noqa: WPS433
        import yaml  # noqa: WPS433
    finally:
        os.chdir(prev)
    return {
        "data_utils": data_utils,
        "CIRRDataset": CIRRDataset,
        "targetpad_transform": targetpad_transform,
        "collate_fn": collate_fn,
        "device": device,
        "extract_index_features": extract_index_features,
        "blip_stage1": blip_stage1,
        "blip_stage2": blip_stage2,
        "yaml": yaml,
    }


def align_gallery_index_to_cirr(
    sim: np.ndarray,
    index_names: List[str],
    split: str = "val",
) -> Tuple[np.ndarray, List[str]]:
    """Reorder gallery columns to match official split.rc2.{split}.json key order."""
    import json

    with open(cirr_paths.split_json(split), "r") as f:
        official = list(json.load(f).keys())
    if index_names == official:
        return sim, index_names
    col = {n: i for i, n in enumerate(index_names)}
    missing = [n for n in official if n not in col]
    if missing:
        raise KeyError(f"BFSAR sim missing {len(missing)} gallery ids, e.g. {missing[:3]}")
    sim = sim[:, [col[n] for n in official]]
    return sim, official


def build_full_val_topk_pt(
    sim: np.ndarray,
    refs: List[str],
    tgts: List[str],
    index_names: List[str],
    k_save: int,
    caption_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Align coarse similarity to all CIRR val triplets."""
    caption_json = caption_json or cirr_paths.caption_json("val")
    with open(caption_json, "r") as f:
        triplets = json.load(f)

    key_to_row = {cirr_paths.pair_key(r, t): i for i, (r, t) in enumerate(zip(refs, tgts))}
    missing: List[str] = []
    tgts_all: List[str] = []
    full_sorted = []
    full_labels = []
    full_group_labels = []

    for t in triplets:
        ref = t["reference"]
        tgt = t["target_hard"]
        key = cirr_paths.pair_key(ref, tgt)
        tgts_all.append(tgt)
        if key in key_to_row:
            r = key_to_row[key]
            s_no_ref, lab, glab = _single_query_rank(
                sim[r], index_names, ref, tgt, t["img_set"]["members"]
            )
        else:
            missing.append(key)
            s_no_ref, lab, glab = _fallback_rank(
                index_names, ref, tgt, t["img_set"]["members"]
            )
        full_sorted.append(s_no_ref)
        full_labels.append(lab)
        full_group_labels.append(glab)

    if missing:
        logger.warning(
            "%d val triplets missing from BFSAR sim cache (alphabetical fallback rank): %s",
            len(missing),
            missing[:5],
        )

    sorted_no_ref = np.stack(full_sorted, axis=0)
    labels_full = np.stack(full_labels, axis=0)
    group_labels_full = np.zeros((len(full_group_labels), 5), dtype=bool)
    for i, g in enumerate(full_group_labels):
        g = np.asarray(g).reshape(-1)
        group_labels_full[i, : min(5, len(g))] = g[:5]
    k_save = min(int(k_save), sorted_no_ref.shape[1])
    return {
        "sorted_index_names": sorted_no_ref[:, :k_save],
        "target_names": tgts_all,
        "index_names": list(index_names),
        "labels": labels_full[:, :k_save],
        "group_labels": group_labels_full,
        "split": "val",
        "source": "coarse_val",
        "missing_scrs_keys": missing,
    }


def _single_query_rank(
    sim_row: np.ndarray,
    index_names: List[str],
    ref: str,
    tgt: str,
    group_members: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-sim_row)
    sorted_names = np.array(index_names)[order]
    ref_mask = sorted_names != ref
    sorted_no_ref = sorted_names[ref_mask]
    labels = sorted_no_ref == tgt
    group_arr = np.array(group_members, dtype=object)
    group_mask = (sorted_no_ref[:, None] == group_arr[None, :]).any(axis=1)
    group_labels = labels[group_mask]
    return sorted_no_ref, labels, group_labels


def _fallback_rank(
    index_names: List[str],
    ref: str,
    tgt: str,
    group_members: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = np.array([n for n in index_names if n != ref])
    labels = names == tgt
    group_arr = np.array(group_members, dtype=object)
    group_mask = (names[:, None] == group_arr[None, :]).any(axis=1)
    group_labels = labels[group_mask]
    return names, labels, group_labels


def save_topk_pt(payload: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    for key in ("labels", "group_labels"):
        if key in out and isinstance(out[key], np.ndarray):
            out[key] = torch.from_numpy(out[key])
    torch.save(out, path)
    logger.info("Saved top-k pt: %s shape=%s", path, payload["sorted_index_names"].shape)
    return path


def load_blip_models(
    stage1_path: Path,
    stage2_path: Path,
    target_ratio: float = 1.25,
) -> Tuple[Any, Any, Any, dict]:
    mods = _import_blip_modules()
    device = mods["device"]
    os.chdir(CR_SRC)
    try:
        yaml = mods["yaml"]
        cfg1 = yaml.load(open(CR_CONFIGS / "retrieval_coco.yaml", "r"), Loader=yaml.Loader)
        med_cfg = str(CR_CONFIGS / "med_config.json")
        # Skip BLIP URL init: stage ckpts fully overwrite weights (offline-friendly).
        model_stage1 = mods["blip_stage1"](
            pretrained="",
            image_size=cfg1["image_size"],
            vit=cfg1["vit"],
            vit_grad_ckpt=cfg1["vit_grad_ckpt"],
            vit_ckpt_layer=cfg1["vit_ckpt_layer"],
            med_config=med_cfg,
        ).to(device)
        state1 = torch.load(stage1_path, map_location=device, weights_only=False)
        model_stage1.load_state_dict(state1["BLIP_Retrieval"])
        model_stage1.eval()

        cfg2 = yaml.load(open(CR_CONFIGS / "nlvr.yaml", "r"), Loader=yaml.Loader)
        model_stage2 = mods["blip_stage2"](
            pretrained="",
            image_size=cfg2["image_size"],
            med_config=str(CR_SRC / "med_config.json"),
            vit=cfg2["vit"],
            vit_grad_ckpt=cfg2["vit_grad_ckpt"],
            vit_ckpt_layer=cfg2["vit_ckpt_layer"],
        ).to(device)
        state2 = torch.load(stage2_path, map_location=device, weights_only=False)
        model_stage2.load_state_dict(state2["BLIP_NLVR"])
        model_stage2.eval()

        preprocess = mods["targetpad_transform"](target_ratio, 384)
    finally:
        os.chdir(REPO_ROOT)
    return model_stage1, model_stage2, preprocess, mods


@torch.no_grad()
def build_test1_topk_pt(coarse_npz: Path, k_save: int) -> Dict[str, Any]:
    """Top-K for CIRR test1, one row per official caption, no labels."""
    d = np.load(coarse_npz, allow_pickle=True)
    if "pairids" not in d.files:
        raise KeyError(
            f"{coarse_npz} has no pairids. Re-run: bash scripts/eval_dpic_cirr.sh --split test"
        )
    sim = d["sim"].astype(np.float32)
    index_names = [str(x) for x in d["index_names"].tolist()]
    pairids = [int(x) for x in d["pairids"].tolist()]
    sim, index_names = align_gallery_index_to_cirr(sim, index_names, split="test1")
    with open(cirr_paths.caption_json("test1"), "r", encoding="utf-8") as f:
        triplets = json.load(f)
    pid_to_row = {pid: i for i, pid in enumerate(pairids)}
    rows = []
    missing = 0
    for t in triplets:
        pid = int(t["pairid"])
        ref = t["reference"]
        if pid not in pid_to_row:
            missing += 1
            names = [n for n in index_names if n != ref]
            rows.append(names[:k_save])
            continue
        order = np.argsort(-sim[pid_to_row[pid]])
        names = np.array(index_names)[order]
        no_ref = [str(n) for n in names if str(n) != ref]
        if len(no_ref) < k_save:
            raise ValueError(f"pair {pid}: gallery after ref-removal {len(no_ref)} < {k_save}")
        rows.append(no_ref[:k_save])
    if missing:
        logger.warning("%d test1 triplets missing from coarse scores", missing)
    return {
        "sorted_index_names": np.stack([np.asarray(row) for row in rows], axis=0),
        "index_names": list(index_names),
        "pairids": [int(t["pairid"]) for t in triplets],
        "split": "test1",
        "source": "dpic_dual_test1",
    }


def run_blip_stage2_rerank(
    topk_pt: Path,
    stage1_path: Path,
    stage2_path: Path,
    k_rerank: int = 50,
    max_queries: Optional[int] = None,
    num_workers: int = 0,
    split: str = "val",
) -> Dict[str, Any]:
    """Run Candidate-Reranking stage-2 on a top-k file (e.g. from BFSAR coarse)."""
    file_split = "test1" if split in ("test", "test1") else "val"
    model_stage1, model_stage2, preprocess, mods = load_blip_models(stage1_path, stage2_path)
    device = mods["device"]
    CIRRDataset = mods["CIRRDataset"]
    extract_index_features = mods["extract_index_features"]
    collate_fn = mods["collate_fn"]

    os.chdir(CR_SRC)
    try:
        os.environ["CIR_NUM_WORKERS"] = str(num_workers)
        classic = CIRRDataset(file_split, "classic", preprocess, load_topk=str(topk_pt), K=k_rerank)
        index_features, index_names = extract_index_features(classic, model_stage2, blip_stage2=True)
        relative = CIRRDataset(file_split, "relative", preprocess, load_topk=str(topk_pt), K=k_rerank)
    finally:
        os.chdir(REPO_ROOT)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset=relative,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
    name_to_feat = dict(zip(index_names, index_features))
    k_rerank = relative.K

    pred_logits = []
    group_logits = []
    target_names: List[str] = []
    reference_names: List[str] = []
    group_members_no_ref: List[List[str]] = []
    k_sorted_names: List[np.ndarray] = []
    pairids: List[int] = []

    n_done = 0
    for batch in tqdm(loader, desc=f"BLIP stage2 rerank {file_split}"):
        if max_queries is not None and n_done >= max_queries:
            break
        if file_split == "test1":
            (
                batch_pairs_id,
                batch_reference_names,
                captions,
                batch_group_members,
                K_sorted_index_names,
            ) = batch
            batch_target_names = [""] * len(batch_reference_names)
            K_labels = None
            for pid in batch_pairs_id:
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

        if K_labels is None:
            score_candidates = True
        else:
            k_lab = K_labels
            if torch.is_tensor(k_lab):
                k_lab = k_lab.cpu().numpy()
            score_candidates = bool(np.asarray(k_lab).reshape(-1).any())
        with torch.no_grad():
            reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            z_t = model_stage1.img_txt_fusion(
                reference_image_features, None, captions, train=False, return_raw=True
            )
            if score_candidates:
                if k_rerank == 1:
                    cand = itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
                    logits = model_stage2.img_txt_fusion_val(z_t, cand, captions).unsqueeze(0)
                else:
                    feats = itemgetter(*K_sorted_index_names)(name_to_feat)
                    parts = []
                    for start in range(0, len(K_sorted_index_names), 50):
                        cand = torch.stack(feats[start:start + 50])
                        parts.append(model_stage2.img_txt_fusion_val(z_t, cand, captions))
                    logits = torch.cat(parts, dim=0).unsqueeze(0)
            else:
                logits = torch.full((1, k_rerank), -99999.99, device=device)
            gm = [x for x in batch_group_members if x not in batch_reference_names]
            g_cand = torch.stack(itemgetter(*gm)(name_to_feat))
            g_logits = model_stage2.img_txt_fusion_val(z_t, g_cand, captions).unsqueeze(0)

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

    return {
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
        "group_logits": group_logits_t.numpy(),
        "pairids": pairids,
    }


def merge_rerank_with_coarse_gallery(
    coarse_sorted_no_ref: np.ndarray,
    reranked_topk: np.ndarray,
    n_reranked: Optional[int] = None,
) -> np.ndarray:
    """Place BLIP-reranked head; append remaining gallery in coarse order.

    If ``n_reranked`` is less than ``len(coarse_sorted_no_ref)``, only the first
    ``n_reranked`` queries are updated; the rest stay at coarse ranking.
    """
    n_q, n_g = coarse_sorted_no_ref.shape
    n_reranked = n_q if n_reranked is None else min(n_reranked, len(reranked_topk), n_q)
    out = []
    for i in range(n_q):
        if i >= n_reranked:
            out.append(list(coarse_sorted_no_ref[i]))
            continue
        head = list(reranked_topk[i])
        head_set = set(head)
        tail = [x for x in coarse_sorted_no_ref[i] if x not in head_set]
        merged = head + tail
        if len(merged) < n_g:
            merged.extend([x for x in coarse_sorted_no_ref[i] if x not in merged])
        out.append(merged[:n_g])
    return np.array(out, dtype=object)


def metrics_stage2_pool(
    rerank_idx: np.ndarray,
    labels_topk: np.ndarray,
    reranked_group_names: np.ndarray,
    target_names: Sequence[str],
) -> Dict[str, float]:
    """Metrics within stage-1 top-K pool (Candidate-Reranking stage2 convention).

    Group recall: after BLIP reranks the five non-reference group images, check
    whether ``target_hard`` is in top-1/2/3 of those names (see validate_stage2.py).
    """
    labels = np.take_along_axis(labels_topk, rerank_idx, axis=1)
    tgts = np.asarray(target_names).reshape(-1, 1)
    group_labels = reranked_group_names == tgts

    def r_at(arr, k):
        k = min(k, arr.shape[1])
        return float(arr[:, :k].sum() / len(arr) * 100)

    return {
        "pool_recall_at1": r_at(labels, 1),
        "pool_recall_at5": r_at(labels, 5),
        "pool_recall_at10": r_at(labels, 10),
        "pool_recall_at50": r_at(labels, 50),
        "pool_group_recall_at1": r_at(group_labels, 1),
        "pool_group_recall_at2": r_at(group_labels, 2),
        "pool_group_recall_at3": r_at(group_labels, 3),
    }
