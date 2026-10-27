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
from bfsar.cirr.metrics import compute_cirr_metrics, rank_by_similarity
from bfsar.rerank.paths import (
    CR_CONFIGS,
    CR_ROOT,
    CR_SRC,
    DEFAULT_CIRR_STAGE1,
    DEFAULT_CIRR_STAGE2,
)

logger = logging.getLogger("hbi_blip_rerank")

ASSET_ROOT = Path(cirr_paths.ASSET_ROOT)
REPO_ROOT = Path(cirr_paths.REPO_ROOT)

# CIRR CR Stage-1/2 defaults (in-repo under weights/cr/).
DEFAULT_STAGE1 = DEFAULT_CIRR_STAGE1
DEFAULT_STAGE2 = DEFAULT_CIRR_STAGE2


def _ensure_cirr_image_symlinks() -> None:
    """Map BLIP path base_path/cirr_dataset/{dev,train,test1} -> BFSAR img_raw."""
    ds_root = Path(cirr_paths.CIRR_DATASET_ROOT)
    img_root = Path(cirr_paths.CIRR_IMAGE_ROOT)
    for sub in ("dev", "train", "test1"):
        link = ds_root / sub
        target = img_root / sub
        if link.is_symlink() or link.is_dir():
            continue
        if not target.is_dir():
            raise FileNotFoundError(f"Missing CIRR images: {target}")
        link.symlink_to(target, target_is_directory=True)
        logger.info("Symlink %s -> %s", link, target)


def _prefer_offline_hf_bert() -> None:
    """Use local Hugging Face cache when bert-base-uncased is already present."""
    snap_root = Path.home() / ".cache/huggingface/hub/models--bert-base-uncased/snapshots"
    if not snap_root.is_dir():
        return
    if not any(snap_root.iterdir()):
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


def coarse_rankings_no_ref(
    sim: np.ndarray,
    index_names: List[str],
    refs: Sequence[str],
    tgts: Sequence[str],
    groups: Sequence[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-gallery coarse order (ref removed), per-query labels, group labels."""
    sorted_names = rank_by_similarity(sim, index_names)
    n_q, n_g = sorted_names.shape
    ref_exp = np.repeat(np.array(refs), n_g).reshape(n_q, n_g)
    ref_mask = sorted_names != ref_exp
    sorted_no_ref = sorted_names[ref_mask].reshape(n_q, n_g - 1)

    tgts_exp = np.repeat(np.array(tgts), n_g - 1).reshape(n_q, -1)
    labels_full = sorted_no_ref == tgts_exp

    group_arr = np.array(groups, dtype=object)
    group_mask = (sorted_no_ref[..., None] == group_arr[:, None, :]).sum(-1).astype(bool)
    group_labels_full = labels_full[group_mask].reshape(n_q, -1)
    return sorted_no_ref, labels_full, group_labels_full


def build_hbi_topk_pt_payload(
    sim: np.ndarray,
    index_names: List[str],
    refs: List[str],
    tgts: List[str],
    groups: List[Sequence[str]],
    k_save: int,
) -> Dict[str, Any]:
    """Build a Candidate-Reranking-compatible top-k dict from BFSAR similarity."""
    sorted_no_ref, labels_full, group_labels_full = coarse_rankings_no_ref(
        sim, index_names, refs, tgts, groups
    )
    k_save = min(int(k_save), sorted_no_ref.shape[1])
    return {
        "sorted_index_names": sorted_no_ref[:, :k_save],
        "target_names": list(tgts),
        "index_names": list(index_names),
        "labels": labels_full[:, :k_save],
        "group_labels": group_labels_full,
        "split": "val",
        "source": "hbi_triple_minus_removed",
    }


def build_full_val_topk_pt(
    sim_hbi: np.ndarray,
    hbi_refs: List[str],
    hbi_tgts: List[str],
    index_names: List[str],
    k_save: int,
    caption_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Align BFSAR coarse sim to all CIRR val triplets (for BLIP CIRRDataset asserts)."""
    caption_json = caption_json or cirr_paths.caption_json("val")
    with open(caption_json, "r") as f:
        triplets = json.load(f)

    key_to_row = {cirr_paths.pair_key(r, t): i for i, (r, t) in enumerate(zip(hbi_refs, hbi_tgts))}
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
                sim_hbi[r], index_names, ref, tgt, t["img_set"]["members"]
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
        "source": "hbi_coarse_full_val",
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
def run_blip_stage2_rerank(
    topk_pt: Path,
    stage1_path: Path,
    stage2_path: Path,
    k_rerank: int = 50,
    max_queries: Optional[int] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Run Candidate-Reranking stage-2 on a top-k file (e.g. from BFSAR coarse)."""
    model_stage1, model_stage2, preprocess, mods = load_blip_models(stage1_path, stage2_path)
    device = mods["device"]
    CIRRDataset = mods["CIRRDataset"]
    extract_index_features = mods["extract_index_features"]
    collate_fn = mods["collate_fn"]

    os.chdir(CR_SRC)
    try:
        os.environ["CIR_NUM_WORKERS"] = str(num_workers)
        classic = CIRRDataset("val", "classic", preprocess, load_topk=str(topk_pt), K=k_rerank)
        index_features, index_names = extract_index_features(classic, model_stage2, blip_stage2=True)
        relative = CIRRDataset("val", "relative", preprocess, load_topk=str(topk_pt), K=k_rerank)
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

    n_done = 0
    for batch in tqdm(loader, desc="BLIP stage2 rerank"):
        if max_queries is not None and n_done >= max_queries:
            break
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

        k_lab = K_labels
        if torch.is_tensor(k_lab):
            k_lab = k_lab.cpu().numpy()
        k_lab = np.asarray(k_lab).reshape(-1)
        if k_lab.any():
            reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            z_t = model_stage1.img_txt_fusion(
                reference_image_features, None, captions, train=False, return_raw=True
            )
            if k_rerank == 1:
                cand = itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
            else:
                cand = torch.stack(itemgetter(*K_sorted_index_names)(name_to_feat))
            logits = model_stage2.img_txt_fusion_val(z_t, cand, captions).unsqueeze(0)
        else:
            logits = torch.full((1, k_rerank), -99999.99, device=device)

        with torch.no_grad():
            reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            z_t = model_stage1.img_txt_fusion(
                reference_image_features, None, captions, train=False, return_raw=True
            )
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


def build_test1_topk_pt(
    sim_hbi: np.ndarray,
    index_names: List[str],
    k_save: int = 200,
) -> Dict[str, Any]:
    """Export BFSAR coarse Top-K for CIRR test1 (no labels; split=test1)."""
    from bfsar.cirr.vision_desc.cirr_triple_test1 import load_test1_triplets

    triplets = load_test1_triplets()
    if sim_hbi.shape[0] != len(triplets):
        raise ValueError(
            f"sim rows {sim_hbi.shape[0]} != test1 triplets {len(triplets)}; re-run compute_hbi_test1_coarse"
        )
    k_save = min(int(k_save), sim_hbi.shape[1] - 1)
    rows = []
    for i, t in enumerate(triplets):
        ref = t["reference"]
        order = np.argsort(-sim_hbi[i])
        names = np.array(index_names)[order]
        no_ref = names[names != ref]
        if len(no_ref) < k_save:
            raise ValueError(f"query {i}: gallery after ref-removal {len(no_ref)} < k_save {k_save}")
        rows.append(no_ref[:k_save])
    return {
        "sorted_index_names": np.stack(rows, axis=0),
        "index_names": list(index_names),
        "split": "test1",
        "source": "hbi_coarse_test1",
    }


def generate_hbi_coarse_test_submission_files(
    sim_hbi: np.ndarray,
    index_names: List[str],
    submission_name: str,
    out_dir: Path,
) -> Dict[str, Path]:
    """CIRR test1 submission from BFSAR coarse sim only (full gallery, no BLIP)."""
    from bfsar.cirr.vision_desc.cirr_triple_test1 import load_test1_triplets

    triplets = load_test1_triplets()
    if sim_hbi.shape[0] != len(triplets):
        raise ValueError(f"sim rows {sim_hbi.shape[0]} != triplets {len(triplets)}")

    sorted_names = rank_by_similarity(sim_hbi, index_names)
    n_queries = len(triplets)
    n_gallery = sorted_names.shape[1]
    ref_names = [t["reference"] for t in triplets]
    pairids = [int(t["pairid"]) for t in triplets]
    group_members = [list(t["img_set"]["members"]) for t in triplets]

    ref_expanded = np.repeat(np.array(ref_names), n_gallery).reshape(n_queries, n_gallery)
    sorted_no_ref = sorted_names[sorted_names != ref_expanded].reshape(n_queries, n_gallery - 1)

    group_members_arr = np.array(group_members, dtype=object)
    group_mask = (sorted_no_ref[..., None] == group_members_arr[:, None, :]).sum(-1).astype(bool)
    sorted_group = sorted_no_ref[group_mask].reshape(n_queries, -1)

    pairid_to_predictions = {
        str(pid): sorted_no_ref[i, :50].tolist() for i, pid in enumerate(pairids)
    }
    pairid_to_group_predictions = {
        str(pid): sorted_group[i, :3].tolist() for i, pid in enumerate(pairids)
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    recall_path = out_dir / f"recall_submission_{submission_name}.json"
    subset_path = out_dir / f"recall_subset_submission_{submission_name}.json"
    submission = {"version": "rc2", "metric": "recall", **pairid_to_predictions}
    group_submission = {"version": "rc2", "metric": "recall_subset", **pairid_to_group_predictions}
    recall_path.write_text(json.dumps(submission, sort_keys=True), encoding="utf-8")
    subset_path.write_text(json.dumps(group_submission, sort_keys=True), encoding="utf-8")
    logger.info("Wrote coarse submissions %s and %s", recall_path, subset_path)
    return {"recall": recall_path, "recall_subset": subset_path}


@torch.no_grad()
def generate_cirr_test_submission_files(
    topk_pt: Path,
    stage1_path: Path,
    stage2_path: Path,
    submission_name: str,
    out_dir: Path,
    k_rerank: int = 50,
    num_workers: int = 0,
) -> Dict[str, Path]:
    """Write recall_submission_*.json and recall_subset_submission_*.json for CIRR test server."""
    model_stage1, model_stage2, preprocess, mods = load_blip_models(stage1_path, stage2_path)
    device = mods["device"]
    CIRRDataset = mods["CIRRDataset"]
    extract_index_features = mods["extract_index_features"]
    collate_fn = mods["collate_fn"]

    os.chdir(CR_SRC)
    try:
        os.environ["CIR_NUM_WORKERS"] = str(num_workers)
        classic = CIRRDataset("test1", "classic", preprocess, load_topk=str(topk_pt), K=k_rerank)
        index_features, index_names = extract_index_features(classic, model_stage2, blip_stage2=True)
        relative = CIRRDataset("test1", "relative", preprocess, load_topk=str(topk_pt), K=k_rerank)
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
    pairs_id: List = []
    pred_logits = []
    group_logits = []
    k_sorted_names: List[np.ndarray] = []
    group_members_no_ref: List[List[str]] = []

    for batch in tqdm(loader, desc="CIRR test1 BLIP rerank"):
        batch_pairs_id, batch_reference_names, captions, batch_group_members, K_sorted_index_names = batch
        K_sorted_index_names = [x[0] for x in K_sorted_index_names]
        batch_group_members = np.array(batch_group_members).T.tolist()[0]
        k_sorted_names.append(np.array(K_sorted_index_names, dtype=object))
        with torch.no_grad():
            reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            z_t = model_stage1.img_txt_fusion(
                reference_image_features, None, captions, train=False, return_raw=True
            )
            if k_rerank == 1:
                cand = itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
            else:
                cand = torch.stack(itemgetter(*K_sorted_index_names)(name_to_feat))
            logits = model_stage2.img_txt_fusion_val(z_t, cand, captions).unsqueeze(0)
            gm = [x for x in batch_group_members if x not in batch_reference_names]
            g_cand = torch.stack(itemgetter(*gm)(name_to_feat))
            g_logits = model_stage2.img_txt_fusion_val(z_t, g_cand, captions).unsqueeze(0)
        pred_logits.append(logits.cpu())
        group_logits.append(g_logits.cpu())
        group_members_no_ref.append(gm)
        pairs_id.extend(batch_pairs_id)

    pred_logits_t = torch.cat(pred_logits, dim=0)
    group_logits_t = torch.cat(group_logits, dim=0)
    k_arr = np.stack(k_sorted_names, axis=0)
    sorted_idx = torch.argsort(pred_logits_t, dim=-1, descending=True).numpy()
    sorted_global = np.take_along_axis(k_arr, sorted_idx, axis=1)

    group_members_arr = np.array(group_members_no_ref, dtype=object)
    group_sorted_idx = torch.argsort(group_logits_t, dim=-1, descending=True).numpy()
    sorted_group = np.take_along_axis(group_members_arr, group_sorted_idx, axis=1)

    pairid_to_predictions = {
        str(int(pid)): sorted_global[i, :50].tolist() for i, pid in enumerate(pairs_id)
    }
    pairid_to_group_predictions = {
        str(int(pid)): sorted_group[i, :3].tolist() for i, pid in enumerate(pairs_id)
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    recall_path = out_dir / f"recall_submission_{submission_name}.json"
    subset_path = out_dir / f"recall_subset_submission_{submission_name}.json"
    submission = {"version": "rc2", "metric": "recall", **pairid_to_predictions}
    group_submission = {"version": "rc2", "metric": "recall_subset", **pairid_to_group_predictions}
    recall_path.write_text(json.dumps(submission, sort_keys=True), encoding="utf-8")
    subset_path.write_text(json.dumps(group_submission, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s and %s", recall_path, subset_path)
    return {"recall": recall_path, "recall_subset": subset_path}
