#!/usr/bin/env python3
"""CIRR test1: BFSAR coarse Top-50 + CQM ITC + CR NLVR, val-best linear fusion.

Test1 has no public labels. This writes official submission JSONs for the CIRR server.
Weights are frozen from val z-score grid: 0.15 BFSAR + 0.40 CQM + 0.45 CR.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr import cirr_paths
from bfsar.mvr.fuse_cirr import (
    DEFAULT_COMPOSER,
    _recover_k_sorted,
    zscore_rows,
)
from bfsar.mvr.cvs_infer import (
    CR_SRC,
    DEFAULT_STAGE1,
    DEFAULT_STAGE2,
    REPO_ROOT,
    align_gallery_index_to_cirr,
    load_blip_models,
)
from bfsar.rerank.paths import DEFAULT_CIRR_TOPK_TEST1

logger = logging.getLogger("fusion_test1")

DEFAULT_TOPK = DEFAULT_CIRR_TOPK_TEST1
DEFAULT_COARSE_NPZ = ROOT / "output/cirr_vision_desc/hbi_triple_removed_test1_sim.npz"
DEFAULT_OUT = ROOT / "output/cirr_rerank/fusion_hbi_cqm_cr_test1"
LAMBDA_1, LAMBDA_2, LAMBDA_3 = 0.25, 0.40, 0.35


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _as_int_pid(pid) -> int:
    if hasattr(pid, "item"):
        return int(pid.item())
    return int(pid)


@torch.no_grad()
def run_cr_test1(topk_pt: Path, stage1: Path, stage2: Path, k_rerank: int, num_workers: int) -> Dict[str, Any]:
    model_stage1, model_stage2, preprocess, mods = load_blip_models(stage1, stage2)
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
        relative, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, shuffle=False
    )
    name_to_feat = dict(zip(index_names, index_features))
    k_rerank = relative.K
    pred, group, refs, gms, k_sorted, pairids = [], [], [], [], [], []
    for batch in tqdm(loader, desc="CR NLVR test1"):
        batch_pairs_id, batch_reference_names, captions, batch_group_members, K_sorted_index_names = batch
        K_sorted_index_names = [x[0] for x in K_sorted_index_names]
        batch_group_members = np.array(batch_group_members).T.tolist()[0]
        k_sorted.append(np.array(K_sorted_index_names, dtype=object))
        ref_feat = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
        z_t = model_stage1.img_txt_fusion(ref_feat, None, captions, train=False, return_raw=True)
        cand = (
            itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
            if k_rerank == 1
            else torch.stack(itemgetter(*K_sorted_index_names)(name_to_feat))
        )
        logits = model_stage2.img_txt_fusion_val(z_t, cand, captions).unsqueeze(0)
        gm = [x for x in batch_group_members if x not in batch_reference_names]
        g_logits = model_stage2.img_txt_fusion_val(z_t, torch.stack(itemgetter(*gm)(name_to_feat)), captions).unsqueeze(0)
        pred.append(logits.cpu())
        group.append(g_logits.cpu())
        refs.extend(batch_reference_names)
        gms.append(gm)
        pairids.extend(_as_int_pid(p) for p in batch_pairs_id)
    return {
        "pred_logits": torch.cat(pred, dim=0).numpy().astype(np.float32),
        "group_logits": torch.cat(group, dim=0).numpy().astype(np.float32),
        "reference_names": list(refs),
        "group_members_no_ref": gms,
        "k_sorted": np.stack(k_sorted, axis=0),
        "pairids": pairids,
    }


def run_cqm_test1(args) -> Dict[str, Any]:
    from bfsar.mvr.cqm_rerank_cirr import run_cirr_blip2_qtr_rerank

    out = run_cirr_blip2_qtr_rerank(
        Path(args.topk_pt),
        split="test1",
        k_rerank=args.k_rerank,
        ckpt_path=Path(args.cqm_ckpt),
        local_files_only=True,
        num_workers=args.num_workers,
        score_mode="itc",
    )
    return {
        "pred_logits": np.asarray(out["pred_logits"], dtype=np.float32),
        "group_logits": np.asarray(out["group_logits"], dtype=np.float32),
        "reference_names": list(out["reference_names"]),
        "group_members_no_ref": out["group_members_no_ref"],
        "k_sorted": _recover_k_sorted(out["reranked_topk"], out["rerank_idx"]),
        "pairids": [int(p) for p in out["pairids"]],
    }


def hbi_scores_test1(
    npz_path: Path,
    pairids: List[int],
    k_sorted: np.ndarray,
    group_members: List[List[str]],
) -> Dict[str, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    sim = d["sim_hbi"].astype(np.float32)
    index_names = list(d["index_names"])
    sim, index_names = align_gallery_index_to_cirr(sim, index_names, split="test1")
    name_to_j = {str(n): i for i, n in enumerate(index_names)}
    pid_to_row = {int(p): i for i, p in enumerate(d["pairids"].tolist())}
    n, k = k_sorted.shape
    scores = np.zeros((n, k), dtype=np.float32)
    g_len = max(len(g) for g in group_members)
    g_scores = np.zeros((n, g_len), dtype=np.float32)
    missing = 0
    for i, pid in enumerate(pairids):
        if pid not in pid_to_row:
            scores[i] = np.linspace(1.0, 0.0, k, dtype=np.float32)
            missing += 1
            continue
        row = sim[pid_to_row[pid]]
        scores[i] = row[[name_to_j[str(nm)] for nm in k_sorted[i]]]
        gcols = [name_to_j[str(nm)] for nm in group_members[i]]
        g_scores[i, : len(gcols)] = row[gcols]
    if missing:
        logger.warning("BFSAR missing %d / %d pairids", missing, n)
    return {"pred_logits": scores, "group_logits": g_scores, "missing": missing}


def _save_scores(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_logits=payload["pred_logits"],
        group_logits=payload["group_logits"],
        reference_names=np.asarray(payload["reference_names"], dtype=object),
        k_sorted=np.asarray(payload["k_sorted"], dtype=object),
        group_members=np.asarray(payload["group_members_no_ref"], dtype=object),
        pairids=np.asarray(payload["pairids"], dtype=np.int64),
    )


def _load_scores(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {
        "pred_logits": d["pred_logits"].astype(np.float32),
        "group_logits": d["group_logits"].astype(np.float32),
        "reference_names": [str(x) for x in d["reference_names"].tolist()],
        "k_sorted": d["k_sorted"],
        "group_members_no_ref": d["group_members"].tolist(),
        "pairids": [int(x) for x in d["pairids"].tolist()],
    }


def write_submissions(
    pairids: List[int],
    ranked50: np.ndarray,
    ranked_group3: np.ndarray,
    out_dir: Path,
    name: str,
) -> Dict[str, Path]:
    recall = {"version": "rc2", "metric": "recall"}
    subset = {"version": "rc2", "metric": "recall_subset"}
    for i, pid in enumerate(pairids):
        recall[str(int(pid))] = [str(x) for x in ranked50[i, :50].tolist()]
        subset[str(int(pid))] = [str(x) for x in ranked_group3[i, :3].tolist()]
    out_dir.mkdir(parents=True, exist_ok=True)
    p_rec = out_dir / f"recall_submission_{name}.json"
    p_sub = out_dir / f"recall_subset_submission_{name}.json"
    p_rec.write_text(json.dumps(recall, sort_keys=True), encoding="utf-8")
    p_sub.write_text(json.dumps(subset, sort_keys=True), encoding="utf-8")
    return {"recall": p_rec, "recall_subset": p_sub}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["cqm", "cr", "fuse", "all"], default="all")
    p.add_argument("--topk_pt", type=str, default=str(DEFAULT_TOPK))
    p.add_argument("--k_rerank", type=int, default=200)
    p.add_argument("--cqm_ckpt", type=str, default=str(DEFAULT_COMPOSER))
    p.add_argument("--coarse_npz", type=str, default=str(DEFAULT_COARSE_NPZ))
    p.add_argument("--stage1_path", type=str, default=str(DEFAULT_STAGE1))
    p.add_argument("--stage2_path", type=str, default=str(DEFAULT_STAGE2))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--submission_name", type=str, default="hbi_cqm_cr_fusion")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lambda_1", type=float, default=LAMBDA_1)
    p.add_argument("--lambda_2", type=float, default=LAMBDA_2)
    p.add_argument("--lambda_3", type=float, default=LAMBDA_3)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cqm_npz = out_dir / "cqm_itc_scores.npz"
    cr_npz = out_dir / "cr_nlvr_scores.npz"

    if args.only in ("cqm", "all"):
        payload = run_cqm_test1(args)
        _save_scores(cqm_npz, payload)
        logger.info("Saved cqm %s n=%d", cqm_npz, len(payload["pairids"]))
        _free_cuda()
        if args.only == "cqm":
            return

    if args.only in ("cr", "all"):
        payload = run_cr_test1(
            Path(args.topk_pt), Path(args.stage1_path), Path(args.stage2_path), args.k_rerank, args.num_workers
        )
        _save_scores(cr_npz, payload)
        logger.info("Saved CR %s n=%d", cr_npz, len(payload["pairids"]))
        _free_cuda()
        if args.only == "cr":
            return

    comp = _load_scores(cqm_npz)
    cr = _load_scores(cr_npz)
    if comp["pairids"] != cr["pairids"]:
        raise RuntimeError("CQM/CR pairid order mismatch")
    if not np.array_equal(comp["k_sorted"].astype(str), cr["k_sorted"].astype(str)):
        raise RuntimeError("CQM/CR top-50 name order mismatch")

    hbi = hbi_scores_test1(
        Path(args.coarse_npz), comp["pairids"], comp["k_sorted"], comp["group_members_no_ref"]
    )
    a, b, g = args.lambda_1, args.lambda_2, args.lambda_3
    if abs(a + b + g - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {a}+{b}+{g}")

    fused = a * zscore_rows(hbi["pred_logits"]) + b * zscore_rows(comp["pred_logits"]) + g * zscore_rows(cr["pred_logits"])
    fused_g = (
        a * zscore_rows(hbi["group_logits"])
        + b * zscore_rows(comp["group_logits"])
        + g * zscore_rows(cr["group_logits"])
    )
    idx = np.argsort(-fused, axis=1)
    ranked50 = np.take_along_axis(comp["k_sorted"], idx, axis=1)
    g_idx = np.argsort(-fused_g, axis=1)
    g_arr = np.array(comp["group_members_no_ref"], dtype=object)
    ranked_g = np.take_along_axis(g_arr, g_idx, axis=1)

    sub_dir = ROOT / "output/cirr_vision_desc/submission_CIRR"
    paths = write_submissions(comp["pairids"], ranked50, ranked_g, sub_dir, args.submission_name)
    meta = {
        "split": "test1",
        "n_queries": len(comp["pairids"]),
        "k_rerank": args.k_rerank,
        "weights_zscore": {"hbi": a, "cqm": b, "cr": g},
        "weights_source": "val grid-search best (not retuned on test1)",
        "hbi_missing_rows": int(hbi["missing"]),
        "topk_pt": args.topk_pt,
        "coarse_npz": args.coarse_npz,
        "cqm_ckpt": args.cqm_ckpt,
        "stage2_path": args.stage2_path,
        "note": "CIRR test1 labels are hidden; upload the two JSON files to the official server.",
        "submission": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "test1_fusion_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s and %s  n=%d", paths["recall"], paths["recall_subset"], len(comp["pairids"]))


if __name__ == "__main__":
    main()
