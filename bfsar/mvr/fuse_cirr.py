#!/usr/bin/env python3
"""CIRR multi-view reranking: coarse DPIC + CQM + CVS.

Per-query z-score, then
    S = 0.25 * S_CRS + 0.40 * S_CQM + 0.35 * S_CVS
on the top-200 pool.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr import cirr_paths
from bfsar.mvr.export_topk import load_coarse_sim_npz
from bfsar.mvr.cvs_infer import (
    DEFAULT_STAGE1,
    DEFAULT_STAGE2,
    align_gallery_index_to_cirr,
    build_full_val_topk_pt,
    build_test1_topk_pt,
    metrics_stage2_pool,
    run_blip_stage2_rerank,
    save_topk_pt,
)
logger = logging.getLogger("bfsar_fuse_cirr")

DEFAULT_TOPK = ROOT / "output/mvr/cirr/top200_val.pt"
DEFAULT_COMPOSER = ROOT / "weights/cqm/cirr/cqm.pt"
DEFAULT_COARSE_NPZ = ROOT / "output/dpic/cirr/val/coarse_sim.npz"
DEFAULT_OUT = ROOT / "output/mvr/cirr"
CR_BASELINE = 53.36


def _recover_k_sorted(reranked_topk: np.ndarray, rerank_idx: np.ndarray) -> np.ndarray:
    inv = np.argsort(rerank_idx, axis=1)
    return np.take_along_axis(reranked_topk, inv, axis=1)


def _as_str_arr(names) -> np.ndarray:
    return np.asarray([str(x) for x in names], dtype=object)


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def minmax_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    den = np.where((hi - lo) < 1e-6, 1.0, hi - lo)
    return (x - lo) / den


def pool_from_scores(
    scores: np.ndarray,
    k_sorted: np.ndarray,
    target_names: List[str],
    group_scores: Optional[np.ndarray],
    group_members: List[List[str]],
) -> Dict[str, float]:
    rerank_idx = np.argsort(-scores, axis=1)
    labels = (_as_str_arr(k_sorted.reshape(-1)).reshape(k_sorted.shape) == np.asarray(target_names).reshape(-1, 1))
    if group_scores is None or not group_members:
        dummy_g = np.asarray([[""] * 5] * len(target_names), dtype=object)
        return metrics_stage2_pool(rerank_idx, labels, dummy_g, target_names)
    g_idx = np.argsort(-group_scores, axis=1)
    g_arr = np.array(group_members, dtype=object)
    reranked_g = np.take_along_axis(g_arr, g_idx, axis=1)
    return metrics_stage2_pool(rerank_idx, labels, reranked_g, target_names)


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _file_split(args) -> str:
    return "test1" if getattr(args, "split", "val") in ("test", "test1") else "val"


def compute_cqm(args) -> Dict[str, Any]:
    from bfsar.mvr.cqm_rerank_cirr import run_cirr_blip2_qtr_rerank

    file_split = _file_split(args)
    logger.info("CQM ITC: %s split=%s", args.cqm_ckpt, file_split)
    out = run_cirr_blip2_qtr_rerank(
        Path(args.topk_pt),
        split=file_split,
        k_rerank=args.k_rerank,
        ckpt_path=Path(args.cqm_ckpt),
        local_files_only=False,
        max_queries=args.max_queries,
        num_workers=args.num_workers,
        score_mode="itc",
    )
    return {
        "pred_logits": np.asarray(out["pred_logits"], dtype=np.float32),
        "target_names": list(out["target_names"]),
        "reference_names": list(out["reference_names"]),
        "group_members_no_ref": out["group_members_no_ref"],
        "k_sorted": _recover_k_sorted(out["reranked_topk"], out["rerank_idx"]),
        "pool_metrics": out.get("pool_metrics") or {},
        "group_logits": np.asarray(out["group_logits"], dtype=np.float32),
        "pairids": [int(x) for x in out.get("pairids") or []],
    }


def compute_cr(args) -> Dict[str, Any]:
    file_split = _file_split(args)
    logger.info("CR NLVR Stage-2: %s split=%s", args.stage2_path, file_split)
    out = run_blip_stage2_rerank(
        Path(args.topk_pt),
        Path(args.stage1_path),
        Path(args.stage2_path),
        k_rerank=args.k_rerank,
        max_queries=args.max_queries,
        num_workers=args.num_workers,
        split=file_split,
    )
    return {
        "pred_logits": np.asarray(out["pred_logits"], dtype=np.float32),
        "target_names": list(out["target_names"]),
        "reference_names": list(out["reference_names"]),
        "group_members_no_ref": out["group_members_no_ref"],
        "k_sorted": _recover_k_sorted(out["reranked_topk"], out["rerank_idx"]),
        "group_logits": np.asarray(out["group_logits"], dtype=np.float32),
        "pairids": [int(x) for x in out.get("pairids") or []],
    }


def compute_coarse(
    coarse_npz: Path,
    reference_names: List[str],
    target_names: List[str],
    k_sorted: np.ndarray,
    group_members: List[List[str]],
    split: str = "val",
    pairids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    gallery_split = "test1" if split in ("test", "test1") else "val"
    sim, coarse_refs, coarse_tgts, index_names = load_coarse_sim_npz(str(coarse_npz))
    sim, index_names = align_gallery_index_to_cirr(sim, index_names, split=gallery_split)
    name_to_j = {str(n): i for i, n in enumerate(index_names)}
    n, k = k_sorted.shape
    scores = np.zeros((n, k), dtype=np.float32)
    g_len = max(len(g) for g in group_members) if group_members else 5
    g_scores = np.zeros((n, g_len), dtype=np.float32)
    missing = 0
    if gallery_split == "test1":
        stored = np.load(coarse_npz, allow_pickle=True)
        coarse_pids = [int(x) for x in stored["pairids"].tolist()]
        pid_to_row = {pid: i for i, pid in enumerate(coarse_pids)}
        if not pairids:
            raise ValueError("CIRR test fusion needs pairids from the reranker")
        row_of = [pid_to_row.get(int(pid)) for pid in pairids]
    else:
        key_to_row = {cirr_paths.pair_key(r, t): i for i, (r, t) in enumerate(zip(coarse_refs, coarse_tgts))}
        row_of = [
            key_to_row.get(cirr_paths.pair_key(str(reference_names[i]), str(target_names[i])))
            for i in range(n)
        ]
    for i, row_i in enumerate(row_of):
        if row_i is None:
            scores[i] = np.linspace(1.0, 0.0, k, dtype=np.float32)
            missing += 1
            continue
        row = sim[row_i]
        cols = [name_to_j[str(nm)] for nm in k_sorted[i]]
        scores[i] = row[cols]
        gcols = [name_to_j[str(nm)] for nm in group_members[i]]
        g_scores[i, : len(gcols)] = row[gcols]
    return scores, g_scores, missing


def _save_npz(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_logits=payload["pred_logits"],
        target_names=np.asarray(payload["target_names"], dtype=object),
        reference_names=np.asarray(payload["reference_names"], dtype=object),
        k_sorted=np.asarray(payload["k_sorted"], dtype=object),
        group_members=np.asarray(payload["group_members_no_ref"], dtype=object),
        group_logits=np.asarray(payload.get("group_logits"), dtype=np.float32)
        if payload.get("group_logits") is not None else np.zeros((0, 0), dtype=np.float32),
        pairids=np.asarray(payload.get("pairids") or [], dtype=np.int64),
    )


def _load_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {
        "pred_logits": d["pred_logits"].astype(np.float32),
        "target_names": [str(x) for x in d["target_names"].tolist()],
        "reference_names": [str(x) for x in d["reference_names"].tolist()],
        "k_sorted": d["k_sorted"],
        "group_members_no_ref": d["group_members"].tolist(),
        "group_logits": d["group_logits"].astype(np.float32) if "group_logits" in d.files and d["group_logits"].size else None,
        "pairids": [int(x) for x in d["pairids"].tolist()] if "pairids" in d.files else [],
    }


def _assert_aligned(a: Dict[str, Any], b: Dict[str, Any], tag: str) -> None:
    if a["reference_names"] != b["reference_names"] or a["target_names"] != b["target_names"]:
        raise RuntimeError(f"Query order mismatch vs {tag}")
    if not np.array_equal(a["k_sorted"].astype(str), b["k_sorted"].astype(str)):
        raise RuntimeError(f"Top-K name order mismatch vs {tag}")


def grid_fusion(
    s_scrs: np.ndarray,
    s_comp: np.ndarray,
    s_cr: np.ndarray,
    k_sorted: np.ndarray,
    target_names: List[str],
    group_members: List[List[str]],
    g_scrs: Optional[np.ndarray],
    g_comp: Optional[np.ndarray],
    g_cr: Optional[np.ndarray],
    norm: str,
    step: float,
) -> Dict[str, Any]:
    norm_fn = zscore_rows if norm == "zscore" else minmax_rows
    h = norm_fn(s_scrs)
    c = norm_fn(s_comp)
    r = norm_fn(s_cr)
    gh = norm_fn(g_scrs) if g_scrs is not None else None
    gc_ = norm_fn(g_comp) if g_comp is not None else None
    gr = norm_fn(g_cr) if g_cr is not None else None

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    rows = []
    best = None
    for a in alphas:
        for b in alphas:
            if a + b > 1.0 + 1e-9:
                continue
            g = float(np.round(1.0 - a - b, 4))
            fused = a * h + b * c + g * r
            g_fused = None
            if gh is not None and gc_ is not None and gr is not None:
                g_fused = a * gh + b * gc_ + g * gr
            m = pool_from_scores(fused, k_sorted, target_names, g_fused, group_members)
            rec = {
                "lambda_1": float(a),
                "lambda_2": float(b),
                "lambda_3": g,
                **{k: float(v) for k, v in m.items()},
            }
            rows.append(rec)
            if best is None or rec["pool_recall_at1"] > best["pool_recall_at1"]:
                best = rec
    singles = {
        "coarse": pool_from_scores(s_scrs, k_sorted, target_names, g_scrs, group_members),
        "cqm": pool_from_scores(s_comp, k_sorted, target_names, g_comp, group_members),
        "cr": pool_from_scores(s_cr, k_sorted, target_names, g_cr, group_members),
    }
    return {"norm": norm, "step": step, "singles": singles, "best": best, "grid": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["cqm", "cr", "fuse", "all"], default="all")
    p.add_argument("--topk_pt", type=str, default=str(DEFAULT_TOPK))
    p.add_argument("--k_rerank", type=int, default=200)
    p.add_argument("--lambda_1", type=float, default=0.25)
    p.add_argument("--lambda_2", type=float, default=0.40)
    p.add_argument("--lambda_3", type=float, default=0.35)
    p.add_argument("--cqm_ckpt", type=str, default=str(DEFAULT_COMPOSER))
    p.add_argument("--coarse_npz", type=str, default=str(DEFAULT_COARSE_NPZ))
    p.add_argument("--stage1_path", type=str, default=str(DEFAULT_STAGE1))
    p.add_argument("--stage2_path", type=str, default=str(DEFAULT_STAGE2))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--submission_name", type=str, default="bfsar")
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def _write_test_submission(args, out_dir, comp, cr, g_scrs, fused_scores, a, b, g, n_miss) -> None:
    import zipfile

    if comp.get("pairids") != cr.get("pairids"):
        raise RuntimeError("CQM and CVS pairid order differ")
    if comp.get("group_logits") is None or cr.get("group_logits") is None:
        raise RuntimeError("Test submission needs group logits from CQM and CVS")
    gh = zscore_rows(g_scrs)
    gc = zscore_rows(comp["group_logits"])
    gr = zscore_rows(cr["group_logits"])
    fused_g = a * gh + b * gc + g * gr
    order = np.argsort(-fused_scores, axis=1)
    ranked = np.take_along_axis(np.asarray(comp["k_sorted"], dtype=object), order, axis=1)
    g_order = np.argsort(-fused_g, axis=1)
    groups = np.asarray(comp["group_members_no_ref"], dtype=object)
    ranked_g = np.take_along_axis(groups, g_order, axis=1)
    recall = {"version": "rc2", "metric": "recall"}
    subset = {"version": "rc2", "metric": "recall_subset"}
    for i, pid in enumerate(comp["pairids"]):
        recall[str(int(pid))] = [str(x) for x in ranked[i, :50].tolist()]
        subset[str(int(pid))] = [str(x) for x in ranked_g[i, :3].tolist()]
    name = args.submission_name
    rec_path = out_dir / f"recall_submission_{name}.json"
    sub_path = out_dir / f"recall_subset_submission_{name}.json"
    rec_path.write_text(json.dumps(recall, sort_keys=True), encoding="utf-8")
    sub_path.write_text(json.dumps(subset, sort_keys=True), encoding="utf-8")
    zip_path = out_dir / f"{name}_cirr_test1.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(rec_path, arcname=rec_path.name)
        zf.write(sub_path, arcname=sub_path.name)
    meta = {
        "split": "test1",
        "n_queries": len(comp["pairids"]),
        "k_rerank": int(args.k_rerank),
        "lambda_1": a,
        "lambda_2": b,
        "lambda_3": g,
        "coarse_missing_rows": n_miss,
        "recall_json": str(rec_path),
        "recall_subset_json": str(sub_path),
        "zip": str(zip_path),
    }
    (out_dir / "submission_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("CIRR test1 submission -> %s", zip_path)


def ensure_topk(args) -> None:
    coarse = Path(args.coarse_npz)
    file_split = _file_split(args)
    if not coarse.is_file():
        flag = "test" if file_split == "test1" else "val"
        raise FileNotFoundError(
            f"Missing coarse scores: {coarse}\n"
            f"Run the coarse step first: bash scripts/eval_dpic_cirr.sh --split {flag}"
        )
    topk = Path(args.topk_pt)
    if topk.is_file():
        return
    if file_split == "test1":
        payload = build_test1_topk_pt(coarse, k_save=int(args.k_rerank))
    else:
        sim, refs, tgts, index_names = load_coarse_sim_npz(str(coarse))
        payload = build_full_val_topk_pt(
            sim, refs, tgts, index_names, k_save=int(args.k_rerank)
        )
    save_topk_pt(payload, topk)
    logger.info("Built top-%d %s from %s", args.k_rerank, file_split, coarse)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    ensure_topk(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cqm_npz = out_dir / "cqm_itc_scores.npz"
    cr_npz = out_dir / "cr_nlvr_scores.npz"

    if args.only in ("cqm", "all"):
        payload = compute_cqm(args)
        _save_npz(cqm_npz, payload)
        logger.info("Saved %s  pool=%s", cqm_npz, payload.get("pool_metrics"))
        _free_cuda()
        if args.only == "cqm":
            return

    if args.only in ("cr", "all"):
        payload = compute_cr(args)
        _save_npz(cr_npz, payload)
        logger.info("Saved %s", cr_npz)
        _free_cuda()
        if args.only == "cr":
            return

    if not cqm_npz.is_file() or not cr_npz.is_file():
        raise FileNotFoundError(f"Need {cqm_npz} and {cr_npz} before fuse")

    comp = _load_npz(cqm_npz)
    cr = _load_npz(cr_npz)
    _assert_aligned(comp, cr, "CR")

    s_scrs, g_scrs, n_miss = compute_coarse(
        Path(args.coarse_npz),
        comp["reference_names"],
        comp["target_names"],
        comp["k_sorted"],
        comp["group_members_no_ref"],
        split=args.split,
        pairids=comp.get("pairids"),
    )
    logger.info("BFSAR scores aligned; missing_sim_rows=%d / %d", n_miss, len(comp["reference_names"]))

    a, b, g = float(args.lambda_1), float(args.lambda_2), float(args.lambda_3)
    h = zscore_rows(s_scrs)
    c = zscore_rows(comp["pred_logits"])
    r = zscore_rows(cr["pred_logits"])
    fused_scores = a * h + b * c + g * r
    if _file_split(args) == "test1":
        _write_test_submission(args, out_dir, comp, cr, g_scrs, fused_scores, a, b, g, n_miss)
        return

    report: Dict[str, Any] = {
        "k_rerank": args.k_rerank,
        "n_queries": len(comp["target_names"]),
        "coarse_missing_rows": n_miss,
        "cr_baseline_paper": CR_BASELINE,
        "cqm_ckpt": args.cqm_ckpt,
        "stage2_path": args.stage2_path,
        "topk_pt": args.topk_pt,
    }
    metrics = pool_from_scores(
        fused_scores,
        comp["k_sorted"],
        comp["target_names"],
        None,
        comp["group_members_no_ref"],
    )
    report["lambda_1"] = a
    report["lambda_2"] = b
    report["lambda_3"] = g
    report["fused"] = {k: float(v) for k, v in metrics.items()}
    out_json = out_dir / "fusion_metrics.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Fused λ=(%.2f, %.2f, %.2f) %s",
        a, b, g, {k: round(float(v), 2) for k, v in metrics.items()},
    )


if __name__ == "__main__":
    main()
