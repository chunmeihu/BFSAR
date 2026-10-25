#!/usr/bin/env python3
"""Score-fusion on the same HBI Top-50: HBI coarse + Composer ITC + CR NLVR.

Per-query z-score (or min-max) each source, then
    s = a * s_HBI + b * s_Composer + (1-a-b) * s_NLVR
and report pool R@1 vs CR-only 53.36.
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.cirr import cirr_paths
from HBI.cirr.vision_desc.export_hbi_cirr_topk import load_hbi_sim_npz
from HBI.cirr.vision_desc.hbi_blip_rerank import (
    DEFAULT_STAGE1,
    DEFAULT_STAGE2,
    align_gallery_index_to_cirr,
    metrics_stage2_pool,
    run_blip_stage2_rerank,
)
from HBI.rerank.paths import DEFAULT_CIRR_TOPK_VAL

logger = logging.getLogger("hbi_composer_cr_fusion")

DEFAULT_TOPK = DEFAULT_CIRR_TOPK_VAL
DEFAULT_COMPOSER = ROOT / "output/cirr_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r1.pt"
DEFAULT_HBI_NPZ = ROOT / "output/cirr_vision_desc/hbi_triple_removed_best_sim.npz"
DEFAULT_OUT = ROOT / "output/cirr_rerank/fusion_hbi_composer_cr"
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


def compute_composer(args) -> Dict[str, Any]:
    from HBI.cirr.vision_desc.cirr_blip2_qtr_rerank import run_cirr_blip2_qtr_rerank

    logger.info("Composer ITC: %s", args.composer_ckpt)
    out = run_cirr_blip2_qtr_rerank(
        Path(args.topk_pt),
        split="val",
        k_rerank=args.k_rerank,
        ckpt_path=Path(args.composer_ckpt),
        local_files_only=True,
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
    }


def compute_cr(args) -> Dict[str, Any]:
    logger.info("CR NLVR Stage-2: %s", args.stage2_path)
    out = run_blip_stage2_rerank(
        Path(args.topk_pt),
        Path(args.stage1_path),
        Path(args.stage2_path),
        k_rerank=args.k_rerank,
        max_queries=args.max_queries,
        num_workers=args.num_workers,
    )
    return {
        "pred_logits": np.asarray(out["pred_logits"], dtype=np.float32),
        "target_names": list(out["target_names"]),
        "reference_names": list(out["reference_names"]),
        "group_members_no_ref": out["group_members_no_ref"],
        "k_sorted": _recover_k_sorted(out["reranked_topk"], out["rerank_idx"]),
    }


def compute_hbi(
    hbi_npz: Path,
    reference_names: List[str],
    target_names: List[str],
    k_sorted: np.ndarray,
    group_members: List[List[str]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    sim, hbi_refs, hbi_tgts, index_names = load_hbi_sim_npz(str(hbi_npz))
    sim, index_names = align_gallery_index_to_cirr(sim, index_names, split="val")
    name_to_j = {str(n): i for i, n in enumerate(index_names)}
    key_to_row = {cirr_paths.pair_key(r, t): i for i, (r, t) in enumerate(zip(hbi_refs, hbi_tgts))}
    n, k = k_sorted.shape
    scores = np.zeros((n, k), dtype=np.float32)
    g_len = max(len(g) for g in group_members) if group_members else 5
    g_scores = np.zeros((n, g_len), dtype=np.float32)
    missing = 0
    for i in range(n):
        key = cirr_paths.pair_key(str(reference_names[i]), str(target_names[i]))
        if key not in key_to_row:
            scores[i] = np.linspace(1.0, 0.0, k, dtype=np.float32)
            missing += 1
            continue
        row = sim[key_to_row[key]]
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
    )


def _load_npz(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {
        "pred_logits": d["pred_logits"].astype(np.float32),
        "target_names": [str(x) for x in d["target_names"].tolist()],
        "reference_names": [str(x) for x in d["reference_names"].tolist()],
        "k_sorted": d["k_sorted"],
        "group_members_no_ref": d["group_members"].tolist(),
    }


def _assert_aligned(a: Dict[str, Any], b: Dict[str, Any], tag: str) -> None:
    if a["reference_names"] != b["reference_names"] or a["target_names"] != b["target_names"]:
        raise RuntimeError(f"Query order mismatch vs {tag}")
    if not np.array_equal(a["k_sorted"].astype(str), b["k_sorted"].astype(str)):
        raise RuntimeError(f"Top-K name order mismatch vs {tag}")


def grid_fusion(
    s_hbi: np.ndarray,
    s_comp: np.ndarray,
    s_cr: np.ndarray,
    k_sorted: np.ndarray,
    target_names: List[str],
    group_members: List[List[str]],
    g_hbi: Optional[np.ndarray],
    g_comp: Optional[np.ndarray],
    g_cr: Optional[np.ndarray],
    norm: str,
    step: float,
) -> Dict[str, Any]:
    norm_fn = zscore_rows if norm == "zscore" else minmax_rows
    h = norm_fn(s_hbi)
    c = norm_fn(s_comp)
    r = norm_fn(s_cr)
    gh = norm_fn(g_hbi) if g_hbi is not None else None
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
                "alpha_hbi": float(a),
                "beta_composer": float(b),
                "gamma_cr": g,
                **{k: float(v) for k, v in m.items()},
            }
            rows.append(rec)
            if best is None or rec["pool_recall_at1"] > best["pool_recall_at1"]:
                best = rec
    singles = {
        "hbi": pool_from_scores(s_hbi, k_sorted, target_names, g_hbi, group_members),
        "composer": pool_from_scores(s_comp, k_sorted, target_names, g_comp, group_members),
        "cr": pool_from_scores(s_cr, k_sorted, target_names, g_cr, group_members),
    }
    return {"norm": norm, "step": step, "singles": singles, "best": best, "grid": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["composer", "cr", "fuse", "all"], default="all")
    p.add_argument("--topk_pt", type=str, default=str(DEFAULT_TOPK))
    p.add_argument("--k_rerank", type=int, default=50)
    p.add_argument("--composer_ckpt", type=str, default=str(DEFAULT_COMPOSER))
    p.add_argument("--hbi_npz", type=str, default=str(DEFAULT_HBI_NPZ))
    p.add_argument("--stage1_path", type=str, default=str(DEFAULT_STAGE1))
    p.add_argument("--stage2_path", type=str, default=str(DEFAULT_STAGE2))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--max_queries", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grid_step", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    composer_npz = out_dir / "composer_itc_scores.npz"
    cr_npz = out_dir / "cr_nlvr_scores.npz"

    if args.only in ("composer", "all"):
        payload = compute_composer(args)
        _save_npz(composer_npz, payload)
        logger.info("Saved %s  pool=%s", composer_npz, payload.get("pool_metrics"))
        _free_cuda()
        if args.only == "composer":
            return

    if args.only in ("cr", "all"):
        payload = compute_cr(args)
        _save_npz(cr_npz, payload)
        logger.info("Saved %s", cr_npz)
        _free_cuda()
        if args.only == "cr":
            return

    if not composer_npz.is_file() or not cr_npz.is_file():
        raise FileNotFoundError(f"Need {composer_npz} and {cr_npz} before fuse")

    comp = _load_npz(composer_npz)
    cr = _load_npz(cr_npz)
    _assert_aligned(comp, cr, "CR")

    s_hbi, g_hbi, n_miss = compute_hbi(
        Path(args.hbi_npz),
        comp["reference_names"],
        comp["target_names"],
        comp["k_sorted"],
        comp["group_members_no_ref"],
    )
    logger.info("HBI scores aligned; missing_sim_rows=%d / %d", n_miss, len(comp["target_names"]))

    report: Dict[str, Any] = {
        "k_rerank": args.k_rerank,
        "n_queries": len(comp["target_names"]),
        "hbi_missing_rows": n_miss,
        "cr_baseline_paper": CR_BASELINE,
        "composer_ckpt": args.composer_ckpt,
        "stage2_path": args.stage2_path,
        "topk_pt": args.topk_pt,
    }
    for norm in ("zscore", "minmax"):
        fused = grid_fusion(
            s_hbi,
            comp["pred_logits"],
            cr["pred_logits"],
            comp["k_sorted"],
            comp["target_names"],
            comp["group_members_no_ref"],
            g_hbi,
            None,
            None,
            norm=norm,
            step=args.grid_step,
        )
        report[norm] = {
            "singles": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in fused["singles"].items()},
            "best": fused["best"],
            "n_grid": len(fused["grid"]),
        }
        logger.info(
            "[%s] singles HBI=%.2f Composer=%.2f CR=%.2f | best fuse R@1=%.2f  a=%.2f b=%.2f g=%.2f  vs CR %.2f  %s",
            norm,
            fused["singles"]["hbi"]["pool_recall_at1"],
            fused["singles"]["composer"]["pool_recall_at1"],
            fused["singles"]["cr"]["pool_recall_at1"],
            fused["best"]["pool_recall_at1"],
            fused["best"]["alpha_hbi"],
            fused["best"]["beta_composer"],
            fused["best"]["gamma_cr"],
            CR_BASELINE,
            "BEATS" if fused["best"]["pool_recall_at1"] > CR_BASELINE + 1e-6 else "NO_BEAT",
        )

    best_z = report["zscore"]["best"]["pool_recall_at1"]
    report["beats_cr_53_36"] = bool(best_z > CR_BASELINE + 1e-6)
    report["delta_vs_cr"] = float(best_z - CR_BASELINE)
    out_json = out_dir / "fusion_metrics.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s  beats=%s  delta=%.2f", out_json, report["beats_cr_53_36"], report["delta_vs_cr"])


if __name__ == "__main__":
    main()
