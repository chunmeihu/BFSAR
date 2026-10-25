#!/usr/bin/env python3
"""Fashion-IQ Original-Split (large gallery) fusion at coarse K=200/300/500.

Hyperparameter: rerank pool size K. Scores are sliced from an already-scored
Top-500 dump (Composer+CR on OpenCLIP dual coarse). Per-query z-score is
recomputed on the sliced pool, then frozen fusion 0.25/0.30/0.45 is applied.

Gallery = split.{cat}.val.json (official Original-Split). Ranking protocol
matches eval_fiq_hbi_composer_cr_fusion._full_metrics (drop reference, merge
Top-K rerank head with the remaining coarse gallery).

CPU-only. Does not overwrite the source Top-500 Composer/CR npz files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.fiq.vision_desc.eval_fiq_hbi_composer_cr_fusion import (  # noqa: E402
    FROZEN,
    _full_metrics,
    _hbi_pool_scores,
    _load_score,
    _score_npz,
    pool_from_scores,
    zscore_rows,
)

CATS = ("dress", "shirt", "toptee")
RECALL_KEYS = (
    "recall_at10",
    "recall_at50",
    "recall_at100",
    "recall_at200",
    "recall_at300",
    "recall_at400",
    "recall_at500",
)
VALSPLIT_PAPER = {
    "dress": (58.45, 78.92),
    "shirt": (64.08, 83.25),
    "toptee": (67.18, 83.86),
    "mean": (63.24, 82.01),
}


def _slice(comp: Dict[str, Any], k: int) -> Dict[str, Any]:
    return {
        "pred_logits": np.asarray(comp["pred_logits"])[:, :k],
        "k_sorted": np.asarray(comp["k_sorted"])[:, :k],
        "labels": np.asarray(comp["labels"])[:, :k],
        "reference_names": comp["reference_names"],
        "target_names": comp["target_names"],
    }


def _grid_best(h, c, r, k_sorted, sim, index_names, refs, tgts, step: float) -> Dict[str, Any]:
    best_r10 = None
    best_joint = None
    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    for aa in alphas:
        for bb in alphas:
            if aa + bb > 1.0 + 1e-9:
                continue
            gg = float(np.round(1.0 - aa - bb, 4))
            m = _full_metrics(aa * h + bb * c + gg * r, k_sorted, sim, index_names, refs, tgts)
            rec = {
                "alpha_coarse": float(aa),
                "beta_composer": float(bb),
                "gamma_cr": gg,
                **m,
            }
            if best_r10 is None or rec["recall_at10"] > best_r10["recall_at10"]:
                best_r10 = rec
            jkey = (rec["recall_at10"] + rec["recall_at50"], rec["recall_at10"], rec["recall_at50"])
            if best_joint is None or jkey > (
                best_joint["recall_at10"] + best_joint["recall_at50"],
                best_joint["recall_at10"],
                best_joint["recall_at50"],
            ):
                best_joint = rec
    grid_best = best_r10
    if (
        best_joint is not None
        and best_r10 is not None
        and best_joint["recall_at10"] >= best_r10["recall_at10"] - 0.3
        and (best_joint["recall_at10"] + best_joint["recall_at50"])
        > (best_r10["recall_at10"] + best_r10["recall_at50"])
    ):
        grid_best = best_joint
    return {"grid_best_r10": grid_best, "grid_best_r10_only": best_r10, "grid_best_joint": best_joint}


def _mean_from_per_cat(per_cat: Dict[str, Any], cats: Sequence[str], path_keys: Sequence[Any]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for cat in cats:
        cur: Any = per_cat[cat]
        for key in path_keys[:-1]:
            cur = cur[key]
        row = cur if path_keys[-1] is None else cur[path_keys[-1]]
        for mk, mv in row.items():
            if isinstance(mv, (int, float)) and (
                mk.startswith("recall") or mk.startswith("pool") or mk == "coverage"
            ):
                acc.setdefault(mk, []).append(float(mv))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fuse_one_k(
    score_dir: Path,
    k: int,
    cats: Sequence[str],
    grid_step: float,
    do_grid: bool,
) -> Dict[str, Any]:
    a, b, g = FROZEN
    per_cat: Dict[str, Any] = {}
    for cat in cats:
        comp_full = _load_score(_score_npz(score_dir, "composer", cat))
        cr_full = _load_score(_score_npz(score_dir, "cr", cat))
        k_max = int(comp_full["k_sorted"].shape[1])
        if k > k_max:
            raise ValueError(f"{cat}: K={k} > scored pool {k_max}")
        if not np.array_equal(comp_full["k_sorted"].astype(str), cr_full["k_sorted"].astype(str)):
            raise RuntimeError(f"{cat}: composer/CR top-k order mismatch")
        comp = _slice(comp_full, k)
        cr = _slice(cr_full, k)
        s_hbi, sim, index_names, refs, tgts = _hbi_pool_scores(
            score_dir, cat, comp["k_sorted"], comp["reference_names"], comp["target_names"],
        )
        labels = comp["labels"]
        s_comp = comp["pred_logits"]
        s_cr = cr["pred_logits"]
        h = zscore_rows(s_hbi)
        c = zscore_rows(s_comp)
        r = zscore_rows(s_cr)
        fused = a * h + b * c + g * r
        singles_full = {
            "coarse": _full_metrics(s_hbi, comp["k_sorted"], sim, index_names, refs, tgts),
            "composer": _full_metrics(s_comp, comp["k_sorted"], sim, index_names, refs, tgts),
            "cr": _full_metrics(s_cr, comp["k_sorted"], sim, index_names, refs, tgts),
        }
        frozen_full = _full_metrics(fused, comp["k_sorted"], sim, index_names, refs, tgts)
        singles_pool = {
            "coarse": pool_from_scores(s_hbi, labels),
            "composer": pool_from_scores(s_comp, labels),
            "cr": pool_from_scores(s_cr, labels),
            "frozen_0.25_0.30_0.45": pool_from_scores(fused, labels),
        }
        grid = _grid_best(h, c, r, comp["k_sorted"], sim, index_names, refs, tgts, grid_step) if do_grid else {}
        per_cat[cat] = {
            "n_queries": len(refs),
            "n_gallery": len(index_names),
            "coverage_at_k": float(np.asarray(labels).any(axis=1).mean() * 100.0),
            "singles_full": singles_full,
            "frozen_0.25_0.30_0.45_full": frozen_full,
            "singles_pool": singles_pool,
            **grid,
        }
        m = frozen_full
        print(
            f"Original-Split K={k} {cat}: gallery={len(index_names)} cov@K={per_cat[cat]['coverage_at_k']:.2f} "
            f"frozen R@10/50={m['recall_at10']:.2f}/{m['recall_at50']:.2f} "
            f"R@200/300/500={m['recall_at200']:.2f}/{m['recall_at300']:.2f}/{m['recall_at500']:.2f}",
            flush=True,
        )

    mean = {
        "coarse": _mean_from_per_cat(per_cat, cats, ["singles_full", "coarse"]),
        "composer": _mean_from_per_cat(per_cat, cats, ["singles_full", "composer"]),
        "cr": _mean_from_per_cat(per_cat, cats, ["singles_full", "cr"]),
        "frozen_0.25_0.30_0.45": _mean_from_per_cat(per_cat, cats, ["frozen_0.25_0.30_0.45_full"]),
        "pool": {
            "coarse": _mean_from_per_cat(per_cat, cats, ["singles_pool", "coarse"]),
            "composer": _mean_from_per_cat(per_cat, cats, ["singles_pool", "composer"]),
            "cr": _mean_from_per_cat(per_cat, cats, ["singles_pool", "cr"]),
            "frozen_0.25_0.30_0.45": _mean_from_per_cat(per_cat, cats, ["singles_pool", "frozen_0.25_0.30_0.45"]),
        },
    }
    if do_grid:
        mean["grid_best_r10"] = _mean_from_per_cat(per_cat, cats, ["grid_best_r10"])
    fm = mean["frozen_0.25_0.30_0.45"]
    print(
        f"Original-Split K={k} MEAN frozen R@10/50={fm['recall_at10']:.2f}/{fm['recall_at50']:.2f} "
        f"R@200/300/500={fm.get('recall_at200', 0):.2f}/{fm.get('recall_at300', 0):.2f}/{fm.get('recall_at500', 0):.2f}",
        flush=True,
    )
    return {
        "k_save": k,
        "k_rerank": k,
        "gallery": "Original-Split = split.{cat}.val.json (official large gallery)",
        "protocol": "drop reference, merge Top-K rerank with remaining coarse gallery",
        "fusion_frozen": {
            "alpha_coarse": a,
            "beta_composer": b,
            "gamma_cr": g,
            "norm": "zscore-per-query on sliced Top-K",
        },
        "coarse": "OpenCLIP dual Path_ref+Path_vis α=0.4 β=0.2 w_ref=0.5",
        "score_source": str(score_dir),
        "note": "Composer/CR logits sliced from Top-500 dump; no GPU rescoring.",
        "mean": mean,
        "per_category": per_cat,
    }


def _frozen_row(report: Dict[str, Any], cat: str | None) -> Dict[str, float]:
    if cat is None:
        src = report["mean"]["frozen_0.25_0.30_0.45"]
        cov = float(np.mean([report["per_category"][c]["coverage_at_k"] for c in CATS]))
    else:
        src = report["per_category"][cat]["frozen_0.25_0.30_0.45_full"]
        cov = report["per_category"][cat]["coverage_at_k"]
    out = {"coverage_at_k": cov}
    for key in RECALL_KEYS:
        if key in src:
            out[key] = float(src[key])
    return out


def build_summary(by_k: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for k, report in by_k.items():
        compact[str(k)] = {
            "mean": _frozen_row(report, None),
            "per_category": {cat: _frozen_row(report, cat) for cat in CATS},
            "n_gallery": {cat: report["per_category"][cat]["n_gallery"] for cat in CATS},
        }
    return compact


def maybe_compare_valsplit(valsplit_json: Path, compact: Dict[str, Any], out_path: Path) -> None:
    if not valsplit_json.is_file():
        return
    small = json.loads(valsplit_json.read_text(encoding="utf-8"))
    rows = []
    for k in compact:
        s = small.get("by_k", {}).get(k, {})
        s_mean = s.get("mean", {}).get("mcot_no_drop", {})
        l_mean = compact[k]["mean"]
        rows.append({
            "k": int(k),
            "small_valsplit_mean_r10": s_mean.get("recall_at10"),
            "small_valsplit_mean_r50": s_mean.get("recall_at50"),
            "large_originalsplit_mean_r10": l_mean.get("recall_at10"),
            "large_originalsplit_mean_r50": l_mean.get("recall_at50"),
            "large_originalsplit_mean_r200": l_mean.get("recall_at200"),
            "large_originalsplit_mean_r300": l_mean.get("recall_at300"),
            "large_originalsplit_mean_r500": l_mean.get("recall_at500"),
            "large_originalsplit_cov_at_k": l_mean.get("coverage_at_k"),
        })
    report = {
        "small_gallery": small.get("gallery"),
        "large_gallery": "Original-Split split.{cat}.val.json",
        "fusion": "frozen 0.25/0.30/0.45 zscore-per-query on coarse Top-K",
        "mcot_paper_valsplit": VALSPLIT_PAPER,
        "rows": rows,
        "by_k": {
            k: {
                "small": small.get("by_k", {}).get(k, {}),
                "large": compact[k],
            }
            for k in compact
        },
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Original-Split Top-K hyperparameter sweep (CPU)")
    p.add_argument(
        "--score_dir",
        type=str,
        default=str(ROOT / "output/fiq_rerank/fusion_k500_openclip_dual"),
        help="Existing Original-Split Top-500 dump with composer_*.npz / cr_*.npz / hbi_fused_*.npz",
    )
    p.add_argument("--ks", type=str, default="200,300,500")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "output/fiq_rerank/ksweep_openclip_dual_originalsplit"),
        help="Summary JSON for the K sweep",
    )
    p.add_argument(
        "--per_k_out_root",
        type=str,
        default=str(ROOT / "output/fiq_rerank"),
        help="Writes fusion_k{K}_openclip_dual/fusion_metrics.json for K=200 and K=300",
    )
    p.add_argument("--write_k500_metrics", action="store_true", help="Also overwrite fusion_k500 fusion_metrics.json")
    p.add_argument("--grid_step", type=float, default=0.1)
    p.add_argument("--no_grid", action="store_true")
    p.add_argument(
        "--valsplit_json",
        type=str,
        default=str(ROOT / "output/fiq_rerank/fusion_k500_openclip_dual_valsplit/valsplit_k200_300_500.json"),
    )
    args = p.parse_args()

    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    per_k_root = Path(args.per_k_out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    do_grid = not args.no_grid

    by_k: Dict[int, Dict[str, Any]] = {}
    for k in ks:
        report = fuse_one_k(score_dir, k, CATS, args.grid_step, do_grid)
        by_k[k] = report
        if k in (200, 300) or (k == 500 and args.write_k500_metrics):
            dest = per_k_root / f"fusion_k{k}_openclip_dual"
            dest.mkdir(parents=True, exist_ok=True)
            path = dest / "fusion_metrics.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {path}", flush=True)

    compact = build_summary(by_k)
    summary = {
        "gallery": "Original-Split = split.{cat}.val.json (official large gallery)",
        "protocol": "drop reference, merge Top-K rerank with remaining coarse gallery",
        "fusion": "frozen 0.25/0.30/0.45 zscore-per-query on sliced Top-K",
        "coarse": "OpenCLIP dual Path_ref+Path_vis α=0.4 β=0.2 w_ref=0.5",
        "score_source": str(score_dir),
        "hyperparameter": "coarse Top-K pool size for Composer+CR fusion",
        "by_k": compact,
    }
    summary_path = out_dir / "originalsplit_k200_300_500.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)

    maybe_compare_valsplit(
        Path(args.valsplit_json),
        compact,
        out_dir / "compare_valsplit_vs_originalsplit_k200_300_500.json",
    )


if __name__ == "__main__":
    main()
