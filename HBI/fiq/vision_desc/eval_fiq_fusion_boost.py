#!/usr/bin/env python3
"""CPU-only FIQ fusion boosts on existing coarse/Composer/CR score dumps.

Variants (pipeline unchanged; only fusion rule):
  1) frozen          — 0.25 / 0.30 / 0.45
  2) percat_locked   — val grid weights locked per category (no re-search)
  3) adaptive_margin — soft weights from top1−top2 z-score margins
  4) adaptive_agree  — if Composer↔CR rank agree, down-weight coarse

Does not load CIRR paths. Writes boost_metrics.json under --out_dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

logger = logging.getLogger("fiq_fusion_boost")

# Locked from compose41_fused val grid (do not re-tune on new splits).
PERCAT_LOCKED = {
    "dress": (0.0, 0.4, 0.6),
    "shirt": (0.1, 0.6, 0.3),
    "toptee": (0.1, 0.4, 0.5),
}


def _margin_weights(h: np.ndarray, c: np.ndarray, r: np.ndarray, temp: float) -> np.ndarray:
    """Per-query soft weights from top1-top2 margins. Shape (n, 3)."""

    def m(x: np.ndarray) -> np.ndarray:
        part = np.partition(x, -2, axis=1)
        return np.maximum(part[:, -1] - part[:, -2], 0.0)

    stack = np.stack([m(h), m(c), m(r)], axis=1)  # (n, 3)
    logits = stack / max(float(temp), 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    w = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)
    return w


def _agree_weights(
    h: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
    base: Tuple[float, float, float],
    agree_boost: float,
) -> np.ndarray:
    """Blend toward Composer+CR when their top-K order agrees (Spearman-ish)."""
    n, k = c.shape
    # rank vectors: argsort of argsort
    rc = np.argsort(np.argsort(-c, axis=1), axis=1).astype(np.float64)
    rr = np.argsort(np.argsort(-r, axis=1), axis=1).astype(np.float64)
    rc = rc - rc.mean(axis=1, keepdims=True)
    rr = rr - rr.mean(axis=1, keepdims=True)
    num = (rc * rr).sum(axis=1)
    den = np.sqrt((rc * rc).sum(axis=1) * (rr * rr).sum(axis=1))
    corr = num / np.maximum(den, 1e-8)  # [-1, 1]
    # map to [0, 1]
    agree = np.clip(0.5 * (corr + 1.0), 0.0, 1.0)
    a0, b0, g0 = base
    # when agree high: shrink coarse, grow composer+CR proportionally
    shrink = agree_boost * agree  # e.g. up to 0.5 of coarse mass redistributed
    a = a0 * (1.0 - shrink)
    rest = (b0 + g0) + a0 * shrink
    # keep composer:CR ratio
    ratio = b0 / max(b0 + g0, 1e-8)
    b = rest * ratio
    g = rest * (1.0 - ratio)
    return np.stack([a, b, g], axis=1)


def _fuse(h: np.ndarray, c: np.ndarray, r: np.ndarray, w: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
    if isinstance(w, tuple):
        a, b, g = w
        return a * h + b * c + g * r
    return w[:, 0:1] * h + w[:, 1:2] * c + w[:, 2:3] * r


def eval_dir(
    out_dir: Path,
    cats: Sequence[str],
    margin_temp: float = 0.5,
    agree_boost: float = 0.5,
    mix_lambda: float = 0.5,
) -> Dict[str, Any]:
    """mix_lambda: adaptive_margin = λ·margin_w + (1-λ)·frozen (stabilizes)."""
    a0, b0, g0 = FROZEN
    per: Dict[str, Any] = {}
    for cat in cats:
        comp = _load_score(_score_npz(out_dir, "composer", cat))
        cr = _load_score(_score_npz(out_dir, "cr", cat))
        s_hbi, sim, index_names, refs, tgts = _hbi_pool_scores(
            out_dir, cat, comp["k_sorted"], comp["reference_names"], comp["target_names"],
        )
        labels = comp["labels"]
        h = zscore_rows(s_hbi)
        c = zscore_rows(comp["pred_logits"])
        r = zscore_rows(cr["pred_logits"])
        k_sorted = comp["k_sorted"]

        variants: Dict[str, Any] = {}
        # frozen
        fused = _fuse(h, c, r, FROZEN)
        variants["frozen"] = {
            "weights": {"a": a0, "b": b0, "g": g0},
            "full": _full_metrics(fused, k_sorted, sim, index_names, refs, tgts),
            "pool": pool_from_scores(fused, labels),
        }
        # percat locked
        w_cat = PERCAT_LOCKED[cat]
        fused = _fuse(h, c, r, w_cat)
        variants["percat_locked"] = {
            "weights": {"a": w_cat[0], "b": w_cat[1], "g": w_cat[2]},
            "full": _full_metrics(fused, k_sorted, sim, index_names, refs, tgts),
            "pool": pool_from_scores(fused, labels),
        }
        # adaptive margin mixed with frozen
        wm = _margin_weights(h, c, r, margin_temp)
        wf = np.array([a0, b0, g0], dtype=np.float64)
        wmix = mix_lambda * wm + (1.0 - mix_lambda) * wf
        wmix = wmix / np.maximum(wmix.sum(axis=1, keepdims=True), 1e-8)
        fused = _fuse(h, c, r, wmix)
        variants["adaptive_margin"] = {
            "temp": margin_temp,
            "mix_lambda": mix_lambda,
            "mean_weights": {
                "a": float(wmix[:, 0].mean()),
                "b": float(wmix[:, 1].mean()),
                "g": float(wmix[:, 2].mean()),
            },
            "full": _full_metrics(fused, k_sorted, sim, index_names, refs, tgts),
            "pool": pool_from_scores(fused, labels),
        }
        # adaptive agree
        wa = _agree_weights(h, c, r, FROZEN, agree_boost)
        fused = _fuse(h, c, r, wa)
        variants["adaptive_agree"] = {
            "agree_boost": agree_boost,
            "mean_weights": {
                "a": float(wa[:, 0].mean()),
                "b": float(wa[:, 1].mean()),
                "g": float(wa[:, 2].mean()),
            },
            "full": _full_metrics(fused, k_sorted, sim, index_names, refs, tgts),
            "pool": pool_from_scores(fused, labels),
        }
        # combo: percat base + margin residual (small)
        w_base = np.array(w_cat, dtype=np.float64)
        wcombo = 0.7 * w_base + 0.3 * wm
        wcombo = wcombo / np.maximum(wcombo.sum(axis=1, keepdims=True), 1e-8)
        fused = _fuse(h, c, r, wcombo)
        variants["percat_plus_margin"] = {
            "full": _full_metrics(fused, k_sorted, sim, index_names, refs, tgts),
            "pool": pool_from_scores(fused, labels),
        }

        per[cat] = {
            "n_queries": len(refs),
            "variants": {
                name: {
                    **{k: v for k, v in meta.items() if k not in ("full", "pool")},
                    "recall_at10": meta["full"]["recall_at10"],
                    "recall_at50": meta["full"]["recall_at50"],
                    "pool_r10": meta["pool"]["pool_recall_at10"],
                }
                for name, meta in variants.items()
            },
            "variants_full": variants,
        }
        best = max(variants.items(), key=lambda kv: kv[1]["full"]["recall_at10"])
        logger.info(
            "%s frozen=%.2f percat=%.2f margin=%.2f agree=%.2f combo=%.2f | best=%s",
            cat,
            variants["frozen"]["full"]["recall_at10"],
            variants["percat_locked"]["full"]["recall_at10"],
            variants["adaptive_margin"]["full"]["recall_at10"],
            variants["adaptive_agree"]["full"]["recall_at10"],
            variants["percat_plus_margin"]["full"]["recall_at10"],
            best[0],
        )

    def mean_of(variant: str, key: str) -> float:
        return float(np.mean([per[c]["variants"][variant][key] for c in cats]))

    mean = {
        v: {
            "recall_at10": mean_of(v, "recall_at10"),
            "recall_at50": mean_of(v, "recall_at50"),
        }
        for v in ("frozen", "percat_locked", "adaptive_margin", "adaptive_agree", "percat_plus_margin")
    }
    best_name = max(mean.items(), key=lambda kv: kv[1]["recall_at10"])[0]
    report = {
        "out_dir": str(out_dir),
        "percat_locked_weights": {k: list(v) for k, v in PERCAT_LOCKED.items()},
        "mean": mean,
        "best_mean_variant": best_name,
        "per_category": {c: per[c]["variants"] for c in cats},
        "baseline_note": "Compare to prior grid_best ~57.91; percat_locked should match that ballpark.",
    }
    out_json = out_dir / "boost_metrics.json"
    # drop bulky variants_full from disk report
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "MEAN R@10 | frozen=%.2f percat=%.2f margin=%.2f agree=%.2f combo=%.2f | best=%s -> %s",
        mean["frozen"]["recall_at10"],
        mean["percat_locked"]["recall_at10"],
        mean["adaptive_margin"]["recall_at10"],
        mean["adaptive_agree"]["recall_at10"],
        mean["percat_plus_margin"]["recall_at10"],
        best_name,
        out_json,
    )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True, help="Dir with hbi_fused_*/composer_*/cr_*")
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--margin_temp", type=float, default=0.5)
    p.add_argument("--agree_boost", type=float, default=0.5)
    p.add_argument("--mix_lambda", type=float, default=0.5)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    if "cirr" in str(out_dir).lower():
        raise ValueError("must not use cirr path")
    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    eval_dir(out_dir, cats, args.margin_temp, args.agree_boost, args.mix_lambda)


if __name__ == "__main__":
    main()
