#!/usr/bin/env python3
"""Dump FIQ Top-K pools from OpenCLIP dual-path SDR-HBI coarse.

Uses the shared-αβ best recipe from sdr_dual_sr_ref_vision_openclip:
  Path_ref / Path_vis Anchor-Debias with α=0.4, β=0.2, then
  S = w·z(S_ref) + (1-w)·z(S_vis), w=0.5

Writes hbi_fused_*.npz + hbi_top{K}_val_*.pt for Composer/CR fusion.
CPU-only; reuses refsim_fusion_openclip WTI cache (no CLIP re-encode).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import load_wti_cache, zscore_rows  # noqa: E402
from HBI.fiq.vision_desc.eval_fiq_hbi_cr_rerank import build_topk_payload  # noqa: E402
from HBI.xr_desc.eval_sdr_dual_sr_ref_vision import r_at, sdr_wti, tgt_ranks  # noqa: E402

logger = logging.getLogger("dump_openclip_dual_topk")

DEFAULT_WTI = str(ROOT / "output/fiq_vision_desc/refsim_fusion_openclip")
DEFAULT_OUT = str(ROOT / "output/fiq_rerank/fusion_k500_openclip_dual")
DEFAULT_ALPHA = 0.4
DEFAULT_BETA = 0.2
DEFAULT_W_REF = 0.5
CATS = ("dress", "shirt", "toptee")


def _abort_cirr(*paths: str) -> None:
    for p in paths:
        if p and "cirr" in str(p).lower():
            raise ValueError(f"must not use cirr path: {p}")


def dual_sim(wti: dict, alpha: float, beta: float, w_ref: float) -> np.ndarray:
    zd = zscore_rows(wti["modified"])
    zm = zscore_rows(wti["relative"])
    zr_ref = zscore_rows(wti["ref"])
    zr_vis = zscore_rows(wti["vision"])
    s_ref = sdr_wti(zd, zm, zr_ref, alpha, beta)
    s_vis = sdr_wti(zd, zm, zr_vis, alpha, beta)
    return (w_ref * zscore_rows(s_ref) + (1.0 - w_ref) * zscore_rows(s_vis)).astype(np.float32)


def dump_cat(
    cat: str,
    wti_dir: Path,
    out_dir: Path,
    k_save: int,
    alpha: float,
    beta: float,
    w_ref: float,
) -> dict:
    wti = load_wti_cache(wti_dir, cat)
    if wti is None:
        raise FileNotFoundError(f"missing WTI cache: {wti_dir}/scores_{cat}.npz")
    idx = list(wti["index_names"])
    refs = list(wti["refs"])
    tgts = list(wti["tgts"])
    fused = dual_sim(wti, alpha, beta, w_ref)

    name = {n: i for i, n in enumerate(idx)}
    ref_i = np.array([name.get(r, -1) for r in refs], dtype=np.int32)
    tgt_i = np.array([name[t] for t in tgts], dtype=np.int32)
    rk = tgt_ranks(fused, ref_i, tgt_i)
    metrics = {
        "recall_at10": r_at(rk, 10),
        "recall_at50": r_at(rk, 50),
        "recall_at100": r_at(rk, 100),
        "recall_at200": r_at(rk, 200),
        "recall_at300": r_at(rk, 300),
        "coverage_at_k": r_at(rk, k_save),
    }

    np.savez_compressed(
        out_dir / f"hbi_fused_{cat}.npz",
        sim=fused,
        index_names=np.asarray(idx, dtype=object),
        refs=np.asarray(refs, dtype=object),
        tgts=np.asarray(tgts, dtype=object),
    )
    payload = build_topk_payload(fused, idx, refs, tgts, cat, k_save)
    payload["source"] = "openclip_dual_path_sdr_hbi"
    payload["weights"] = {"alpha": alpha, "beta": beta, "w_ref": w_ref}
    pt = out_dir / f"hbi_top{k_save}_val_{cat}.pt"
    torch.save(payload, pt)
    meta = {
        "coarse": metrics,
        "recipe": {
            "name": "sdr_dual_sr_ref_vision_openclip_shared_ab_best",
            "alpha": alpha,
            "beta": beta,
            "w_ref": w_ref,
            "path_ref": "S_r=reference image",
            "path_vis": "S_r=vision_side WTI",
        },
        "topk_pt": str(pt),
        "n_queries": len(refs),
        "n_gallery": len(idx),
        "wti_dir": str(wti_dir),
    }
    (out_dir / f"coarse_{cat}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "%s dual R@10/50/100/300=%.2f/%.2f/%.2f/%.2f | top%d coverage=%.2f",
        cat,
        metrics["recall_at10"],
        metrics["recall_at50"],
        metrics["recall_at100"],
        metrics["recall_at300"],
        k_save,
        metrics["coverage_at_k"],
    )
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--wti_dir", type=str, default=DEFAULT_WTI)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--k_save", type=int, default=500)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--w_ref", type=float, default=DEFAULT_W_REF)
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    args = p.parse_args()
    _abort_cirr(args.wti_dir, args.out_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    per = {}
    for cat in cats:
        per[cat] = dump_cat(
            cat, Path(args.wti_dir), out_dir, args.k_save, args.alpha, args.beta, args.w_ref
        )
    mean = {k: float(np.mean([per[c][k] for c in cats])) for k in next(iter(per.values()))}
    summary = {
        "exp": "dump_openclip_dual_topk",
        "recipe": f"dual Path_ref+Path_vis α={args.alpha} β={args.beta} w_ref={args.w_ref}",
        "k_save": args.k_save,
        "wti_dir": args.wti_dir,
        "mean": mean,
        "per_category": per,
    }
    (out_dir / "coarse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "mean R@10/50/100/300=%.2f/%.2f/%.2f/%.2f coverage@%d=%.2f -> %s",
        mean["recall_at10"],
        mean["recall_at50"],
        mean["recall_at100"],
        mean["recall_at300"],
        args.k_save,
        mean["coverage_at_k"],
        out_dir,
    )


if __name__ == "__main__":
    main()
