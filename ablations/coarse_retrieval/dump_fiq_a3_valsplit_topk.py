#!/usr/bin/env python3
"""Dump FIQ VAL-Split (small gallery) Top-K from Original-Split A3 dual scores.

Gallery = unique ref ∪ tgt in val-triplet order (MCoT). CPU-only. Writes under
output/ablations/. Does not re-encode CLIP.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir
from HBI.fiq.vision_desc.dump_fiq_openclip_dual_topk import dual_sim
from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import load_wti_cache
from HBI.fiq.vision_desc.eval_fiq_hbi_cr_rerank import build_topk_payload
from HBI.xr_desc.eval_sdr_dual_sr_ref_vision import r_at, tgt_ranks

logger = logging.getLogger("dump_a3_valsplit_topk")
CATS = ("dress", "shirt", "toptee")


def mcot_gallery(refs, tgts):
    seen = set()
    gal = []
    for r, t in zip(refs, tgts):
        for n in (r, t):
            if n not in seen:
                seen.add(n)
                gal.append(n)
    return gal


def subset_wti(wti, gal):
    name = {n: i for i, n in enumerate(wti["index_names"])}
    cols = np.array([name[n] for n in gal], dtype=np.int64)
    out = {k: np.asarray(wti[k])[:, cols] for k in ("vision", "modified", "relative", "ref")}
    out["index_names"] = list(gal)
    out["refs"] = list(wti["refs"])
    out["tgts"] = list(wti["tgts"])
    return out


def dump_cat(cat, wti_dir, out_dir, k_save, alpha, beta, w_ref):
    wti = load_wti_cache(wti_dir, cat)
    if wti is None:
        raise FileNotFoundError(f"missing {wti_dir}/scores_{cat}.npz")
    gal = mcot_gallery(wti["refs"], wti["tgts"])
    small = subset_wti(wti, gal)
    fused = dual_sim(small, alpha, beta, w_ref)
    idx = small["index_names"]
    refs = small["refs"]
    tgts = small["tgts"]
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
        "n_queries": len(refs),
        "n_gallery": len(idx),
    }
    np.savez_compressed(
        out_dir / f"hbi_fused_{cat}.npz",
        sim=fused,
        index_names=np.asarray(idx, dtype=object),
        refs=np.asarray(refs, dtype=object),
        tgts=np.asarray(tgts, dtype=object),
    )
    payload = build_topk_payload(fused, idx, refs, tgts, cat, k_save)
    payload["source"] = "a3_entity_banzhaf_dual_valsplit_subset"
    payload["weights"] = {"alpha": alpha, "beta": beta, "w_ref": w_ref}
    pt = out_dir / f"hbi_top{k_save}_val_{cat}.pt"
    torch.save(payload, pt)
    meta = {
        "gallery": "val-split",
        "coarse": metrics,
        "topk_pt": str(pt),
        "n_queries": len(refs),
        "n_gallery": len(idx),
        "wti_dir": str(wti_dir),
    }
    (out_dir / f"coarse_{cat}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "%s valsplit gallery=%d R@10/50=%.2f/%.2f cov@%d=%.2f",
        cat, len(idx), metrics["recall_at10"], metrics["recall_at50"],
        k_save, metrics["coverage_at_k"],
    )
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--wti_dir",
        default=str(ROOT / "output/ablations/fiq_coarse/A3_entity_only_dual"),
    )
    p.add_argument(
        "--out_dir",
        default=str(ROOT / "output/ablations/fiq_rerank/A3_dual_k500_valsplit"),
    )
    p.add_argument("--k_save", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--w_ref", type=float, default=0.5)
    args = p.parse_args()
    out = ensure_ablation_out_dir(args.out_dir)
    wti_dir = Path(args.wti_dir)
    per = {}
    for cat in CATS:
        per[cat] = dump_cat(cat, wti_dir, out, args.k_save, args.alpha, args.beta, args.w_ref)
    mean = {k: float(np.mean([per[c][k] for c in CATS if k in per[c] and isinstance(per[c][k], float)]))
            for k in ("recall_at10", "recall_at50", "recall_at100", "recall_at200", "recall_at300", "coverage_at_k")}
    summary = {
        "exp": "dump_a3_dual_valsplit_topk",
        "gallery": "VAL-Split unique ref∪tgt",
        "k_save": args.k_save,
        "wti_dir": str(wti_dir),
        "mean": mean,
        "per_category": per,
    }
    (out / "coarse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("mean R@10/50=%.2f/%.2f -> %s", mean["recall_at10"], mean["recall_at50"], out)


if __name__ == "__main__":
    main()
