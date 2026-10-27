#!/usr/bin/env python3
"""Export BFSAR coarse Top-K in Candidate-Reranking-CIR .pt format (replaces BLIP stage-1 top-k)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from bfsar.mvr.cvs_infer import (
    align_gallery_index_to_cirr,
    build_full_val_topk_pt,
    save_topk_pt,
)
from bfsar.rerank.paths import DEFAULT_CIRR_TOPK_VAL


def load_coarse_sim_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return (
        d["sim_hbi"].astype(np.float32),
        list(d["refs"]),
        list(d["tgts"]),
        list(d["index_names"]),
    )

logger = logging.getLogger("export_hbi_topk")

DEFAULT_COARSE_NPZ = str(ROOT / "output/cirr_vision_desc/hbi_triple_removed_best_sim.npz")
DEFAULT_OUT = str(DEFAULT_CIRR_TOPK_VAL)


def parse_args():
    p = argparse.ArgumentParser(description="Export BFSAR val Top-K for BLIP stage-2 train/val")
    p.add_argument("--split", type=str, default="val", choices=["val"], help="Only val supported for BFSAR E0a")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--coarse_sim_npz", type=str, default=DEFAULT_COARSE_NPZ)
    p.add_argument("--output_pt", type=str, default=DEFAULT_OUT)
    p.add_argument("--recompute_hbi", action="store_true")
    p.add_argument("--init_model", type=str, default=str(ROOT / "output/cirr_retrieval/pytorch_model.bin.best.10"))
    p.add_argument("--dual_json", type=str, default=str(ROOT / "data/cirr/vision_desc/val_dual_targets.json"))
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.split != "val":
        raise SystemExit("BFSAR E0a cache is val-only; use Candidate-Reranking stage1 for test1 top-k.")

    npz = Path(args.coarse_sim_npz)
    if not npz.is_file():
        raise FileNotFoundError(f"Missing coarse scores: {npz}")
    sim_hbi, hbi_refs, hbi_tgts, index_names = load_coarse_sim_npz(str(npz))
    sim_hbi, index_names = align_gallery_index_to_cirr(sim_hbi, index_names)
    payload = build_full_val_topk_pt(
        sim_hbi, hbi_refs, hbi_tgts, index_names, k_save=args.k
    )
    out = save_topk_pt(payload, Path(args.output_pt))
    hit = float(payload["labels"].any(axis=1).mean() * 100)
    logger.info("Done: %s | target-in-top-%d recall ~ %.2f%% (4181 queries)", out, args.k, hit)


if __name__ == "__main__":
    main()
