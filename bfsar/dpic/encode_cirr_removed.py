#!/usr/bin/env python3
"""CIRR val: triple fusion minus Removed similarity (comma-split phrases).

S = w_v·S_vision + w_m·S_modified + w_r·S_relative - μ·S_removed

Removed: llm_info Removed split by comma; per-phrase BFSAR sim aggregated by mean or max.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr import cirr_paths
from bfsar.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader
from bfsar.cirr.eval_fgsa import encode_gallery, _similarity_matrix
from bfsar.cirr.modeling_fgsa import FGSAModel
from bfsar.cirr.vision_desc.neg_phrase import encode_neg_phrase_sim
from bfsar.dpic.encode_cirr import (
    CIRRTripleVisionDataset,
    collate_fn,
    encode_three,
    weighted_fuse,
    _metrics,
)
from bfsar.cirr.vision_desc.pos_neg_utils import split_legacy_constraints
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("triple_vision_removed")

DEFAULT_CKPT = str(ROOT / "weights" / "fgsa" / "cirr" / "pytorch_model.bin")
DEFAULT_DESCS = str(ROOT / "data" / "cirr" / "vision_desc" / "val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "metrics_triple_vision_removed.json")

ALL_METRIC_KEYS = (
    "recall_at1", "recall_at5", "recall_at10", "recall_at50",
    "recall_at100", "recall_at200",
    "group_recall_at1", "group_recall_at2", "group_recall_at3",
)


def metrics_row(m: dict) -> dict:
    return {k: m[k] for k in ALL_METRIC_KEYS if k in m}


def log_metrics(name: str, m: dict, extra: str = "") -> None:
    suffix = f" | {extra}" if extra else ""
    logger.info(
        "%s%s | R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f | "
        "G-R@1=%.2f G-R@2=%.2f G-R@3=%.2f",
        name, suffix,
        m["recall_at1"], m["recall_at5"], m["recall_at10"], m["recall_at50"],
        m["recall_at100"], m["recall_at200"],
        m["group_recall_at1"], m["group_recall_at2"], m["group_recall_at3"],
    )


def load_neg_items(refs: List[str], tgts: List[str]):
    with open(cirr_paths.llm_json("val"), "r") as f:
        llm = json.load(f)
    neg_items_all = []
    for ref, tgt in zip(refs, tgts):
        key = cirr_paths.pair_key(ref, tgt)
        neg_items_all.append(split_legacy_constraints(llm.get(key, {}).get("Removed") or ""))
    has_neg = np.array([1 if items else 0 for items in neg_items_all], dtype=np.float32)
    return neg_items_all, has_neg


def prepare_removed_sim(sim_removed: np.ndarray, center: bool) -> np.ndarray:
    """Optional per-query gallery demean (see --removed_center)."""
    if not center:
        return sim_removed
    return (sim_removed - sim_removed.mean(axis=1, keepdims=True)).astype(np.float32)


def fuse_triple_minus_removed(
    sim_triple: np.ndarray,
    sim_removed: np.ndarray,
    has_neg: np.ndarray,
    mu: float,
) -> np.ndarray:
    out = sim_triple.astype(np.float64).copy()
    if mu != 0:
        out -= mu * sim_removed.astype(np.float64) * has_neg[:, None]
    return out.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DESCS)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--removed_agg", type=str, default="max", choices=("mean", "max"),
                   help="Aggregate comma-split Removed phrase sims: mean or max")
    p.add_argument("--removed_center", action="store_true",
                   help="Subtract per-query gallery mean from S_removed before penalty")
    p.add_argument("--grid_joint", action="store_true",
                   help="Joint grid over (w_vision,w_modified,w_relative) and mu")
    p.add_argument("--weight_step", type=float, default=0.1)
    p.add_argument("--mu_step", type=float, default=0.05)
    p.add_argument("--w_vision", type=float, default=0.3)
    p.add_argument("--w_modified", type=float, default=0.3)
    p.add_argument("--w_relative", type=float, default=0.4)
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--agg_module", type=str, default="None")
    p.add_argument("--interaction", type=str, default="fgsa")
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--estimator", type=str, default="None")
    p.add_argument("--max_frames", type=int, default=12)
    p.add_argument("--image_resolution", type=int, default=224)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--image_encode_microbatch", type=int, default=12)
    p.add_argument("--v_rate0", type=float, default=0.25)
    p.add_argument("--v_rate1", type=float, default=0.5)
    p.add_argument("--t_rate0", type=float, default=0.25)
    p.add_argument("--t_rate1", type=float, default=0.5)
    p.add_argument("--rate", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--kl", type=float, default=2.0)
    p.add_argument("--lambda_b", type=float, default=1.0)
    p.add_argument("--cirr_image_root", type=str, default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = ClipTokenizer()
    dataset = CIRRTripleVisionDataset(
        args.dual_json, tokenizer, max_words=args.max_words, max_samples=args.max_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = FGSAModel(args)
    state = torch.load(args.init_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded %s | missing=%d unexpected=%d", args.init_model, len(missing), len(unexpected))
    model = model.to(device).eval()

    gallery_loader, _, _ = build_cirr_gallery_dataloader(args, split="val")
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    variants, refs, tgts, groups = encode_three(model, loader, device)

    sim = {}
    with torch.no_grad():
        for key in ("vision", "modified", "relative"):
            feats, cls, masks = variants[key]
            sim[key] = _similarity_matrix(
                model, feats, cls, masks, gallery_feats, gallery_masks, device,
                args.query_batch, args.gallery_batch,
            )

    neg_items_all, has_neg = load_neg_items(refs, tgts)
    sim_removed = encode_neg_phrase_sim(
        model, neg_items_all, gallery_feats, gallery_masks, device,
        args.query_batch, args.gallery_batch, agg=args.removed_agg,
    )
    sim_removed = prepare_removed_sim(sim_removed, args.removed_center)

    weight_step = float(args.weight_step)
    mu_step = float(args.mu_step)
    mu_vals = np.arange(0.0, 1.0 + 1e-9, mu_step)
    w_vals = np.arange(0.0, 1.0 + 1e-9, weight_step)

    grid = []
    best = None

    if args.grid_joint:
        logger.info("Joint grid: weight_step=%.2f mu_step=%.2f removed_agg=%s",
                    weight_step, mu_step, args.removed_agg)
        for wv, wm, mu in itertools.product(w_vals, w_vals, mu_vals):
            wr = 1.0 - wv - wm
            if wr < -1e-9:
                continue
            if wr < 0:
                continue
            if abs(wr) < 1e-9:
                wr = 0.0
            sim_triple = weighted_fuse(
                [sim["vision"], sim["modified"], sim["relative"]], [wv, wm, wr]
            )
            fused = fuse_triple_minus_removed(sim_triple, sim_removed, has_neg, float(mu))
            m = _metrics(fused, index_names, refs, tgts, groups)
            row = {
                "w_vision": round(float(wv), 4),
                "w_modified": round(float(wm), 4),
                "w_relative": round(float(wr), 4),
                "mu": round(float(mu), 4),
                **metrics_row(m),
            }
            grid.append(row)
            if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
                abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
                and m["group_recall_at1"] > best["group_recall_at1"]
            ):
                best = row.copy()
    else:
        wv, wm, wr = args.w_vision, args.w_modified, args.w_relative
        sim_triple = weighted_fuse([sim["vision"], sim["modified"], sim["relative"]], [wv, wm, wr])
        for mu in mu_vals:
            fused = fuse_triple_minus_removed(sim_triple, sim_removed, has_neg, float(mu))
            m = _metrics(fused, index_names, refs, tgts, groups)
            row = {
                "w_vision": round(float(wv), 4),
                "w_modified": round(float(wm), 4),
                "w_relative": round(float(wr), 4),
                "mu": round(float(mu), 4),
                **metrics_row(m),
            }
            grid.append(row)
            if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
                abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
                and m["group_recall_at1"] > best["group_recall_at1"]
            ):
                best = row.copy()

    grid.sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))

    wv0, wm0, wr0 = args.w_vision, args.w_modified, args.w_relative
    sim_triple_ref = weighted_fuse(
        [sim["vision"], sim["modified"], sim["relative"]], [wv0, wm0, wr0]
    )
    results = {
        "vision_mod_rel_fixed": _metrics(sim_triple_ref, index_names, refs, tgts, groups),
        "removed_only": _metrics(sim_removed, index_names, refs, tgts, groups),
        "triple_minus_removed_best": {
            **metrics_row(best),
            "weights": {
                "w_vision": best["w_vision"],
                "w_modified": best["w_modified"],
                "w_relative": best["w_relative"],
                "mu": best["mu"],
            },
        },
    }
    results["vision_mod_rel_fixed"]["weights"] = {
        "w_vision": wv0, "w_modified": wm0, "w_relative": wr0,
    }

    log_metrics("vision_mod_rel_fixed", results["vision_mod_rel_fixed"],
                f"w=({wv0:.2f},{wm0:.2f},{wr0:.2f})")
    log_metrics("removed_only", results["removed_only"], f"agg={args.removed_agg}")
    bw = results["triple_minus_removed_best"]["weights"]
    log_metrics(
        "triple_minus_removed_best", results["triple_minus_removed_best"],
        f"w=({bw['w_vision']:.2f},{bw['w_modified']:.2f},{bw['w_relative']:.2f}) μ={bw['mu']:.2f}",
    )

    logger.info("Top-15 joint grid by R@1:")
    for row in grid[:15]:
        log_metrics(
            f"  w=({row['w_vision']:.1f},{row['w_modified']:.1f},{row['w_relative']:.1f}) μ={row['mu']:.2f}",
            row,
        )

    out = {
        "n_queries": len(refs),
        "init_model": args.init_model,
        "dual_json": args.dual_json,
        "formula": "S = w_v*S_vision + w_m*S_modified + w_r*S_relative - mu*S_removed",
        "removed_source": "llm_info Removed (comma-split)",
        "removed_agg": args.removed_agg,
        "removed_center": bool(args.removed_center),
        "has_neg_ratio": float(has_neg.mean()),
        "grid_joint": bool(args.grid_joint),
        "weight_step": weight_step,
        "mu_step": mu_step,
        "metrics": results,
        "grid_top30": grid[:30],
        "grid_all_count": len(grid),
        "reference_baseline_r1": 41.48,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s (%d grid points)", args.output_json, len(grid))


if __name__ == "__main__":
    main()
