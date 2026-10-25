#!/usr/bin/env python3
"""Fuse best HBI triple-minus-Removed score with CLIP ref→gallery similarity.

S_final = S_HBI * (S_img + eps) ^ gamma

S_HBI: w=(0.3,0.3,0.4), mu=0.1, Removed agg=max (best from prior grid).
S_img: raw ViT-L/14 ref→gallery cosine (ref_gallery_clip_sim.npz).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.cirr import cirr_paths
from HBI.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader
from HBI.cirr.eval_cirr_hbi import encode_gallery, _similarity_matrix
from HBI.cirr.modeling_hbi_cirr import HBICIRR
from HBI.cirr.vision_desc.eval_hbi_pos_neg import encode_neg_phrase_sim
from HBI.cirr.vision_desc.eval_hbi_triple_vision import (
    CIRRTripleVisionDataset,
    collate_fn,
    encode_three,
    weighted_fuse,
    _metrics,
)
from HBI.cirr.vision_desc.eval_hbi_triple_vision_removed import (
    ALL_METRIC_KEYS,
    fuse_triple_minus_removed,
    load_neg_items,
    log_metrics,
    metrics_row,
    prepare_removed_sim,
)
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("triple_clip_gate")

DEFAULT_CKPT = str(ROOT / "output" / "cirr_retrieval" / "pytorch_model.bin.best.10")
DEFAULT_DESCS = str(ROOT / "data" / "cirr" / "vision_desc" / "val_dual_targets.json")
DEFAULT_CLIP_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "ref_gallery_clip_sim.npz")
DEFAULT_HBI_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "hbi_triple_removed_best_sim.npz")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "metrics_triple_clip_gate.json")

BEST_W = (0.3, 0.3, 0.4)
BEST_MU = 0.1
BEST_REMOVED_AGG = "max"


def normalize_rows(sim: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return sim.astype(np.float32)
    if mode == "minmax":
        lo = sim.min(axis=1, keepdims=True)
        hi = sim.max(axis=1, keepdims=True)
        return ((sim - lo) / (hi - lo + 1e-6)).astype(np.float32)
    if mode == "zscore":
        mu = sim.mean(axis=1, keepdims=True)
        std = sim.std(axis=1, keepdims=True)
        return ((sim - mu) / (std + 1e-6)).astype(np.float32)
    raise ValueError(f"Unknown norm mode: {mode}")


def fuse_gate(
    sim_hbi: np.ndarray,
    sim_img: np.ndarray,
    gamma: float,
    norm_mode: str,
    eps: float,
) -> np.ndarray:
    h = normalize_rows(sim_hbi, norm_mode)
    v = normalize_rows(sim_img, norm_mode)
    if norm_mode == "zscore":
        # Map to (0,1) for stable power; equivalent to per-query min-max on z-scores.
        h = normalize_rows(h, "minmax")
        v = normalize_rows(v, "minmax")
    if gamma == 0.0:
        return h.astype(np.float32)
    return (h * np.power(v + eps, gamma)).astype(np.float32)


def load_clip_sim(path: str) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    d = np.load(path, allow_pickle=True)
    sim_img = d["sim_img"].astype(np.float32)
    refs = list(d["refs"])
    tgts = list(d["tgts"])
    index_names = list(d["index_names"])
    return sim_img, refs, tgts, index_names


def align_clip_sim(
    sim_img: np.ndarray,
    clip_refs: List[str],
    clip_tgts: List[str],
    index_names: List[str],
    hbi_refs: List[str],
    hbi_tgts: List[str],
) -> np.ndarray:
    if clip_refs == hbi_refs and clip_tgts == hbi_tgts:
        return sim_img
    key_to_row = {f"{r}{t}": i for i, (r, t) in enumerate(zip(clip_refs, clip_tgts))}
    rows = []
    for r, t in zip(hbi_refs, hbi_tgts):
        k = f"{r}{t}"
        if k not in key_to_row:
            raise KeyError(f"Missing CLIP sim row for {k}")
        rows.append(sim_img[key_to_row[k]])
    return np.stack(rows, axis=0)


@torch.no_grad()
def compute_hbi_sim(args) -> Tuple[np.ndarray, List[str], List[str], list, List[str]]:
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

    model = HBICIRR(args)
    state = torch.load(args.init_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()
    logger.info("Loaded HBI checkpoint %s", args.init_model)

    gallery_loader, _, _ = build_cirr_gallery_dataloader(args, split="val")
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    variants, refs, tgts, groups = encode_three(model, loader, device)

    sim_parts = {}
    for key in ("vision", "modified", "relative"):
        feats, cls, masks = variants[key]
        sim_parts[key] = _similarity_matrix(
            model, feats, cls, masks, gallery_feats, gallery_masks, device,
            args.query_batch, args.gallery_batch,
        )

    neg_items_all, has_neg = load_neg_items(refs, tgts)
    sim_removed = encode_neg_phrase_sim(
        model, neg_items_all, gallery_feats, gallery_masks, device,
        args.query_batch, args.gallery_batch, agg=BEST_REMOVED_AGG,
    )
    sim_removed = prepare_removed_sim(sim_removed, center=False)

    wv, wm, wr = BEST_W
    sim_triple = weighted_fuse(
        [sim_parts["vision"], sim_parts["modified"], sim_parts["relative"]], [wv, wm, wr]
    )
    sim_hbi = fuse_triple_minus_removed(sim_triple, sim_removed, has_neg, BEST_MU)
    return sim_hbi, refs, tgts, groups, index_names


def load_or_compute_hbi(args) -> Tuple[np.ndarray, List[str], List[str], list, List[str]]:
    cache = Path(args.hbi_sim_npz)
    if cache.is_file() and not args.recompute_hbi:
        d = np.load(cache, allow_pickle=True)
        logger.info("Loaded cached S_HBI from %s", cache)
        return (
            d["sim_hbi"].astype(np.float32),
            list(d["refs"]),
            list(d["tgts"]),
            [list(g) for g in d["groups"]],
            list(d["index_names"]),
        )

    sim_hbi, refs, tgts, groups, index_names = compute_hbi_sim(args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        sim_hbi=sim_hbi.astype(np.float32),
        refs=np.array(refs, dtype=object),
        tgts=np.array(tgts, dtype=object),
        groups=np.array(groups, dtype=object),
        index_names=np.array(index_names, dtype=object),
        w_vision=BEST_W[0],
        w_modified=BEST_W[1],
        w_relative=BEST_W[2],
        mu=BEST_MU,
        removed_agg=BEST_REMOVED_AGG,
    )
    logger.info("Wrote cached S_HBI to %s shape=%s", cache, sim_hbi.shape)
    return sim_hbi, refs, tgts, groups, index_names


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DESCS)
    p.add_argument("--clip_sim_npz", type=str, default=DEFAULT_CLIP_NPZ)
    p.add_argument("--hbi_sim_npz", type=str, default=DEFAULT_HBI_NPZ)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--recompute_hbi", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--gamma_step", type=float, default=0.25)
    p.add_argument("--gamma_max", type=float, default=3.0)
    p.add_argument(
        "--norm_modes",
        type=str,
        nargs="+",
        default=["raw", "minmax"],
        choices=["raw", "minmax", "zscore"],
    )
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--agg_module", type=str, default="None")
    p.add_argument("--interaction", type=str, default="wti")
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
    p.add_argument("--skl", type=float, default=1.0)
    p.add_argument("--cirr_image_root", type=str, default=None)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    sim_img_all, clip_refs, clip_tgts, index_names = load_clip_sim(args.clip_sim_npz)
    sim_hbi, refs, tgts, groups, hbi_index_names = load_or_compute_hbi(args)
    if list(index_names) != list(hbi_index_names):
        raise ValueError("Gallery index_names mismatch between CLIP npz and HBI cache")
    sim_img = align_clip_sim(sim_img_all, clip_refs, clip_tgts, index_names, refs, tgts)

    m_hbi = _metrics(sim_hbi, index_names, refs, tgts, groups)
    log_metrics("S_HBI baseline (cached best)", m_hbi,
                f"w={BEST_W} mu={BEST_MU} agg={BEST_REMOVED_AGG}")

    gamma_vals = np.arange(0.0, args.gamma_max + 1e-9, args.gamma_step)
    grid = []
    best = None

    for norm_mode in args.norm_modes:
        for gamma in gamma_vals:
            gamma = float(round(gamma, 4))
            fused = fuse_gate(sim_hbi, sim_img, gamma, norm_mode, args.eps)
            m = _metrics(fused, index_names, refs, tgts, groups)
            row = {
                "norm_mode": norm_mode,
                "gamma": gamma,
                "eps": args.eps,
                **metrics_row(m),
            }
            grid.append(row)
            if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
                abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
                and m["group_recall_at1"] > best["group_recall_at1"]
            ):
                best = row.copy()

    grid.sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))

    logger.info("Top-10 gamma grid:")
    for row in grid[:10]:
        log_metrics(
            f"  norm={row['norm_mode']} gamma={row['gamma']:.2f}",
            row,
        )

    out = {
        "n_queries": len(refs),
        "formula": "S_final = S_HBI * (S_img + eps) ^ gamma",
        "s_hbi": {
            "source": "triple minus Removed best",
            "w_vision": BEST_W[0],
            "w_modified": BEST_W[1],
            "w_relative": BEST_W[2],
            "mu": BEST_MU,
            "removed_agg": BEST_REMOVED_AGG,
            "metrics": metrics_row(m_hbi),
        },
        "s_img_source": args.clip_sim_npz,
        "hbi_sim_npz": args.hbi_sim_npz,
        "eps": args.eps,
        "gamma_step": args.gamma_step,
        "gamma_max": args.gamma_max,
        "norm_modes": args.norm_modes,
        "best": best,
        "grid_top20": grid[:20],
        "grid_all_count": len(grid),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log_metrics("best fused", best, f"norm={best['norm_mode']} gamma={best['gamma']:.2f}")
    logger.info("Wrote %s (%d grid points)", args.output_json, len(grid))


if __name__ == "__main__":
    main()
