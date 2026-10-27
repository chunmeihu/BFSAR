#!/usr/bin/env python3
"""FIQ coarse fusion: existing text branches + reference↔gallery visual similarity.

Does NOT retrain BFSAR / Banzhaf. Does NOT touch CIRR.
Literature:
  - Fashion-IQ 2020 2nd place: Bayesian-weighted sum of score matrices,
    including Text-only and Image-retrieval priors with RTIC
    (arXiv:2007.06404) — RTIC+Image retrieval lifts Avg vs RTIC alone.
  - Prefer small w to limit reference dominance; per-query z-score
    before mixing so branch scales match.

Writes only under output/fiq_vision_desc/refsim_fusion_v1/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr.eval_fgsa import encode_gallery, _similarity_matrix
from bfsar.cirr.modeling_fgsa import FGSAModel
from bfsar.cirr.vision_desc.eval_hbi_pos_neg import encode_neg_phrase_sim
from bfsar.dpic.encode_cirr import encode_three, weighted_fuse
from bfsar.dpic.encode_cirr_removed import (
    fuse_triple_minus_removed,
    prepare_removed_sim,
)
from bfsar.cirr.vision_desc.pos_neg_utils import split_legacy_constraints
from bfsar.fiq import fiq_paths
from bfsar.fiq.fiq_dataloader_builders import build_fiq_gallery_dataloader
from bfsar.dpic.encode_fiq import (
    FIQTripleVisionDataset,
    collate_fn,
    fiq_metrics_from_sim,
)
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("fiq_refsim_fusion")

DEFAULT_CKPT = str(ROOT / "output/fiq_retrieval/pytorch_model.bin.best.5")
DEFAULT_DUAL = str(ROOT / "data/fashioniq/vision_desc/val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output/fiq_vision_desc/refsim_fusion_v1")
# FIQ-grid best text fusion (mean R@10≈38.58)
BASE_W = (0.3, 0.4, 0.3)
BASE_MU = 0.0


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def masked_mean_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (feat * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def ref_gallery_cosine(
    gallery_feats: torch.Tensor,
    gallery_masks: torch.Tensor,
    index_names: Sequence[str],
    refs: Sequence[str],
) -> np.ndarray:
    """Per-query cosine(ref_image, all gallery) in pooled BFSAR visual space."""
    pooled = F.normalize(masked_mean_pool(gallery_feats, gallery_masks), dim=-1)
    name_to_i = {str(n): i for i, n in enumerate(index_names)}
    g = pooled.numpy().astype(np.float64)
    gram = (g @ g.T).astype(np.float32)
    n_q = len(refs)
    out = np.zeros((n_q, len(index_names)), dtype=np.float32)
    n_miss = 0
    for i, ref in enumerate(refs):
        j = name_to_i.get(str(ref))
        if j is None:
            n_miss += 1
            continue
        out[i] = gram[j]
    if n_miss:
        logger.warning("ref missing from gallery: %d / %d", n_miss, n_q)
    return out


def fuse_scores(
    branches: Dict[str, np.ndarray],
    weights: Dict[str, float],
    *,
    norm: str,
) -> np.ndarray:
    acc = None
    for name, w in weights.items():
        if abs(w) < 1e-12:
            continue
        s = branches[name]
        if norm == "zscore":
            s = zscore_rows(s)
        elif norm == "raw":
            s = np.asarray(s, dtype=np.float64)
        else:
            raise ValueError(norm)
        acc = w * s if acc is None else acc + w * s
    return np.asarray(acc, dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="FIQ text branches + ref-gallery sim fusion")
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DUAL)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--removed_agg", type=str, default="max", choices=("mean", "max"))
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=fiq_paths.CLIP_VITL14_CHECKPOINT)
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
    p.add_argument("--fiq_image_root", type=str, default=None)
    p.add_argument("--grid_ref", type=str, default="0,0.05,0.1,0.15,0.2,0.25,0.3")
    p.add_argument("--save_npz", type=int, default=1)
    return p.parse_args()


def eval_category(model, args, tokenizer, category: str, device) -> Dict:
    dataset = FIQTripleVisionDataset(
        args.dual_json, tokenizer, category=category,
        max_words=args.max_words, max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    gallery_loader, n_g, _ = build_fiq_gallery_dataloader(args, split="val", category=category)
    logger.info("FIQ %s gallery=%d queries=%d", category, n_g, len(dataset))
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    variants, refs, tgts, _groups = encode_three(model, loader, device)

    sim: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for key in ("vision", "modified", "relative"):
            feats, cls, masks = variants[key]
            sim[key] = _similarity_matrix(
                model, feats, cls, masks, gallery_feats, gallery_masks, device,
                args.query_batch, args.gallery_batch,
            )

    neg_items_all = [split_legacy_constraints(q["removed"]) for q in dataset.queries]
    has_neg = np.array([1 if items else 0 for items in neg_items_all], dtype=np.float32)
    sim_removed = encode_neg_phrase_sim(
        model, neg_items_all, gallery_feats, gallery_masks, device,
        args.query_batch, args.gallery_batch, agg=args.removed_agg,
    )
    sim_removed = prepare_removed_sim(sim_removed, center=False)
    sim_ref = ref_gallery_cosine(gallery_feats.cpu(), gallery_masks.cpu(), index_names, refs)

    branches = {
        "vision": sim["vision"],
        "modified": sim["modified"],
        "relative": sim["relative"],
        "removed": sim_removed,
        "ref": sim_ref,
    }

    wv0, wm0, wr0 = BASE_W
    mu0 = BASE_MU
    singles = {
        "vision": fiq_metrics_from_sim(branches["vision"], index_names, refs, tgts),
        "modified": fiq_metrics_from_sim(branches["modified"], index_names, refs, tgts),
        "relative": fiq_metrics_from_sim(branches["relative"], index_names, refs, tgts),
        "ref": fiq_metrics_from_sim(branches["ref"], index_names, refs, tgts),
    }
    base_triple = weighted_fuse(
        [branches["vision"], branches["modified"], branches["relative"]],
        [wv0, wm0, wr0],
    )
    base_fused = fuse_triple_minus_removed(base_triple, branches["removed"], has_neg, float(mu0))
    singles["baseline_0.3_0.4_0.3_mu0"] = fiq_metrics_from_sim(base_fused, index_names, refs, tgts)
    logger.info(
        "%s singles R@10 vision=%.2f mod=%.2f rel=%.2f ref=%.2f baseline=%.2f",
        category,
        singles["vision"]["recall_at10"],
        singles["modified"]["recall_at10"],
        singles["relative"]["recall_at10"],
        singles["ref"]["recall_at10"],
        singles["baseline_0.3_0.4_0.3_mu0"]["recall_at10"],
    )

    ref_grid = [float(x) for x in args.grid_ref.split(",") if x.strip()]
    grid_rows = []
    best = {"zscore": None, "raw": None}
    for norm in ("zscore", "raw"):
        for w in ref_grid:
            # Challenge-style: scale text mass by (1-w), add w on image prior
            scale = max(0.0, 1.0 - w)
            weights = {
                "vision": wv0 * scale,
                "modified": wm0 * scale,
                "relative": wr0 * scale,
                "ref": w,
            }
            fused = fuse_scores(branches, weights, norm=norm)
            if abs(mu0) > 1e-12:
                rem_s = zscore_rows(branches["removed"]) if norm == "zscore" else np.asarray(
                    branches["removed"], dtype=np.float64
                )
                fused = fused - mu0 * scale * rem_s * has_neg[:, None].astype(np.float64)
                fused = np.asarray(fused, dtype=np.float32)
            m = fiq_metrics_from_sim(fused, index_names, refs, tgts)
            row = {"norm": norm, "w": w, "weights": weights, **m}
            grid_rows.append(row)
            if best[norm] is None or m["recall_at10"] > best[norm]["recall_at10"]:
                best[norm] = row
        logger.info(
            "%s best %s: R@10=%.2f R@50=%.2f w=%.2f",
            category, norm, best[norm]["recall_at10"], best[norm]["recall_at50"], best[norm]["w"],
        )

    return {
        "n_queries": len(refs),
        "n_gallery": len(index_names),
        "singles": singles,
        "grid_best": best,
        "grid_top": sorted(grid_rows, key=lambda r: (-r["recall_at10"], -r["recall_at50"]))[:20],
        "branches": branches if args.save_npz else None,
        "has_neg": has_neg if args.save_npz else None,
        "index_names": index_names if args.save_npz else None,
        "refs": refs if args.save_npz else None,
        "tgts": tgts if args.save_npz else None,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if "cirr" in os.path.abspath(str(out_dir)).lower():
        raise ValueError(f"Must not write under cirr path: {out_dir}")
    if "cirr" in os.path.abspath(args.init_model).lower():
        raise ValueError(f"Must not load CIRR ckpt: {args.init_model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    args.distributed = 0

    tokenizer = ClipTokenizer()
    model = FGSAModel(args)
    state = torch.load(args.init_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded %s miss=%d unexp=%d", args.init_model, len(miss), len(unexp))
    model = model.to(device).eval()

    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    wv0, wm0, wr0 = BASE_W
    mu0 = BASE_MU
    per_cat_payload = {}

    for cat in cats:
        logger.info("=== category %s ===", cat)
        payload = eval_category(model, args, tokenizer, cat, device)
        branches = payload.pop("branches")
        has_neg = payload.pop("has_neg")
        index_names = payload.pop("index_names")
        refs = payload.pop("refs")
        tgts = payload.pop("tgts")
        per_cat_payload[cat] = payload
        if args.save_npz and branches is not None:
            np.savez_compressed(
                out_dir / f"scores_{cat}.npz",
                vision=branches["vision"],
                modified=branches["modified"],
                relative=branches["relative"],
                removed=branches["removed"],
                ref=branches["ref"],
                has_neg=has_neg,
                index_names=np.array(index_names, dtype=object),
                refs=np.array(refs, dtype=object),
                tgts=np.array(tgts, dtype=object),
            )

    def mean_metric(path_fn):
        return {
            k: float(np.mean([path_fn(per_cat_payload[c])[k] for c in cats]))
            for k in ("recall_at10", "recall_at50", "recall_at100", "recall_at200", "recall_at300")
        }

    out = {
        "exp": "fiq_refsim_fusion_v1",
        "init_model": args.init_model,
        "dual_json": args.dual_json,
        "base_text_weights": {"w_vision": wv0, "w_modified": wm0, "w_relative": wr0, "mu": mu0},
        "fusion_notes": [
            "FIQ2020 2nd place: weighted sum of score matrices (incl. image retrieval prior).",
            "Primary recipe: per-query z-score then weighted sum; text weights scaled by (1-w).",
            "Keep w small (≤0.3) to avoid reference dominance.",
        ],
        "mean": {
            "baseline_0.3_0.4_0.3_mu0": mean_metric(lambda d: d["singles"]["baseline_0.3_0.4_0.3_mu0"]),
            "ref_only": mean_metric(lambda d: d["singles"]["ref"]),
            "best_zscore": mean_metric(lambda d: d["grid_best"]["zscore"]),
            "best_raw": mean_metric(lambda d: d["grid_best"]["raw"]),
        },
        "per_category": {
            c: {k: v for k, v in per_cat_payload[c].items()} for c in cats
        },
    }

    # Shared w maximizing mean R@10 under zscore (and raw)
    if args.save_npz:
        ref_grid = [float(x) for x in args.grid_ref.split(",") if x.strip()]
        for norm in ("zscore", "raw"):
            shared = []
            for w in ref_grid:
                scale = max(0.0, 1.0 - w)
                r10s, r50s = [], []
                for cat in cats:
                    d = np.load(out_dir / f"scores_{cat}.npz", allow_pickle=True)
                    branches = {
                        "vision": d["vision"],
                        "modified": d["modified"],
                        "relative": d["relative"],
                        "ref": d["ref"],
                    }
                    weights = {
                        "vision": wv0 * scale,
                        "modified": wm0 * scale,
                        "relative": wr0 * scale,
                        "ref": w,
                    }
                    fused = fuse_scores(branches, weights, norm=norm)
                    m = fiq_metrics_from_sim(
                        fused,
                        [str(x) for x in d["index_names"].tolist()],
                        [str(x) for x in d["refs"].tolist()],
                        [str(x) for x in d["tgts"].tolist()],
                    )
                    r10s.append(m["recall_at10"])
                    r50s.append(m["recall_at50"])
                shared.append({
                    "w": w,
                    "recall_at10": float(np.mean(r10s)),
                    "recall_at50": float(np.mean(r50s)),
                    "per_cat_r10": {c: float(r10s[i]) for i, c in enumerate(cats)},
                    "per_cat_r50": {c: float(r50s[i]) for i, c in enumerate(cats)},
                })
            shared.sort(key=lambda x: (-x["recall_at10"], -x["recall_at50"]))
            out[f"shared_w_{norm}"] = {"best": shared[0], "all": shared}

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_dir / "metrics.json")
    logger.info(
        "MEAN baseline R@10=%.2f | ref_only=%.2f | best_zscore=%.2f | best_raw=%.2f",
        out["mean"]["baseline_0.3_0.4_0.3_mu0"]["recall_at10"],
        out["mean"]["ref_only"]["recall_at10"],
        out["mean"]["best_zscore"]["recall_at10"],
        out["mean"]["best_raw"]["recall_at10"],
    )
    for norm in ("zscore", "raw"):
        key = f"shared_w_{norm}"
        if key in out:
            b = out[key]["best"]
            logger.info(
                "Shared %s best w=%.2f mean R@10=%.2f R@50=%.2f",
                norm, b["w"], b["recall_at10"], b["recall_at50"],
            )


if __name__ == "__main__":
    main()
