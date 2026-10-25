#!/usr/bin/env python3
"""Fashion-IQ val coarse: CIRR frozen triple − Removed.

S = 0.3 S_vision + 0.3 S_modified + 0.4 S_relative − 0.1 S_removed

Per-category galleries (official FIQ). Does not retune weights. Does not write under cirr/.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.fiq import fiq_paths
from HBI.fiq.eval_fiq_hbi import compute_fiq_metrics
from HBI.fiq.fiq_dataloader_builders import build_fiq_gallery_dataloader
from HBI.cirr.eval_cirr_hbi import encode_gallery, _similarity_matrix
from HBI.cirr.metrics_cirr_hbi import rank_by_similarity
from HBI.cirr.modeling_hbi_cirr import HBICIRR
from HBI.cirr.vision_desc.eval_hbi_pos_neg import encode_neg_phrase_sim
from HBI.cirr.vision_desc.eval_hbi_triple_vision import encode_three, weighted_fuse
from HBI.cirr.vision_desc.eval_hbi_triple_vision_removed import (
    fuse_triple_minus_removed,
    prepare_removed_sim,
)
from HBI.cirr.vision_desc.pos_neg_utils import split_legacy_constraints
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer


def _fiq_mod(row: dict) -> str:
    caps = row.get("captions") or []
    parts = [str(c).strip() for c in caps if str(c).strip()]
    return " ".join(parts)

logger = logging.getLogger("fiq_triple_vision_removed")

BEST_W = (0.3, 0.3, 0.4)
BEST_MU = 0.1
DEFAULT_CKPT = str(ROOT / "output" / "fiq_retrieval" / "pytorch_model.bin.best.5")
DEFAULT_DESCS = str(ROOT / "data" / "fashioniq" / "vision_desc" / "val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output" / "fiq_vision_desc" / "metrics_triple_vision_removed.json")


class FIQTripleVisionDataset(Dataset):
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
    }

    def __init__(
        self,
        dual_json: str,
        tokenizer,
        category: str,
        max_words: int = 77,
        max_samples: Optional[int] = None,
        skip_empty: bool = True,
        split: str = "val",
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.split = split
        llm_path = fiq_paths.llm_json(category, split)
        cap_path = fiq_paths.caption_json(category, split)
        if os.path.isfile(llm_path):
            with open(llm_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            src = "llm"
        else:
            with open(cap_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            src = "official_captions"
        dual = {}
        if dual_json and os.path.isfile(dual_json):
            with open(dual_json, "r", encoding="utf-8") as f:
                dual = json.load(f)

        self.queries = []
        n_skip_empty = 0
        n_vision_fallback = 0
        n_modified_fallback = 0
        for idx, row in enumerate(rows):
            if max_samples is not None and idx >= max_samples:
                break
            ref = str(row["candidate"])
            # Test captions have no target; keep unique placeholder so labels stay False
            # and duplicate candidates never collide on pair_key.
            if "target" in row and row["target"] is not None and str(row["target"]).strip():
                tgt = str(row["target"])
            else:
                tgt = f"__q{idx}__"
            key = fiq_paths.pair_key(ref, tgt)
            modified = str(row.get("ModifiedImageCaption") or "").strip()
            relative = _fiq_mod(row).strip()
            if not relative:
                n_skip_empty += 1
                continue
            if not modified:
                if skip_empty:
                    n_skip_empty += 1
                    continue
                modified = relative
                n_modified_fallback += 1
            vision = str((dual.get(key) or {}).get("vision_side") or "").strip()
            if not vision:
                vision = relative
                n_vision_fallback += 1
            self.queries.append(
                {
                    "idx": idx,
                    "vision": vision,
                    "modified": modified,
                    "relative": relative,
                    "removed": str(row.get("Removed") or ""),
                    "reference_name": ref,
                    "target_name": tgt,
                    "group_members": [ref, tgt],
                    "category": category,
                }
            )
        self.n_skip_empty = n_skip_empty
        self.n_vision_fallback = n_vision_fallback
        self.n_modified_fallback = n_modified_fallback
        print(
            f"[FIQTripleVisionDataset] split={split} src={src} cat={category} "
            f"queries={len(self.queries)} skip_empty={n_skip_empty} "
            f"modified_fallback={n_modified_fallback} vision_fallback={n_vision_fallback}"
        )

    def __len__(self):
        return len(self.queries)

    def _tokenize(self, text: str):
        words = self.tokenizer.tokenize(text or " ")
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        max_len = self.max_words - 1
        if len(words) > max_len:
            words = words[:max_len]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        return torch.LongTensor(input_ids), torch.LongTensor(input_mask)

    def __getitem__(self, index):
        q = self.queries[index]
        v_ids, v_mask = self._tokenize(q["vision"])
        m_ids, m_mask = self._tokenize(q["modified"])
        r_ids, r_mask = self._tokenize(q["relative"])
        return (
            v_ids, v_mask,
            m_ids, m_mask,
            r_ids, r_mask,
            q["reference_name"],
            q["target_name"],
            q["group_members"],
            q["idx"],
        )


def collate_fn(batch):
    tensors = [torch.stack([b[i] for b in batch]) for i in range(6)]
    refs = [b[6] for b in batch]
    tgts = [b[7] for b in batch]
    groups = [list(b[8]) for b in batch]
    idx = torch.tensor([b[9] for b in batch], dtype=torch.long)
    return (*tensors, refs, tgts, groups, idx)


def fiq_metrics_from_sim(sim, index_names, refs, tgts) -> Dict[str, float]:
    ranked = rank_by_similarity(sim, index_names)
    return compute_fiq_metrics(ranked, refs, tgts)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DESCS)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--removed_agg", type=str, default="max", choices=("mean", "max"))
    p.add_argument("--w_vision", type=float, default=BEST_W[0])
    p.add_argument("--w_modified", type=float, default=BEST_W[1])
    p.add_argument("--w_relative", type=float, default=BEST_W[2])
    p.add_argument("--mu", type=float, default=BEST_MU)
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=fiq_paths.CLIP_VITL14_CHECKPOINT)
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
    p.add_argument("--fiq_image_root", type=str, default=None)
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

    sim = {}
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

    wv, wm, wr, mu = args.w_vision, args.w_modified, args.w_relative, args.mu
    sim_triple = weighted_fuse([sim["vision"], sim["modified"], sim["relative"]], [wv, wm, wr])
    fused = fuse_triple_minus_removed(sim_triple, sim_removed, has_neg, float(mu))

    out = {
        "n_queries": len(refs),
        "n_gallery": len(index_names),
        "has_neg_ratio": float(has_neg.mean()),
        "vision": fiq_metrics_from_sim(sim["vision"], index_names, refs, tgts),
        "modified": fiq_metrics_from_sim(sim["modified"], index_names, refs, tgts),
        "relative": fiq_metrics_from_sim(sim["relative"], index_names, refs, tgts),
        "triple": fiq_metrics_from_sim(sim_triple, index_names, refs, tgts),
        "removed_only": fiq_metrics_from_sim(sim_removed, index_names, refs, tgts),
        "triple_minus_removed": fiq_metrics_from_sim(fused, index_names, refs, tgts),
    }
    for name in ("vision", "modified", "relative", "triple", "triple_minus_removed"):
        m = out[name]
        logger.info(
            "FIQ %s %s: R@10=%.2f R@50=%.2f",
            category, name, m["recall_at10"], m["recall_at50"],
        )
    return out


def mean_over_cats(per_cat: Dict[str, Dict], field: str) -> Dict[str, float]:
    cats = list(per_cat.keys())
    return {
        "recall_at10": float(np.mean([per_cat[c][field]["recall_at10"] for c in cats])),
        "recall_at50": float(np.mean([per_cat[c][field]["recall_at50"] for c in cats])),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    if "cirr" in os.path.abspath(args.output_json).lower() or "cirr" in os.path.abspath(args.init_model).lower():
        if "cirr" in os.path.abspath(args.output_json).lower():
            raise ValueError(f"FIQ eval must not write under a cirr path: {args.output_json}")
        logger.warning("init_model path contains 'cirr'; FIQ coarse should use fiq_retrieval best ckpt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    args.distributed = 0

    tokenizer = ClipTokenizer()
    model = HBICIRR(args)
    state = torch.load(args.init_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded %s | missing=%d unexpected=%d", args.init_model, len(missing), len(unexpected))
    model = model.to(device).eval()

    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    per_cat = {}
    for cat in cats:
        per_cat[cat] = eval_category(model, args, tokenizer, cat, device)

    summary = {
        "vision": mean_over_cats(per_cat, "vision"),
        "modified": mean_over_cats(per_cat, "modified"),
        "relative": mean_over_cats(per_cat, "relative"),
        "triple": mean_over_cats(per_cat, "triple"),
        "triple_minus_removed": mean_over_cats(per_cat, "triple_minus_removed"),
    }
    for name, m in summary.items():
        logger.info("FIQ mean %s: R@10=%.2f R@50=%.2f", name, m["recall_at10"], m["recall_at50"])

    out = {
        "init_model": args.init_model,
        "dual_json": args.dual_json,
        "formula": "S = 0.3*S_vision + 0.3*S_modified + 0.4*S_relative - 0.1*S_removed",
        "weights": {
            "w_vision": args.w_vision,
            "w_modified": args.w_modified,
            "w_relative": args.w_relative,
            "mu": args.mu,
        },
        "removed_agg": args.removed_agg,
        "mean": summary,
        "per_category": per_cat,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
