#!/usr/bin/env python3
"""CIRR val: fuse vision-side imagined desc + ModifiedImageCaption + relative caption.

BFSAR text↔image only (no gallery TT). Equal average + weight grid search.
Does not modify BFSAR v1 training code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr import cirr_paths
from bfsar.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader
from bfsar.cirr.eval_fgsa import encode_gallery, _similarity_matrix
from bfsar.cirr.metrics import compute_cirr_metrics, rank_by_similarity
from bfsar.cirr.modeling_fgsa import FGSAModel
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("triple_vision_eval")

DEFAULT_CKPT = str(ROOT / "weights" / "fgsa" / "cirr" / "pytorch_model.bin")
DEFAULT_DESCS = str(ROOT / "data" / "cirr" / "vision_desc" / "val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "metrics_triple_vision.json")


class CIRRTripleVisionDataset(Dataset):
    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
    }

    def __init__(
        self,
        dual_json: str,
        tokenizer,
        max_words: int = 77,
        max_samples: Optional[int] = None,
        split: str = "val",
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words
        file_split = "test1" if split in ("test", "test1") else "val"
        self.split = file_split
        with open(cirr_paths.caption_json(file_split), "r") as f:
            triplets = json.load(f)
        with open(dual_json, "r", encoding="utf-8") as f:
            dual = json.load(f)
        with open(cirr_paths.llm_json(file_split), "r") as f:
            llm = json.load(f)

        self.queries = []
        for idx, t in enumerate(triplets):
            if max_samples is not None and idx >= max_samples:
                break
            ref = t["reference"]
            if t.get("target_hard"):
                tgt = t["target_hard"]
                key = cirr_paths.pair_key(ref, tgt)
            else:
                tgt = ""
                key = cirr_paths.test1_llm_key(t.get("pairid"), ref)
            item = dual.get(key) or {}
            vision = str(item.get("vision_side") or "").strip()
            modified = str(llm.get(key, {}).get("ModifiedImageCaption") or "").strip()
            relative = str(t.get("caption") or "").strip()
            if file_split == "test1":
                # Test1 keeps every official pair. An empty generated caption is
                # filled from the human caption, and an empty human caption from
                # the generated one. Val still drops a pair when either side is empty.
                if modified and not relative:
                    relative = modified
                elif relative and not modified:
                    modified = relative
                if not relative:
                    continue
                if not modified:
                    modified = relative
            elif not modified or not relative:
                continue
            if not vision:
                vision = relative  # fallback; shouldn't happen after full gen
            self.queries.append(
                {
                    "idx": idx,
                    "pairid": int(t["pairid"]) if t.get("pairid") is not None else -1,
                    "vision": vision,
                    "modified": modified,
                    "relative": relative,
                    "reference_name": ref,
                    "target_name": tgt,
                    "group_members": t["img_set"]["members"],
                }
            )
        print(f"[CIRRTripleVisionDataset] queries={len(self.queries)}")

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


@torch.no_grad()
def encode_three(model, loader, device):
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    buckets = {k: {"feats": [], "cls": [], "masks": []} for k in ("vision", "modified", "relative")}
    refs, tgts, groups = [], [], []
    for batch in tqdm(loader, desc="encode triple queries"):
        (
            v_ids, v_mask, m_ids, m_mask, r_ids, r_mask,
            ref_names, tgt_names, group_members, _,
        ) = batch
        for key, ids, mask in (
            ("vision", v_ids, v_mask),
            ("modified", m_ids, m_mask),
            ("relative", r_ids, r_mask),
        ):
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            feat, cls = model.get_text_feat(ids, mask)
            buckets[key]["feats"].append(feat.cpu())
            buckets[key]["cls"].append(cls.cpu())
            buckets[key]["masks"].append(mask.cpu())
        refs.extend(list(ref_names))
        tgts.extend(list(tgt_names))
        for g in group_members:
            groups.append(list(g))
    out = {}
    for key, d in buckets.items():
        out[key] = (
            torch.cat(d["feats"], dim=0),
            torch.cat(d["cls"], dim=0),
            torch.cat(d["masks"], dim=0),
        )
    return out, refs, tgts, groups


def _metrics(sim, index_names, refs, tgts, groups):
    ranked = rank_by_similarity(sim, index_names)
    m = compute_cirr_metrics(ranked, refs, tgts, groups)
    # Extended classic recalls (not in v1 metrics helper).
    n = len(refs)
    hits100 = 0
    hits200 = 0
    for i in range(n):
        names = ranked[i]
        ref = refs[i]
        tgt = tgts[i]
        filtered = [x for x in names if x != ref]
        try:
            rank = filtered.index(tgt) + 1
        except ValueError:
            continue
        if rank <= 100:
            hits100 += 1
        if rank <= 200:
            hits200 += 1
    m["recall_at100"] = 100.0 * hits100 / max(n, 1)
    m["recall_at200"] = 100.0 * hits200 / max(n, 1)
    return m


def weighted_fuse(sims: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    fused = np.zeros_like(sims[0], dtype=np.float64)
    for s, wi in zip(sims, w):
        fused += wi * s.astype(np.float64)
    return fused.astype(np.float32)


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
    p.add_argument("--weight_step", type=float, default=0.1,
                   help="Grid step for (w_vision, w_modified, w_relative) with sum=1")
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

    results = {
        "vision_only": _metrics(sim["vision"], index_names, refs, tgts, groups),
        "modified_only": _metrics(sim["modified"], index_names, refs, tgts, groups),
        "relative_only": _metrics(sim["relative"], index_names, refs, tgts, groups),
    }

    # Two-way baselines for context
    results["mod_rel_equal"] = _metrics(
        weighted_fuse([sim["modified"], sim["relative"]], [0.5, 0.5]),
        index_names, refs, tgts, groups,
    )
    results["vision_mod_rel_equal"] = _metrics(
        weighted_fuse([sim["vision"], sim["modified"], sim["relative"]], [1 / 3, 1 / 3, 1 / 3]),
        index_names, refs, tgts, groups,
    )

    # Weight grid: w_v + w_m + w_r = 1, step
    step = float(args.weight_step)
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    grid = []
    best = None
    for wv, wm in itertools.product(vals, vals):
        wr = 1.0 - wv - wm
        if wr < -1e-9:
            continue
        if abs(wr) < 1e-9:
            wr = 0.0
        if wr < 0 or wr > 1.0 + 1e-9:
            continue
        fused = weighted_fuse([sim["vision"], sim["modified"], sim["relative"]], [wv, wm, wr])
        m = _metrics(fused, index_names, refs, tgts, groups)
        row = {
            "w_vision": round(float(wv), 4),
            "w_modified": round(float(wm), 4),
            "w_relative": round(float(wr), 4),
            "recall_at1": m["recall_at1"],
            "recall_at5": m["recall_at5"],
            "group_recall_at1": m["group_recall_at1"],
        }
        grid.append(row)
        if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
            abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
            and m["group_recall_at1"] > best["group_recall_at1"]
        ):
            best = {**row, **{k: m[k] for k in m}}

    grid.sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))
    results["vision_mod_rel_best_weight"] = {
        k: best[k] for k in (
            "recall_at1", "recall_at5", "recall_at10", "recall_at50",
            "recall_at100", "recall_at200",
            "group_recall_at1", "group_recall_at2", "group_recall_at3",
        ) if k in best
    }
    results["vision_mod_rel_best_weight"]["weights"] = {
        "w_vision": best["w_vision"],
        "w_modified": best["w_modified"],
        "w_relative": best["w_relative"],
    }

    for name, m in results.items():
        if "weights" in m:
            logger.info(
                "%s | weights=%s | R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f | G-R@1=%.2f",
                name, m["weights"],
                m["recall_at1"], m["recall_at5"], m["recall_at10"], m["recall_at50"],
                m["group_recall_at1"],
            )
        else:
            logger.info(
                "%s | R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f | G-R@1=%.2f G-R@2=%.2f G-R@3=%.2f",
                name,
                m["recall_at1"], m["recall_at5"], m["recall_at10"], m["recall_at50"],
                m["group_recall_at1"], m["group_recall_at2"], m["group_recall_at3"],
            )

    logger.info("Top-10 weight combos by R@1:")
    for row in grid[:10]:
        logger.info(
            "  wv=%.2f wm=%.2f wr=%.2f -> R@1=%.2f G-R@1=%.2f",
            row["w_vision"], row["w_modified"], row["w_relative"],
            row["recall_at1"], row["group_recall_at1"],
        )

    out = {
        "n_queries": len(refs),
        "init_model": args.init_model,
        "dual_json": args.dual_json,
        "max_words": args.max_words,
        "weight_step": step,
        "metrics": results,
        "weight_grid_top20": grid[:20],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
