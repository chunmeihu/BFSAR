#!/usr/bin/env python3
"""CIRR val: S = S_base + λ·S_pos - μ·S_neg with BFSAR text→gallery similarity.

- S_base: llm_info ModifiedImageCaption (target description)
- S_pos:  concatenated existent objects (one string per query)
- S_neg:  mean similarity over each nonexistent object phrase separately
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr import cirr_paths
from bfsar.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader
from bfsar.cirr.eval_fgsa import encode_gallery, _similarity_matrix
from bfsar.cirr.metrics import compute_cirr_metrics, rank_by_similarity
from bfsar.cirr.modeling_fgsa import FGSAModel
from bfsar.cirr.vision_desc.pos_neg_utils import split_legacy_constraints
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("pos_neg_eval")

DEFAULT_CKPT = str(ROOT / "weights" / "fgsa" / "cirr" / "pytorch_model.bin")
DEFAULT_POS_NEG = str(ROOT / "output" / "cirr_vision_desc" / "val_pos_neg.json")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "metrics_pos_neg.json")


class CIRRPosNegDataset(Dataset):
    """Load S_base / S_pos / S_neg from llm_info or optional Qwen pos_neg JSON."""

    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
    }

    def __init__(
        self,
        tokenizer,
        max_words: int = 77,
        max_samples: Optional[int] = None,
        constraint_source: str = "llm_info",
        pos_neg_json: Optional[str] = None,
        llm_json: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.constraint_source = constraint_source

        with open(cirr_paths.caption_json("val"), "r") as f:
            triplets = json.load(f)
        llm_path = llm_json or cirr_paths.llm_json("val")
        with open(llm_path, "r") as f:
            llm = json.load(f)

        pos_neg: Dict[str, dict] = {}
        if constraint_source == "qwen" and pos_neg_json and Path(pos_neg_json).is_file():
            with open(pos_neg_json, "r", encoding="utf-8") as f:
                pos_neg = json.load(f)

        self.queries = []
        for idx, t in enumerate(triplets):
            if max_samples is not None and idx >= max_samples:
                break
            ref = t["reference"]
            tgt = t["target_hard"]
            key = cirr_paths.pair_key(ref, tgt)
            llm_item = llm.get(key) or {}
            base = str(llm_item.get("ModifiedImageCaption") or "").strip()
            if not base:
                continue

            if constraint_source == "qwen":
                pn = pos_neg.get(key) or {}
                pos_concat = str(pn.get("pos_concat") or "").strip()
                neg_items = [str(x).strip() for x in (pn.get("neg_items") or []) if str(x).strip()]
                if not pos_concat:
                    pos_concat = str(llm_item.get("Retained") or "").strip()
                if not neg_items:
                    neg_items = split_legacy_constraints(llm_item.get("Removed") or "")
            else:
                # llm_info: Retained as one string; Removed split into neg phrases
                pos_concat = str(llm_item.get("Retained") or "").strip()
                neg_items = split_legacy_constraints(llm_item.get("Removed") or "")

            self.queries.append(
                {
                    "idx": idx,
                    "base": base,
                    "pos_concat": pos_concat,
                    "neg_items": neg_items,
                    "reference_name": ref,
                    "target_name": tgt,
                    "group_members": t["img_set"]["members"],
                }
            )
        n = len(self.queries)
        has_pos = sum(1 for q in self.queries if q["pos_concat"])
        has_neg = sum(1 for q in self.queries if q["neg_items"])
        print(
            f"[CIRRPosNegDataset] source={constraint_source} queries={n} "
            f"has_retained={has_pos} has_removed={has_neg}"
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
        base_ids, base_mask = self._tokenize(q["base"])
        pos_ids, pos_mask = self._tokenize(q["pos_concat"] or " ")
        has_pos = bool(q["pos_concat"] and q["pos_concat"].strip())
        return (
            base_ids,
            base_mask,
            pos_ids,
            pos_mask,
            torch.tensor(1 if has_pos else 0, dtype=torch.float32),
            q["neg_items"],
            q["reference_name"],
            q["target_name"],
            q["group_members"],
            q["idx"],
        )


def collate_fn(batch):
    base_ids = torch.stack([b[0] for b in batch])
    base_mask = torch.stack([b[1] for b in batch])
    pos_ids = torch.stack([b[2] for b in batch])
    pos_mask = torch.stack([b[3] for b in batch])
    has_pos = torch.stack([b[4] for b in batch])
    neg_items = [list(b[5]) for b in batch]
    refs = [b[6] for b in batch]
    tgts = [b[7] for b in batch]
    groups = [list(b[8]) for b in batch]
    idx = torch.tensor([b[9] for b in batch], dtype=torch.long)
    return base_ids, base_mask, pos_ids, pos_mask, has_pos, neg_items, refs, tgts, groups, idx


@torch.no_grad()
def _encode_texts(model, ids, mask, device):
    if hasattr(model, "module"):
        model = model.module
    ids = ids.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    feat, cls = model.get_text_feat(ids, mask)
    return feat.cpu(), cls.cpu(), mask.cpu()


@torch.no_grad()
def encode_base_pos(
    model, loader, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str], List, List[List[str]]]:
    base_feats, base_cls, base_masks = [], [], []
    pos_feats, pos_cls, pos_masks = [], [], []
    refs, tgts, groups, neg_items_all = [], [], [], []

    for batch in tqdm(loader, desc="encode base+pos"):
        base_ids, base_mask, pos_ids, pos_mask, _, neg_items, ref_names, tgt_names, group_members, _ = batch
        bf, bc, bm = _encode_texts(model, base_ids, base_mask, device)
        pf, pc, pm = _encode_texts(model, pos_ids, pos_mask, device)
        base_feats.append(bf)
        base_cls.append(bc)
        base_masks.append(bm)
        pos_feats.append(pf)
        pos_cls.append(pc)
        pos_masks.append(pm)
        refs.extend(list(ref_names))
        tgts.extend(list(tgt_names))
        for g in group_members:
            groups.append(list(g))
        neg_items_all.extend(neg_items)

    return (
        torch.cat(base_feats, dim=0),
        torch.cat(base_cls, dim=0),
        torch.cat(base_masks, dim=0),
        torch.cat(pos_feats, dim=0),
        torch.cat(pos_cls, dim=0),
        torch.cat(pos_masks, dim=0),
        refs,
        tgts,
        groups,
        neg_items_all,
    )


@torch.no_grad()
def encode_neg_phrase_sim(
    model,
    neg_items_all: List[List[str]],
    gallery_feats,
    gallery_masks,
    device,
    query_batch: int,
    gallery_batch: int,
    agg: str = "mean",
) -> np.ndarray:
    """BFSAR sim from Removed phrases. agg: mean | max over comma-split phrases."""
    n_q = len(neg_items_all)
    n_g = gallery_feats.shape[0]
    out = np.zeros((n_q, n_g), dtype=np.float32)

    flat_texts: List[str] = []
    flat_map: List[int] = []
    for qi, items in enumerate(neg_items_all):
        for item in items:
            flat_texts.append(item)
            flat_map.append(qi)

    if not flat_texts:
        return out

    tokenizer = ClipTokenizer()
    max_words = 77
    special = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>"}

    def tokenize_one(text: str):
        words = tokenizer.tokenize(text or " ")
        words = [special["CLS_TOKEN"]] + words
        if len(words) > max_words - 1:
            words = words[: max_words - 1]
        words = words + [special["SEP_TOKEN"]]
        input_ids = tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_words:
            input_ids.append(0)
            input_mask.append(0)
        return torch.LongTensor(input_ids), torch.LongTensor(input_mask)

    batch_size = 64
    flat_sims: List[np.ndarray] = []
    for start in tqdm(range(0, len(flat_texts), batch_size), desc=f"encode neg phrases ({agg})"):
        chunk = flat_texts[start : start + batch_size]
        ids_list, mask_list = [], []
        for t in chunk:
            tid, tmask = tokenize_one(t)
            ids_list.append(tid)
            mask_list.append(tmask)
        ids = torch.stack(ids_list)
        mask = torch.stack(mask_list)
        feat, cls, m = _encode_texts(model, ids, mask, device)
        sim = _similarity_matrix(
            model, feat, cls, m, gallery_feats, gallery_masks, device, query_batch, gallery_batch
        )
        flat_sims.append(sim)

    flat_sim = np.concatenate(flat_sims, axis=0)
    if agg == "max":
        for fi, qi in enumerate(flat_map):
            out[qi] = np.maximum(out[qi], flat_sim[fi])
    elif agg == "mean":
        counts = np.zeros(n_q, dtype=np.int32)
        for fi, qi in enumerate(flat_map):
            out[qi] += flat_sim[fi]
            counts[qi] += 1
        for qi in range(n_q):
            if counts[qi] > 0:
                out[qi] /= counts[qi]
    else:
        raise ValueError(f"unsupported agg={agg!r}, use mean or max")
    return out


def encode_neg_mean_sim(
    model,
    neg_items_all: List[List[str]],
    gallery_feats,
    gallery_masks,
    device,
    query_batch: int,
    gallery_batch: int,
) -> np.ndarray:
    return encode_neg_phrase_sim(
        model, neg_items_all, gallery_feats, gallery_masks, device,
        query_batch, gallery_batch, agg="mean",
    )


def _metrics(sim, index_names, refs, tgts, groups):
    ranked = rank_by_similarity(sim, index_names)
    return compute_cirr_metrics(ranked, refs, tgts, groups)


def fuse_pos_neg(
    sim_base: np.ndarray,
    sim_pos: np.ndarray,
    sim_neg: np.ndarray,
    has_pos: np.ndarray,
    has_neg: np.ndarray,
    lam: float,
    mu: float,
) -> np.ndarray:
    fused = sim_base.astype(np.float64).copy()
    if lam != 0:
        mask = has_pos.astype(np.float64)[:, None]
        fused += lam * sim_pos.astype(np.float64) * mask
    if mu != 0:
        mask = has_neg.astype(np.float64)[:, None]
        fused -= mu * sim_neg.astype(np.float64) * mask
    return fused.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument(
        "--constraint_source",
        type=str,
        default="llm_info",
        choices=["llm_info", "qwen"],
        help="llm_info: Retained/Removed from llm_info_cirr_val.json; qwen: val_pos_neg.json",
    )
    p.add_argument("--pos_neg_json", type=str, default=DEFAULT_POS_NEG)
    p.add_argument("--llm_json", type=str, default=None)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--lambda_pos", type=float, default=1.0)
    p.add_argument("--mu_neg", type=float, default=0.3)
    p.add_argument(
        "--grid_search",
        action="store_true",
        help="Search best (lambda, mu); default when neither --lambda_pos nor --mu_neg overridden",
    )
    p.add_argument("--weight_step", type=float, default=0.1)
    p.add_argument(
        "--empty_retained",
        type=str,
        default="skip",
        choices=["skip"],
        help="When Retained is empty: skip lambda*S_pos for that query (recommended)",
    )
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
    dataset = CIRRPosNegDataset(
        tokenizer,
        max_words=args.max_words,
        max_samples=args.max_samples,
        constraint_source=args.constraint_source,
        pos_neg_json=args.pos_neg_json,
        llm_json=args.llm_json,
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

    (
        base_feats, base_cls, base_masks,
        pos_feats, pos_cls, pos_masks,
        refs, tgts, groups, neg_items_all,
    ) = encode_base_pos(model, loader, device)

    with torch.no_grad():
        sim_base = _similarity_matrix(
            model, base_feats, base_cls, base_masks, gallery_feats, gallery_masks, device,
            args.query_batch, args.gallery_batch,
        )
        sim_pos = _similarity_matrix(
            model, pos_feats, pos_cls, pos_masks, gallery_feats, gallery_masks, device,
            args.query_batch, args.gallery_batch,
        )
    sim_neg = encode_neg_mean_sim(
        model, neg_items_all, gallery_feats, gallery_masks, device,
        args.query_batch, args.gallery_batch,
    )

    has_pos = np.array([1 if (q["pos_concat"] and q["pos_concat"].strip()) else 0 for q in dataset.queries])
    has_neg = np.array([1 if q["neg_items"] else 0 for q in dataset.queries])

    results = {
        "base_only": _metrics(sim_base, index_names, refs, tgts, groups),
        "pos_only": _metrics(sim_pos, index_names, refs, tgts, groups),
        "neg_only": _metrics(sim_neg, index_names, refs, tgts, groups),
    }

    lam = float(args.lambda_pos)
    mu = float(args.mu_neg)
    grid = []
    do_grid = args.grid_search
    if do_grid:
        step = float(args.weight_step)
        vals = np.arange(0.0, 1.0 + 1e-9, step)
        best = None
        for gl, gm in itertools.product(vals, vals):
            fused = fuse_pos_neg(sim_base, sim_pos, sim_neg, has_pos, has_neg, float(gl), float(gm))
            m = _metrics(fused, index_names, refs, tgts, groups)
            row = {
                "lambda": round(float(gl), 4),
                "mu": round(float(gm), 4),
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
        results["base_plus_pos_minus_neg_best"] = {
            k: best[k]
            for k in (
                "recall_at1", "recall_at5", "recall_at10", "recall_at50",
                "group_recall_at1", "group_recall_at2", "group_recall_at3",
            )
            if k in best
        }
        results["base_plus_pos_minus_neg_best"]["weights"] = {
            "lambda": best["lambda"],
            "mu": best["mu"],
        }
    else:
        fused = fuse_pos_neg(sim_base, sim_pos, sim_neg, has_pos, has_neg, lam, mu)
        m = _metrics(fused, index_names, refs, tgts, groups)
        results["base_plus_pos_minus_neg"] = {
            k: m[k]
            for k in (
                "recall_at1", "recall_at5", "recall_at10", "recall_at50",
                "group_recall_at1", "group_recall_at2", "group_recall_at3",
            )
        }
        results["base_plus_pos_minus_neg"]["weights"] = {"lambda": lam, "mu": mu}

    for name, m in results.items():
        if "weights" in m:
            logger.info(
                "%s | λ=%.2f μ=%.2f | R@1=%.2f R@5=%.2f | G-R@1=%.2f",
                name, m["weights"]["lambda"], m["weights"]["mu"],
                m["recall_at1"], m["recall_at5"], m["group_recall_at1"],
            )
        else:
            logger.info(
                "%s | R@1=%.2f R@5=%.2f R@10=%.2f | G-R@1=%.2f",
                name, m["recall_at1"], m["recall_at5"], m["recall_at10"], m["group_recall_at1"],
            )

    grid.sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))
    if do_grid:
        logger.info("Top-10 (λ, μ) by R@1:")
        for row in grid[:10]:
            logger.info(
                "  λ=%.2f μ=%.2f -> R@1=%.2f G-R@1=%.2f",
                row["lambda"], row["mu"], row["recall_at1"], row["group_recall_at1"],
            )

    out = {
        "n_queries": len(refs),
        "init_model": args.init_model,
        "constraint_source": args.constraint_source,
        "llm_json": args.llm_json or cirr_paths.llm_json("val"),
        "pos_neg_json": args.pos_neg_json if args.constraint_source == "qwen" else None,
        "formula": "S = S_base + lambda * S_pos - mu * S_neg",
        "s_base_source": "llm_info ModifiedImageCaption",
        "s_pos_source": "llm_info Retained (one string; skip term if empty)",
        "s_neg_source": "llm_info Removed split by comma, mean sim (skip term if empty)",
        "empty_retained_policy": args.empty_retained,
        "has_pos_ratio": float(has_pos.mean()),
        "has_neg_ratio": float(has_neg.mean()),
        "lambda": lam if not do_grid else None,
        "mu": mu if not do_grid else None,
        "grid_search": bool(do_grid),
        "metrics": results,
    }
    if do_grid:
        out["weight_step"] = float(args.weight_step)
        out["weight_grid_top20"] = grid[:20]
        out["lambda"] = results.get("base_plus_pos_minus_neg_best", {}).get("weights", {}).get("lambda")
        out["mu"] = results.get("base_plus_pos_minus_neg_best", {}).get("weights", {}).get("mu")
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
