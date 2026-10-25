#!/usr/bin/env python3
"""Fashion-IQ: HBI fused coarse Top-200 + official CR Stage-2 rerank (K=100).

Uses Candidate-Reranking-CIR Fashion-IQ checkpoints (blip.pt), not CIRR.
Does not write under cirr output paths. Does not retune fusion weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.fiq import fiq_paths
from HBI.fiq.eval_fiq_hbi import compute_fiq_metrics
from HBI.fiq.fiq_dataloader_builders import build_fiq_gallery_dataloader
from HBI.fiq.vision_desc.eval_fiq_hbi_triple import (
    FIQTripleVisionDataset,
    collate_fn as hbi_collate,
)
from HBI.cirr.eval_cirr_hbi import encode_gallery, _similarity_matrix
from HBI.cirr.metrics_cirr_hbi import rank_by_similarity
from HBI.cirr.modeling_hbi_cirr import HBICIRR
from HBI.cirr.vision_desc.eval_hbi_pos_neg import encode_neg_phrase_sim
from HBI.cirr.vision_desc.eval_hbi_triple_vision import encode_three, weighted_fuse
from HBI.cirr.vision_desc.eval_hbi_triple_vision_removed import (
    fuse_triple_minus_removed,
    prepare_removed_sim,
)
from HBI.cirr.vision_desc.hbi_blip_rerank import (
    load_blip_models,
    merge_rerank_with_coarse_gallery,
)
from HBI.cirr.vision_desc.pos_neg_utils import split_legacy_constraints
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from HBI.rerank.paths import DEFAULT_FIQ_STAGE1, DEFAULT_FIQ_STAGE2

logger = logging.getLogger("fiq_hbi_cr_rerank")

DEFAULT_HBI = str(ROOT / "output" / "fiq_retrieval" / "pytorch_model.bin.best.5")
DEFAULT_DUAL = str(ROOT / "data" / "fashioniq" / "vision_desc" / "val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output" / "fiq_vision_desc" / "metrics_hbi_cr_rerank.json")
DEFAULT_STAGE1 = str(DEFAULT_FIQ_STAGE1)
DEFAULT_STAGE2 = str(DEFAULT_FIQ_STAGE2)


def _abort_cirr_paths(*paths: str) -> None:
    for p in paths:
        if "cirr" in os.path.abspath(p).lower():
            raise ValueError(f"FIQ rerank must not use a cirr path: {p}")


def recall_at_ks(
    sorted_index_names: np.ndarray,
    refs: Sequence[str],
    tgts: Sequence[str],
    ks: Sequence[int] = (10, 50, 100, 200),
) -> Dict[str, float]:
    base = compute_fiq_metrics(sorted_index_names, refs, tgts)
    n_queries = len(tgts)
    ranks = np.empty(n_queries, dtype=np.int64)
    for i in range(n_queries):
        ranking = [name for name in sorted_index_names[i] if name != refs[i]]
        try:
            ranks[i] = ranking.index(tgts[i])
        except ValueError:
            ranks[i] = len(ranking)
    out = dict(base)
    for k in ks:
        out[f"recall_at{k}"] = float((ranks < k).mean() * 100.0)
    return out


def cr_joined_caption(captions: Sequence[str]) -> str:
    parts = [str(c).strip(".?, ").strip() for c in (captions or []) if str(c).strip()]
    if len(parts) >= 2:
        return f"{parts[0].capitalize()} and {parts[1]}"
    if parts:
        return parts[0].capitalize()
    return " "


class FIQClassicImages(Dataset):
    def __init__(self, names: Sequence[str], preprocess):
        self.names = list(names)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        path = fiq_paths.image_path(name)
        if path is None:
            raise FileNotFoundError(f"Missing FIQ image: {name}")
        img = self.preprocess(Image.open(path).convert("RGB"))
        return name, img


def encode_hbi_fused(model, args, tokenizer, category: str, device):
    split = getattr(args, "split", "val")
    dataset = FIQTripleVisionDataset(
        args.dual_json,
        tokenizer,
        category=category,
        max_words=args.max_words,
        max_samples=args.max_samples,
        skip_empty=False,
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=hbi_collate,
    )
    gallery_loader, n_g, _ = build_fiq_gallery_dataloader(args, split=split, category=category)
    logger.info("HBI encode %s gallery=%d queries=%d", category, n_g, len(dataset))
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
    sim_triple = weighted_fuse(
        [sim["vision"], sim["modified"], sim["relative"]],
        [args.w_vision, args.w_modified, args.w_relative],
    )
    fused = fuse_triple_minus_removed(sim_triple, sim_removed, has_neg, float(args.mu))
    return {
        "sim": np.asarray(fused, dtype=np.float32),
        "index_names": list(index_names),
        "refs": list(refs),
        "tgts": list(tgts),
    }


def align_to_official(
    hbi: Dict, category: str, split: str = "val",
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    with open(fiq_paths.caption_json(category, split), "r", encoding="utf-8") as f:
        official = json.load(f)
    # Test (no target): HBI rows already follow official caption order.
    if not official or "target" not in official[0]:
        if len(hbi["refs"]) != len(official):
            raise RuntimeError(
                f"{category}/{split}: HBI queries {len(hbi['refs'])} != official {len(official)}"
            )
        return (
            np.asarray(hbi["sim"], dtype=np.float32),
            list(hbi["index_names"]),
            list(hbi["refs"]),
            list(hbi["tgts"]),
        )
    key_to_row = {fiq_paths.pair_key(r, t): i for i, (r, t) in enumerate(zip(hbi["refs"], hbi["tgts"]))}
    index_names = hbi["index_names"]
    name_arr = np.array(index_names)
    sim = hbi["sim"]
    aligned = []
    refs, tgts = [], []
    n_fallback = 0
    for row in official:
        ref = str(row["candidate"])
        tgt = str(row["target"])
        refs.append(ref)
        tgts.append(tgt)
        key = fiq_paths.pair_key(ref, tgt)
        if key in key_to_row:
            aligned.append(sim[key_to_row[key]])
        else:
            n_fallback += 1
            order = np.argsort(name_arr)
            scores = np.full(len(index_names), -1e6, dtype=np.float32)
            scores[order] = np.linspace(0.0, -1.0, num=len(index_names), dtype=np.float32)
            if ref in {n: i for i, n in enumerate(index_names)}:
                scores[index_names.index(ref)] = -1e9
            aligned.append(scores)
    if n_fallback:
        logger.warning("%s official queries without HBI row: %d (alpha fallback)", category, n_fallback)
    return np.stack(aligned, axis=0), index_names, refs, tgts


def build_topk_payload(
    sim: np.ndarray,
    index_names: List[str],
    refs: List[str],
    tgts: List[str],
    category: str,
    k_save: int,
    split: str = "val",
) -> Dict:
    sorted_names = rank_by_similarity(sim, index_names)
    n_q, n_g = sorted_names.shape
    no_ref = []
    labels = []
    for i in range(n_q):
        ranking = [str(n) for n in sorted_names[i] if str(n) != str(refs[i])]
        if len(ranking) < k_save:
            raise ValueError(f"{category} query {i}: ranking {len(ranking)} < k_save {k_save}")
        no_ref.append(ranking[:k_save])
        labels.append([n == str(tgts[i]) for n in ranking[:k_save]])
    return {
        "sorted_index_names": np.array(no_ref),
        "labels": torch.tensor(np.asarray(labels, dtype=bool)),
        "target_names": list(tgts),
        "index_names": list(index_names),
        "dress_types": category,
        "split": split,
        "source": "hbi_triple_minus_removed",
        "reference_names": list(refs),
    }


def extract_blip_index_cpu(model_stage2, names: List[str], preprocess, batch_size: int, num_workers: int, device):
    ds = FIQClassicImages(names, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    chunks = []
    out_names = []
    model_stage2.eval()
    with torch.no_grad():
        for batch_names, images in tqdm(loader, desc="BLIP gallery embed"):
            images = images.to(device, non_blocking=True)
            feats = model_stage2.img_embed(images)
            chunks.append(feats.detach().cpu())
            out_names.extend(list(batch_names))
    return torch.cat(chunks, dim=0), out_names


def rerank_category(
    payload: Dict,
    model_stage1,
    model_stage2,
    preprocess,
    category: str,
    k_rerank: int,
    device,
    num_workers: int,
    gallery_batch: int,
    cand_chunk: int = 50,
    split: str = "val",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    split = str(payload.get("split") or split)
    with open(fiq_paths.caption_json(category, split), "r", encoding="utf-8") as f:
        official = json.load(f)
    refs = [str(x) for x in payload["reference_names"]]
    tgts = [str(x) for x in payload["target_names"]]
    # Val: map by (ref, tgt). Test: no target + duplicate candidates → row order.
    if official and "target" in official[0]:
        cap_by_pair = {
            (str(row["candidate"]), str(row["target"])): cr_joined_caption(row.get("captions") or [])
            for row in official
        }
        captions = []
        missing_cap = 0
        for r, t in zip(refs, tgts):
            cap = cap_by_pair.get((r, t))
            if cap is None:
                missing_cap += 1
                captions.append(" ")
            else:
                captions.append(cap)
        if missing_cap:
            raise RuntimeError(
                f"{category}: {missing_cap}/{len(refs)} payload pairs missing from official captions"
            )
    else:
        if len(official) != len(refs):
            raise RuntimeError(
                f"{category}/{split}: official {len(official)} != payload {len(refs)}"
            )
        for i, row in enumerate(official):
            if str(row["candidate"]) != refs[i]:
                raise RuntimeError(
                    f"{category}/{split}: candidate order mismatch at {i}: "
                    f"{row['candidate']} vs {refs[i]}"
                )
        captions = [cr_joined_caption(row.get("captions") or []) for row in official]

    index_names = list(payload["index_names"])
    topk = np.array(payload["sorted_index_names"], dtype=object)[:, :k_rerank]
    labels = payload["labels"].numpy()[:, :k_rerank] if torch.is_tensor(payload["labels"]) else np.asarray(payload["labels"])[:, :k_rerank]

    gallery_feats, feat_names = extract_blip_index_cpu(
        model_stage2, index_names, preprocess, gallery_batch, num_workers, device,
    )
    name_to_i = {n: i for i, n in enumerate(feat_names)}
    name_to_feat = {}
    for n, i in name_to_i.items():
        name_to_feat[n] = gallery_feats[i]

    # Test has no labels: always score. Val keeps previous in-pool gate.
    force_score = split == "test" or not labels.any()

    pred = []
    model_stage1.eval()
    model_stage2.eval()
    for i in tqdm(range(len(refs)), desc=f"CR rerank {category}"):
        in_pool = force_score or bool(labels[i].any())
        cand_names = [str(x) for x in list(topk[i])]
        if in_pool:
            ref_feat = name_to_feat[refs[i]].unsqueeze(0).to(device)
            z_t = model_stage1.img_txt_fusion(
                ref_feat, None, [captions[i]], train=False, return_raw=True
            )
            parts = []
            chunk = max(int(cand_chunk), 1)
            for s in range(0, len(cand_names), chunk):
                e = min(s + chunk, len(cand_names))
                cand = torch.stack([name_to_feat[n] for n in cand_names[s:e]]).to(device)
                part = model_stage2.img_txt_fusion_val(z_t, cand, [captions[i]]).detach().cpu().numpy()
                parts.append(np.asarray(part).reshape(-1))
            logits = np.concatenate(parts, axis=0).astype(np.float32)
        else:
            logits = np.full((k_rerank,), -1e5, dtype=np.float32)
        pred.append(logits)

    pred = np.stack(pred, axis=0)
    rerank_idx = np.argsort(-pred, axis=1)
    reranked = np.take_along_axis(topk, rerank_idx, axis=1)
    pool_hits = np.take_along_axis(labels, rerank_idx, axis=1)

    def r_at(arr, k):
        k = min(k, arr.shape[1])
        return float(arr[:, :k].sum() / len(arr) * 100.0)

    pool_metrics = {
        "coverage": float(labels.any(axis=1).mean() * 100.0),
        "pool_recall_at10": r_at(pool_hits, 10),
        "pool_recall_at50": r_at(pool_hits, 50),
        "pool_recall_at100": r_at(pool_hits, 100),
        "pool_recall_at300": r_at(pool_hits, 300),
    }
    return reranked, pred, pool_metrics


def parse_args():
    p = argparse.ArgumentParser(description="FIQ HBI Top-200 + CR Stage-2 K=100")
    p.add_argument("--init_model", type=str, default=DEFAULT_HBI)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DUAL)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--stage1_ckpt", type=str, default=DEFAULT_STAGE1)
    p.add_argument("--stage2_ckpt", type=str, default=DEFAULT_STAGE2)
    p.add_argument("--out_dir", type=str, default=str(ROOT / "output" / "fiq_vision_desc"))
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--k_save", type=int, default=200)
    p.add_argument("--k_rerank", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--cr_gallery_batch", type=int, default=16)
    p.add_argument("--removed_agg", type=str, default="max")
    p.add_argument("--w_vision", type=float, default=0.3)
    p.add_argument("--w_modified", type=float, default=0.3)
    p.add_argument("--w_relative", type=float, default=0.4)
    p.add_argument("--mu", type=float, default=0.1)
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
    p.add_argument("--skip_hbi", action="store_true")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    _abort_cirr_paths(args.output_json, args.init_model, args.out_dir, args.stage1_ckpt, args.stage2_ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    args.distributed = 0
    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    k_save, k_rerank = int(args.k_save), int(args.k_rerank)

    tokenizer = ClipTokenizer()
    per_cat = {}
    payloads = {}

    need_hbi = (not args.skip_hbi) or any(
        not (out_dir / f"hbi_fused_{c}.npz").exists() for c in cats
    )
    if need_hbi:
        model = HBICIRR(args)
        state = torch.load(args.init_model, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        logger.info("Loaded HBI %s missing=%d unexpected=%d", args.init_model, len(missing), len(unexpected))
        model = model.to(device).eval()
        for cat in cats:
            hbi = encode_hbi_fused(model, args, tokenizer, cat, device)
            np.savez_compressed(
                out_dir / f"hbi_fused_{cat}.npz",
                sim=hbi["sim"],
                index_names=np.array(hbi["index_names"], dtype=object),
                refs=np.array(hbi["refs"], dtype=object),
                tgts=np.array(hbi["tgts"], dtype=object),
            )
            sim, index_names, refs, tgts = align_to_official(hbi, cat)
            ranked = rank_by_similarity(sim, index_names)
            coarse = recall_at_ks(ranked, refs, tgts)
            logger.info(
                "HBI fused %s R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f",
                cat, coarse["recall_at10"], coarse["recall_at50"],
                coarse["recall_at100"], coarse["recall_at200"],
            )
            payload = build_topk_payload(sim, index_names, refs, tgts, cat, k_save)
            pt_path = out_dir / f"hbi_top{k_save}_val_{cat}.pt"
            torch.save(payload, pt_path)
            payloads[cat] = payload
            per_cat[cat] = {"coarse": coarse, "topk_pt": str(pt_path), "n_queries": len(refs), "n_gallery": len(index_names)}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        for cat in cats:
            d = np.load(out_dir / f"hbi_fused_{cat}.npz", allow_pickle=True)
            hbi = {
                "sim": d["sim"],
                "index_names": [str(x) for x in d["index_names"].tolist()],
                "refs": [str(x) for x in d["refs"].tolist()],
                "tgts": [str(x) for x in d["tgts"].tolist()],
            }
            sim, index_names, refs, tgts = align_to_official(hbi, cat)
            ranked = rank_by_similarity(sim, index_names)
            coarse = recall_at_ks(ranked, refs, tgts)
            payload = build_topk_payload(sim, index_names, refs, tgts, cat, k_save)
            payloads[cat] = payload
            per_cat[cat] = {"coarse": coarse, "n_queries": len(refs), "n_gallery": len(index_names)}

    logger.info("Loading FIQ CR Stage-1/2: %s | %s", args.stage1_ckpt, args.stage2_ckpt)
    model_stage1, model_stage2, preprocess, _mods = load_blip_models(
        Path(args.stage1_ckpt), Path(args.stage2_ckpt)
    )
    model_stage1 = model_stage1.to(device).eval()
    model_stage2 = model_stage2.to(device).eval()

    for cat in cats:
        payload = payloads[cat]
        d = np.load(out_dir / f"hbi_fused_{cat}.npz", allow_pickle=True)
        hbi = {
            "sim": d["sim"],
            "index_names": [str(x) for x in d["index_names"].tolist()],
            "refs": [str(x) for x in d["refs"].tolist()],
            "tgts": [str(x) for x in d["tgts"].tolist()],
        }
        sim_a, index_names, refs, tgts = align_to_official(hbi, cat)
        coarse_sorted = rank_by_similarity(sim_a, index_names)
        n_g = coarse_sorted.shape[1]
        ref_exp = np.repeat(np.asarray(refs, dtype=object), n_g).reshape(len(refs), n_g)
        coarse_no_ref = coarse_sorted[coarse_sorted != ref_exp].reshape(len(refs), n_g - 1)

        reranked, _pred, pool_metrics = rerank_category(
            payload, model_stage1, model_stage2, preprocess, cat,
            k_rerank, device, args.workers, args.cr_gallery_batch,
        )
        merged = merge_rerank_with_coarse_gallery(coarse_no_ref, reranked)
        rerank_metrics = recall_at_ks(merged, refs, tgts)
        per_cat[cat]["pool"] = pool_metrics
        per_cat[cat]["rerank"] = rerank_metrics
        logger.info(
            "FIQ %s CR-rerank K=%d: R@10=%.2f R@50=%.2f | pool coverage=%.2f R@10=%.2f R@50=%.2f",
            cat, k_rerank, rerank_metrics["recall_at10"], rerank_metrics["recall_at50"],
            pool_metrics["coverage"], pool_metrics["pool_recall_at10"], pool_metrics["pool_recall_at50"],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def mean_field(field, key):
        return {
            k: float(np.mean([per_cat[c][field][k] for c in cats]))
            for k in per_cat[cats[0]][field]
            if k.startswith("recall") or k.startswith("pool") or k == "coverage"
        }

    summary = {
        "coarse": mean_field("coarse", "recall"),
        "rerank": mean_field("rerank", "recall"),
        "pool": mean_field("pool", "pool"),
    }
    logger.info(
        "FIQ mean coarse R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f",
        summary["coarse"]["recall_at10"], summary["coarse"]["recall_at50"],
        summary["coarse"]["recall_at100"], summary["coarse"]["recall_at200"],
    )
    logger.info(
        "FIQ mean rerank R@10=%.2f R@50=%.2f (pool K=%d, coverage=%.2f)",
        summary["rerank"]["recall_at10"], summary["rerank"]["recall_at50"],
        k_rerank, summary["pool"]["coverage"],
    )
    out = {
        "init_model": args.init_model,
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt,
        "k_save": k_save,
        "k_rerank": k_rerank,
        "formula_coarse": "S = 0.3*S_vision + 0.3*S_modified + 0.4*S_relative - 0.1*S_removed",
        "mean": summary,
        "per_category": per_cat,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
