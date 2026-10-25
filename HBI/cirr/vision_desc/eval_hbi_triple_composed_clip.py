#!/usr/bin/env python3
"""Fuse S_HBI with CLIP composed similarity (cand-ref vs Modified text).

S_comp = cos(normalize(E_img(I_cand) - E_img(I_ref)), E_text(T_mod))
S_final = S_HBI + beta * S_comp

E_img, E_text: raw unfine-tuned CLIP ViT-L/14.
T_mod: llm_info ModifiedImageCaption.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.cirr import cirr_paths
from HBI.cirr.clip_loader import build_clip_from_config
from HBI.cirr.eval_cirr_clip_baseline import build_clip_gallery_dataloader, encode_gallery_clip
from HBI.cirr.vision_desc.encode_ref_gallery_clip_sim import (
    CIRRUniqueRefCLIPDataset,
    encode_ref_images_clip,
    load_triple_query_meta,
)
from HBI.cirr.vision_desc.eval_hbi_triple_clip_gate import load_or_compute_hbi
from HBI.cirr.vision_desc.eval_hbi_triple_vision import _metrics
from HBI.cirr.vision_desc.eval_hbi_triple_vision_removed import log_metrics, metrics_row
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("triple_composed_clip")

DEFAULT_DESCS = str(ROOT / "data" / "cirr" / "vision_desc" / "val_dual_targets.json")
DEFAULT_HBI_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "hbi_triple_removed_best_sim.npz")
DEFAULT_COMPOSED_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "composed_clip_sim.npz")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "metrics_triple_composed_clip.json")

SPECIAL_TOKEN = {
    "CLS_TOKEN": "<|startoftext|>",
    "SEP_TOKEN": "<|endoftext|>",
}


def build_raw_clip(config, device: str):
    clip, meta = build_clip_from_config(config)
    clip.load_state_dict(meta.state_dict, strict=False)
    clip = clip.to(device).eval()
    return clip, meta


def tokenize_texts(tokenizer, texts: List[str], max_words: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ids_list, mask_list = [], []
    for text in texts:
        words = tokenizer.tokenize(text or " ")
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        if len(words) > max_words - 1:
            words = words[: max_words - 1]
        words.append(SPECIAL_TOKEN["SEP_TOKEN"])
        ids = tokenizer.convert_tokens_to_ids(words)
        mask = [1] * len(ids)
        while len(ids) < max_words:
            ids.append(0)
            mask.append(0)
        ids_list.append(ids)
        mask_list.append(mask)
    return torch.LongTensor(ids_list), torch.LongTensor(mask_list)


@torch.no_grad()
def encode_modified_texts_clip(
    clip,
    modified_texts: List[str],
    tokenizer,
    device: str,
    max_words: int = 77,
    batch_size: int = 64,
) -> torch.Tensor:
    clip.eval()
    feats = []
    for start in tqdm(range(0, len(modified_texts), batch_size), desc="encode Modified (CLIP text)"):
        batch = modified_texts[start : start + batch_size]
        ids, mask = tokenize_texts(tokenizer, batch, max_words)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        emb = clip.encode_text(ids, mask=mask)
        emb = F.normalize(emb, dim=-1)
        feats.append(emb.cpu())
    return torch.cat(feats, dim=0)


def compute_composed_sim(
    ref_feats: torch.Tensor,
    ref_idx: List[int],
    gal_feats: torch.Tensor,
    text_feats: torch.Tensor,
    device: str = "cpu",
    chunk_q: int = 32,
) -> np.ndarray:
    """S_comp[q,g] = cos(normalize(E_g[g]-E_r[q]), E_t[q])."""
    gal_n = F.normalize(gal_feats, dim=-1)
    text_n = F.normalize(text_feats, dim=-1)
    ref_n = F.normalize(ref_feats, dim=-1)
    ref_per_q = ref_n[ref_idx]

    n_q = text_n.size(0)
    n_g = gal_n.size(0)
    out = np.zeros((n_q, n_g), dtype=np.float32)

    for qs in range(0, n_q, chunk_q):
        qe = min(qs + chunk_q, n_q)
        diff = gal_n.unsqueeze(0) - ref_per_q[qs:qe].unsqueeze(1)
        diff_norm = diff.norm(dim=-1, keepdim=True)
        diff_n = diff / diff_norm.clamp(min=1e-6)
        # Zero diff (e.g. gallery == ref): cosine undefined -> 0
        scores = (diff_n * text_n[qs:qe].unsqueeze(1)).sum(dim=-1)
        scores = torch.where(diff_norm.squeeze(-1) < 1e-6, torch.zeros_like(scores), scores)
        out[qs:qe] = scores.cpu().numpy()
    return out


def load_modified_texts(dual_json: str, max_samples: int | None) -> Tuple[List[str], List[str], List[str]]:
    refs, tgts, _, queries = load_triple_query_meta(dual_json, max_samples)
    with open(cirr_paths.llm_json("val"), "r", encoding="utf-8") as f:
        llm = json.load(f)
    texts = []
    for q in queries:
        key = cirr_paths.pair_key(q["reference_name"], q["target_name"])
        texts.append(str(llm.get(key, {}).get("ModifiedImageCaption") or "").strip())
    return texts, refs, tgts


def load_relative_texts(dual_json: str, max_samples: int | None) -> Tuple[List[str], List[str], List[str]]:
    """Query-side edit instruction from CIRR val caption (not LLM target description)."""
    refs, tgts, _, queries = load_triple_query_meta(dual_json, max_samples)
    texts = [str(q.get("relative") or "").strip() for q in queries]
    return texts, refs, tgts


def load_composed_texts(
    dual_json: str, max_samples: int | None, text_source: str = "relative",
) -> Tuple[List[str], List[str], List[str]]:
    if text_source == "relative":
        return load_relative_texts(dual_json, max_samples)
    if text_source == "modified":
        return load_modified_texts(dual_json, max_samples)
    raise ValueError(f"unknown text_source={text_source!r}")


@torch.no_grad()
def load_or_compute_composed_sim(args, refs: List[str]) -> np.ndarray:
    cache = Path(args.composed_sim_npz)
    if cache.is_file() and not args.recompute_composed:
        d = np.load(cache, allow_pickle=True)
        logger.info("Loaded cached S_comp from %s", cache)
        return d["sim_composed"].astype(np.float32)

    config = SimpleNamespace(
        base_encoder=args.base_encoder,
        clip_checkpoint=args.clip_checkpoint,
        batch_size_val=args.batch_size,
        workers=args.workers,
        image_resolution=args.image_resolution,
    )
    clip, meta = build_raw_clip(config, args.device)
    logger.info("Loaded raw CLIP from %s", meta.model_path)

    gallery_loader, _ = build_clip_gallery_dataloader(
        split="val",
        batch_size=args.batch_size,
        workers=args.workers,
        image_resolution=args.image_resolution,
    )
    gal_feats, index_names = encode_gallery_clip(clip, gallery_loader, args.device)

    ref_feats_unique, ref_names_unique = encode_ref_images_clip(
        clip, refs, args.device, args.batch_size, args.workers, args.image_resolution,
    )
    ref_to_idx = {n: i for i, n in enumerate(ref_names_unique)}
    ref_idx = [ref_to_idx[r] for r in refs]

    modified_texts, _, _ = load_modified_texts(args.dual_json, args.max_samples)
    tokenizer = ClipTokenizer()
    text_feats = encode_modified_texts_clip(
        clip, modified_texts, tokenizer, args.device, args.max_words, args.batch_size,
    )

    sim_composed = compute_composed_sim(ref_feats_unique, ref_idx, gal_feats, text_feats)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        sim_composed=sim_composed.astype(np.float32),
        index_names=np.array(index_names, dtype=object),
        clip_checkpoint=str(meta.model_path),
    )
    logger.info(
        "Wrote S_comp to %s shape=%s | min=%.4f max=%.4f mean=%.4f",
        cache, sim_composed.shape,
        float(sim_composed.min()), float(sim_composed.max()), float(sim_composed.mean()),
    )
    return sim_composed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dual_json", type=str, default=DEFAULT_DESCS)
    p.add_argument("--hbi_sim_npz", type=str, default=DEFAULT_HBI_NPZ)
    p.add_argument("--composed_sim_npz", type=str, default=DEFAULT_COMPOSED_NPZ)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--recompute_hbi", action="store_true")
    p.add_argument("--recompute_composed", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--beta_step", type=float, default=0.1)
    p.add_argument("--beta_max", type=float, default=3.0)
    p.add_argument("--init_model", type=str, default=str(ROOT / "output" / "cirr_retrieval" / "pytorch_model.bin.best.10"))
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--image_resolution", type=int, default=224)
    # HBI cache args (only if recompute_hbi)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--agg_module", type=str, default="None")
    p.add_argument("--interaction", type=str, default="wti")
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--estimator", type=str, default="None")
    p.add_argument("--max_frames", type=int, default=12)
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    sim_hbi, refs, tgts, groups, index_names = load_or_compute_hbi(args)
    sim_comp = load_or_compute_composed_sim(args, refs)

    if sim_hbi.shape != sim_comp.shape:
        raise ValueError(f"Shape mismatch: S_HBI {sim_hbi.shape} vs S_comp {sim_comp.shape}")

    m_hbi = _metrics(sim_hbi, index_names, refs, tgts, groups)
    m_comp = _metrics(sim_comp, index_names, refs, tgts, groups)
    log_metrics("S_HBI baseline", m_hbi)
    log_metrics("S_comp composed-only", m_comp)

    beta_vals = np.arange(0.0, args.beta_max + 1e-9, args.beta_step)
    grid = []
    best = None

    for beta in beta_vals:
        beta = float(round(beta, 4))
        fused = (sim_hbi.astype(np.float64) + beta * sim_comp.astype(np.float64)).astype(np.float32)
        m = _metrics(fused, index_names, refs, tgts, groups)
        row = {"beta": beta, **metrics_row(m)}
        grid.append(row)
        if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
            abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
            and m["group_recall_at1"] > best["group_recall_at1"]
        ):
            best = row.copy()

    grid.sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))

    logger.info("Top-10 beta grid:")
    for row in grid[:10]:
        log_metrics(f"  beta={row['beta']:.2f}", row)

    out = {
        "n_queries": len(refs),
        "formula": "S_final = S_HBI + beta * cos(normalize(E_cand-E_ref), E_mod)",
        "text_source": "llm_info ModifiedImageCaption",
        "clip": "raw ViT-L/14 (unfine-tuned)",
        "s_hbi_metrics": metrics_row(m_hbi),
        "s_comp_metrics": metrics_row(m_comp),
        "composed_sim_npz": args.composed_sim_npz,
        "hbi_sim_npz": args.hbi_sim_npz,
        "beta_step": args.beta_step,
        "beta_max": args.beta_max,
        "best": best,
        "grid_top20": grid[:20],
        "grid_all_count": len(grid),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log_metrics("best fused", best, f"beta={best['beta']:.2f}")
    logger.info("Wrote %s (%d grid points)", args.output_json, len(grid))


if __name__ == "__main__":
    main()
