#!/usr/bin/env python3
"""Composed visual coherence with query-side relative caption + calibrated fusion.

S_comp = cos(normalize(E_cand - E_ref), E_relative)
Fusion strategies (per-query z-score on both branches unless noted):
  - zscore_add:  z_hbi + beta * z_comp
  - blend:       alpha * z_hbi + (1-alpha) * z_comp
  - rerank:      S_hbi + beta * z_comp  (boost only within top-K by S_hbi)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.cirr import cirr_paths
from HBI.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader
from HBI.cirr.clip_loader import build_clip_from_config
from HBI.cirr.eval_cirr_clip_baseline import build_clip_gallery_dataloader, encode_gallery_clip
from HBI.cirr.image_multi_crop import multi_crop_pil
from HBI.cirr.modeling_hbi_cirr import HBICIRR
from HBI.cirr.vision_desc.encode_ref_gallery_clip_sim import encode_ref_images_clip, load_triple_query_meta
from HBI.cirr.vision_desc.eval_hbi_triple_clip_gate import load_or_compute_hbi
from HBI.cirr.vision_desc.eval_hbi_triple_composed_clip import (
    compute_composed_sim,
    encode_modified_texts_clip,
    load_composed_texts,
    tokenize_texts,
)
from HBI.cirr.vision_desc.eval_hbi_triple_vision import _metrics
from HBI.cirr.vision_desc.eval_hbi_triple_vision_removed import log_metrics, metrics_row
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("composed_relative_fuse")

DEFAULT_DESCS = str(ROOT / "data" / "cirr" / "vision_desc" / "val_dual_targets.json")
DEFAULT_HBI_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "hbi_triple_removed_best_sim.npz")
DEFAULT_CLIP_COMP_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "composed_clip_relative_sim.npz")
DEFAULT_HBI_COMP_NPZ = str(ROOT / "output" / "cirr_vision_desc" / "composed_hbi_relative_sim.npz")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "metrics_triple_composed_relative_fuse.json")


def per_query_zscore(sim: np.ndarray) -> np.ndarray:
    mu = sim.mean(axis=1, keepdims=True)
    std = sim.std(axis=1, keepdims=True)
    return ((sim - mu) / (std + 1e-6)).astype(np.float32)


def fuse_zscore_add(sim_hbi: np.ndarray, sim_comp: np.ndarray, beta: float) -> np.ndarray:
    zh = per_query_zscore(sim_hbi)
    zc = per_query_zscore(sim_comp)
    return (zh + beta * zc).astype(np.float32)


def fuse_blend(sim_hbi: np.ndarray, sim_comp: np.ndarray, alpha: float) -> np.ndarray:
    zh = per_query_zscore(sim_hbi)
    zc = per_query_zscore(sim_comp)
    return (alpha * zh + (1.0 - alpha) * zc).astype(np.float32)


def fuse_rerank(sim_hbi: np.ndarray, sim_comp: np.ndarray, topk: int, beta: float) -> np.ndarray:
    n_q, _ = sim_hbi.shape
    out = sim_hbi.astype(np.float64).copy()
    zc = per_query_zscore(sim_comp)
    k = min(topk, sim_hbi.shape[1])
    for q in range(n_q):
        idx = np.argsort(-sim_hbi[q])[:k]
        out[q, idx] += beta * zc[q, idx]
    return out.astype(np.float32)


class CIRRUniqueRefHBIDataset(Dataset):
    def __init__(self, ref_names: List[str], num_crops: int = 12, image_resolution: int = 224):
        with open(cirr_paths.split_json("val"), "r") as f:
            self.name_to_relpath = json.load(f)
        self.image_root = cirr_paths.CIRR_IMAGE_ROOT
        self.ref_names = sorted(set(ref_names))
        self.num_crops = num_crops
        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, index):
        name = self.ref_names[index]
        path = cirr_paths.image_path(self.name_to_relpath, name, self.image_root)
        image = Image.open(path).convert("RGB")
        w, h = image.size
        if min(w, h) < 2:
            scale = max(2 / w, 2 / h)
            image = image.resize((max(2, int(w * scale)), max(2, int(h * scale)),), Image.BICUBIC)
        crops = multi_crop_pil(image, self.num_crops)
        tensors = torch.stack([self.transform(c) for c in crops], dim=0)
        crop_mask = torch.ones(self.num_crops, dtype=torch.long)
        return tensors, crop_mask, name


@torch.no_grad()
def encode_gallery_hbi_pooled(model, gallery_loader, device) -> Tuple[torch.Tensor, List[str]]:
    model.eval()
    feats, names = [], []
    for crops, crop_mask, name, _ in tqdm(gallery_loader, desc="encode gallery (HBI pooled)"):
        crops = crops.to(device, non_blocking=True)
        b, k = crops.size(0), crops.size(1)
        flat = crops.view(b * k, *crops.shape[2:])
        cls, _ = model.clip.encode_image(flat, return_hidden=True)
        cls = cls.float().view(b, k, -1).mean(dim=1)
        cls = F.normalize(cls, dim=-1)
        feats.append(cls.cpu())
        names.extend(list(name))
    return torch.cat(feats, dim=0), names


@torch.no_grad()
def encode_refs_hbi_pooled(
    model, ref_names: List[str], device, batch_size: int, workers: int, num_crops: int, image_resolution: int,
) -> Tuple[torch.Tensor, List[str]]:
    ds = CIRRUniqueRefHBIDataset(ref_names, num_crops=num_crops, image_resolution=image_resolution)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    feats, names = [], []
    for crops, crop_mask, batch_names in tqdm(loader, desc="encode ref (HBI pooled)"):
        crops = crops.to(device, non_blocking=True)
        b, k = crops.size(0), crops.size(1)
        flat = crops.view(b * k, *crops.shape[2:])
        cls, _ = model.clip.encode_image(flat, return_hidden=True)
        cls = cls.float().view(b, k, -1).mean(dim=1)
        cls = F.normalize(cls, dim=-1)
        feats.append(cls.cpu())
        names.extend(list(batch_names))
    return torch.cat(feats, dim=0), names


@torch.no_grad()
def encode_text_hbi_cls(
    model, texts: List[str], tokenizer, device, max_words: int = 77, batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode relative (HBI cls)"):
        batch = texts[start : start + batch_size]
        ids, mask = tokenize_texts(tokenizer, batch, max_words)
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        _, cls = model.get_text_feat(ids, mask, shaped=True)
        cls = F.normalize(cls, dim=-1)
        feats.append(cls.cpu())
    return torch.cat(feats, dim=0)


def load_hbi_model(args, device):
    model = HBICIRR(args)
    state = torch.load(args.init_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval()


def _cache_ok(path: Path, text_source: str, encoder: str, recompute: bool) -> bool:
    if recompute or not path.is_file():
        return False
    d = np.load(path, allow_pickle=True)
    return str(d.get("text_source", "")) == text_source and str(d.get("encoder", "")) == encoder


@torch.no_grad()
def compute_clip_composed_relative(args, refs: List[str]) -> np.ndarray:
    cache = Path(args.clip_comp_npz)
    if _cache_ok(cache, "relative", "clip", args.recompute_clip):
        d = np.load(cache, allow_pickle=True)
        logger.info("Loaded CLIP S_comp (relative) from %s", cache)
        return d["sim_composed"].astype(np.float32)

    config = SimpleNamespace(
        base_encoder=args.base_encoder,
        clip_checkpoint=args.clip_checkpoint,
        batch_size_val=args.batch_size,
        workers=args.workers,
        image_resolution=args.image_resolution,
    )
    clip, meta = build_clip_from_config(config)
    clip.load_state_dict(meta.state_dict, strict=False)
    clip = clip.to(args.device).eval()

    gallery_loader, _ = build_clip_gallery_dataloader(
        "val", args.batch_size, args.workers, args.image_resolution,
    )
    gal_feats, _ = encode_gallery_clip(clip, gallery_loader, args.device)
    ref_feats, ref_names = encode_ref_images_clip(
        clip, refs, args.device, args.batch_size, args.workers, args.image_resolution,
    )
    ref_idx = [ref_names.index(r) for r in refs]
    texts, _, _ = load_composed_texts(args.dual_json, args.max_samples, "relative")
    text_feats = encode_modified_texts_clip(
        clip, texts, ClipTokenizer(), args.device, args.max_words, args.batch_size,
    )
    sim = compute_composed_sim(ref_feats, ref_idx, gal_feats, text_feats)
    np.savez_compressed(cache, sim_composed=sim.astype(np.float32), text_source="relative", encoder="clip")
    logger.info("Wrote CLIP S_comp (relative) %s min=%.4f max=%.4f mean=%.4f",
                sim.shape, float(np.nanmin(sim)), float(np.nanmax(sim)), float(np.nanmean(sim)))
    return sim


@torch.no_grad()
def compute_hbi_composed_relative(args, refs: List[str]) -> np.ndarray:
    cache = Path(args.hbi_comp_npz)
    if _cache_ok(cache, "relative", "hbi_pooled", args.recompute_hbi_comp):
        d = np.load(cache, allow_pickle=True)
        logger.info("Loaded HBI S_comp (relative) from %s", cache)
        return d["sim_composed"].astype(np.float32)

    model = load_hbi_model(args, args.device)
    gallery_loader, _, _ = build_cirr_gallery_dataloader(args, split="val")
    gal_feats, _ = encode_gallery_hbi_pooled(model, gallery_loader, args.device)
    ref_feats, ref_names = encode_refs_hbi_pooled(
        model, refs, args.device, args.batch_size_val, args.workers,
        args.max_frames, args.image_resolution,
    )
    ref_idx = [ref_names.index(r) for r in refs]
    texts, _, _ = load_composed_texts(args.dual_json, args.max_samples, "relative")
    text_feats = encode_text_hbi_cls(
        model, texts, ClipTokenizer(), args.device, args.max_words, args.batch_size_val,
    )
    sim = compute_composed_sim(ref_feats, ref_idx, gal_feats, text_feats)
    np.savez_compressed(cache, sim_composed=sim.astype(np.float32), text_source="relative", encoder="hbi_pooled")
    logger.info("Wrote HBI S_comp (relative) %s min=%.4f max=%.4f mean=%.4f",
                sim.shape, float(np.nanmin(sim)), float(np.nanmax(sim)), float(np.nanmean(sim)))
    return sim


def grid_search(
    sim_hbi: np.ndarray,
    sim_comp: np.ndarray,
    mode: str,
    index_names, refs, tgts, groups,
    beta_vals: np.ndarray,
    alpha_vals: np.ndarray,
    rerank_ks: List[int],
) -> Tuple[dict, List[dict]]:
    comp_m = _metrics(sim_comp, index_names, refs, tgts, groups)
    grid: List[dict] = []
    best: Optional[dict] = None

    if mode == "zscore_add":
        for beta in beta_vals:
            beta = float(round(float(beta), 4))
            fused = fuse_zscore_add(sim_hbi, sim_comp, beta)
            m = _metrics(fused, index_names, refs, tgts, groups)
            row = {"mode": mode, "beta": beta, **metrics_row(m)}
            grid.append(row)
            if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
                abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
                and m["group_recall_at1"] > best["group_recall_at1"]
            ):
                best = row.copy()
    elif mode == "blend":
        for alpha in alpha_vals:
            alpha = float(round(float(alpha), 4))
            fused = fuse_blend(sim_hbi, sim_comp, alpha)
            m = _metrics(fused, index_names, refs, tgts, groups)
            row = {"mode": mode, "alpha": alpha, **metrics_row(m)}
            grid.append(row)
            if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
                abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
                and m["group_recall_at1"] > best["group_recall_at1"]
            ):
                best = row.copy()
    elif mode == "rerank":
        for k in rerank_ks:
            for beta in beta_vals:
                beta = float(round(float(beta), 4))
                fused = fuse_rerank(sim_hbi, sim_comp, k, beta)
                m = _metrics(fused, index_names, refs, tgts, groups)
                row = {"mode": mode, "topk": int(k), "beta": beta, **metrics_row(m)}
                grid.append(row)
                if best is None or m["recall_at1"] > best["recall_at1"] + 1e-9 or (
                    abs(m["recall_at1"] - best["recall_at1"]) < 1e-9
                    and m["group_recall_at1"] > best["group_recall_at1"]
                ):
                    best = row.copy()
    else:
        raise ValueError(mode)

    grid.sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))
    assert best is not None
    return {"s_comp_only": metrics_row(comp_m), "best": best, "grid_top10": grid[:10]}, grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dual_json", type=str, default=DEFAULT_DESCS)
    p.add_argument("--hbi_sim_npz", type=str, default=DEFAULT_HBI_NPZ)
    p.add_argument("--clip_comp_npz", type=str, default=DEFAULT_CLIP_COMP_NPZ)
    p.add_argument("--hbi_comp_npz", type=str, default=DEFAULT_HBI_COMP_NPZ)
    p.add_argument("--output_json", type=str, default=DEFAULT_OUT)
    p.add_argument("--recompute_hbi", action="store_true")
    p.add_argument("--recompute_clip", action="store_true")
    p.add_argument("--recompute_hbi_comp", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--beta_step", type=float, default=0.1)
    p.add_argument("--beta_max", type=float, default=1.0)
    p.add_argument("--alpha_min", type=float, default=0.5)
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--rerank_ks", type=int, nargs="+", default=[50, 100, 200])
    p.add_argument("--init_model", type=str, default=str(ROOT / "output" / "cirr_retrieval" / "pytorch_model.bin.best.10"))
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--image_resolution", type=int, default=224)
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
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sim_hbi, refs, tgts, groups, index_names = load_or_compute_hbi(args)
    sim_clip = compute_clip_composed_relative(args, refs)
    sim_hbi_comp = compute_hbi_composed_relative(args, refs)

    m_hbi = _metrics(sim_hbi, index_names, refs, tgts, groups)
    log_metrics("S_HBI baseline", m_hbi)

    beta_vals = np.arange(0.0, args.beta_max + 1e-9, args.beta_step)
    alpha_vals = np.arange(args.alpha_min, 1.0 + 1e-9, args.alpha_step)
    modes = ("zscore_add", "blend", "rerank")

    results = {
        "text_source": "relative (CIRR val caption)",
        "s_comp_formula": "cos(normalize(E_cand-E_ref), E_relative)",
        "s_hbi_baseline": metrics_row(m_hbi),
        "beta_step": args.beta_step,
        "beta_max": args.beta_max,
        "rerank_ks": args.rerank_ks,
        "encoders": {},
        "comparison_table": [],
    }

    for enc_name, sim_comp in (("clip", sim_clip), ("hbi_pooled", sim_hbi_comp)):
        logger.info("======== encoder=%s ========", enc_name)
        enc_out = {"encoder": enc_name, "fusion_modes": {}}
        for mode in modes:
            logger.info("--- %s / %s ---", enc_name, mode)
            summary, _ = grid_search(
                sim_hbi, sim_comp, mode, index_names, refs, tgts, groups,
                beta_vals, alpha_vals, args.rerank_ks,
            )
            enc_out["fusion_modes"][mode] = summary
            b = summary["best"]
            c = summary["s_comp_only"]
            log_metrics(f"{enc_name} S_comp only", c)
            param = b.get("beta", b.get("alpha"))
            param_name = "beta" if "beta" in b else "alpha"
            extra = f" topk={b['topk']}" if "topk" in b else ""
            log_metrics(f"{enc_name} {mode} best {param_name}={param}{extra}", b)
            enc_out["fusion_modes"][mode] = summary
            results["comparison_table"].append({
                "encoder": enc_name,
                "fusion_mode": mode,
                "s_comp_r1": c["recall_at1"],
                **{k: v for k, v in b.items() if k != "mode"},
            })
        results["encoders"][enc_name] = enc_out

    results["comparison_table"].sort(key=lambda x: (-x["recall_at1"], -x["group_recall_at1"]))
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
