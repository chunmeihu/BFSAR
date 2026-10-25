#!/usr/bin/env python3
"""Encode ref→gallery image similarity with raw (unfine-tuned) CLIP ViT-L/14.

Single center-crop, L2-normalized image embeddings, cosine similarity.
Query order matches CIRRTripleVisionDataset (same val triples + dual_json filter).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

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
from HBI.cirr.clip_loader import build_clip_from_config
from HBI.cirr.eval_cirr_clip_baseline import (
    build_clip_gallery_dataloader,
    encode_gallery_clip,
    _similarity_matrix,
)
from HBI.cirr.metrics_cirr_hbi import compute_cirr_metrics, rank_by_similarity
from HBI.cirr.vision_desc.eval_hbi_triple_vision import CIRRTripleVisionDataset
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("encode_ref_gallery_clip")

DEFAULT_DESCS = str(ROOT / "data" / "cirr" / "vision_desc" / "val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output" / "cirr_vision_desc" / "ref_gallery_clip_sim.npz")


def clip_transform(image_resolution: int = 224):
    return Compose([
        Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class CIRRUniqueRefCLIPDataset(Dataset):
    """Unique reference images for triple-vision query set."""

    def __init__(self, ref_names: List[str], image_resolution: int = 224):
        with open(cirr_paths.split_json("val"), "r") as f:
            self.name_to_relpath = json.load(f)
        self.image_root = cirr_paths.CIRR_IMAGE_ROOT
        self.ref_names = sorted(set(ref_names))
        self.transform = clip_transform(image_resolution)

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, index):
        name = self.ref_names[index]
        path = cirr_paths.image_path(self.name_to_relpath, name, self.image_root)
        image = Image.open(path).convert("RGB")
        return self.transform(image), name


@torch.no_grad()
def encode_ref_images_clip(
    clip,
    ref_names: List[str],
    device: str,
    batch_size: int,
    workers: int,
    image_resolution: int = 224,
) -> Tuple[torch.Tensor, List[str]]:
    dataset = CIRRUniqueRefCLIPDataset(ref_names, image_resolution=image_resolution)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    clip.eval()
    feats, names = [], []
    for images, batch_names in tqdm(loader, desc="encode ref images (CLIP)"):
        images = images.to(device, non_blocking=True)
        emb = clip.encode_image(images)
        emb = F.normalize(emb, dim=-1)
        feats.append(emb.cpu())
        names.extend(list(batch_names))
    return torch.cat(feats, dim=0), names


def load_triple_query_meta(dual_json: str, max_samples: int | None) -> Tuple[List[str], List[str], List[str], List]:
    tokenizer = ClipTokenizer()
    ds = CIRRTripleVisionDataset(dual_json, tokenizer, max_samples=max_samples)
    refs = [q["reference_name"] for q in ds.queries]
    tgts = [q["target_name"] for q in ds.queries]
    groups = [q["group_members"] for q in ds.queries]
    return refs, tgts, groups, ds.queries


def build_raw_clip(config, device: str):
    clip, meta = build_clip_from_config(config)
    clip.load_state_dict(meta.state_dict, strict=False)
    clip = clip.to(device).eval()
    return clip, meta


def parse_args():
    p = argparse.ArgumentParser(description="Encode ref→gallery CLIP ViT-L/14 cosine similarity")
    p.add_argument("--dual_json", type=str, default=DEFAULT_DESCS)
    p.add_argument("--output_npz", type=str, default=DEFAULT_OUT)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--query_batch", type=int, default=64)
    p.add_argument("--gallery_batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image_resolution", type=int, default=224)
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--skip_metrics", action="store_true", help="Skip img-img only R@K sanity metrics")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = SimpleNamespace(
        base_encoder=args.base_encoder,
        clip_checkpoint=args.clip_checkpoint,
        batch_size_val=args.batch_size,
        workers=args.workers,
        image_resolution=args.image_resolution,
    )

    refs, tgts, groups, _ = load_triple_query_meta(args.dual_json, args.max_samples)
    logger.info("Queries aligned with triple vision: n=%d", len(refs))

    clip, meta = build_raw_clip(config, args.device)
    logger.info("Loaded raw CLIP from %s (no HBI finetune)", meta.model_path)

    gallery_loader, _ = build_clip_gallery_dataloader(
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        image_resolution=args.image_resolution,
    )
    gallery_feats, index_names = encode_gallery_clip(clip, gallery_loader, args.device)
    logger.info("Gallery encoded: n=%d dim=%d", gallery_feats.size(0), gallery_feats.size(1))

    ref_feats_unique, ref_names_unique = encode_ref_images_clip(
        clip,
        refs,
        args.device,
        args.batch_size,
        args.workers,
        args.image_resolution,
    )
    ref_to_idx: Dict[str, int] = {n: i for i, n in enumerate(ref_names_unique)}
    ref_idx = [ref_to_idx[r] for r in refs]
    ref_feats = ref_feats_unique[ref_idx]
    logger.info("Reference images: unique=%d queries=%d", len(ref_names_unique), len(refs))

    sim_img = _similarity_matrix(
        ref_feats,
        gallery_feats,
        query_batch=args.query_batch,
        gallery_batch=args.gallery_batch,
    )
    logger.info(
        "S_img stats: min=%.4f max=%.4f mean=%.4f",
        float(sim_img.min()), float(sim_img.max()), float(sim_img.mean()),
    )

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        sim_img=sim_img.astype(np.float32),
        refs=np.array(refs, dtype=object),
        tgts=np.array(tgts, dtype=object),
        index_names=np.array(index_names, dtype=object),
        clip_checkpoint=str(meta.model_path),
        encoder=args.base_encoder,
        finetuned=False,
    )
    logger.info("Wrote %s shape=%s", out_path, sim_img.shape)

    if not args.skip_metrics:
        ranked = rank_by_similarity(sim_img, index_names)
        m = compute_cirr_metrics(ranked, refs, tgts, groups)
        logger.info(
            "ref->gallery CLIP only | R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f | "
            "G-R@1=%.2f G-R@2=%.2f G-R@3=%.2f",
            m["recall_at1"], m["recall_at5"], m["recall_at10"], m["recall_at50"],
            m["group_recall_at1"], m["group_recall_at2"], m["group_recall_at3"],
        )


if __name__ == "__main__":
    main()
