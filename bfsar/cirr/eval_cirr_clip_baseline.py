"""CIRR val baseline: raw CLIP + LLM ModifiedImageCaption -> gallery similarity."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
from tqdm import tqdm

from bfsar.cirr import cirr_paths
from bfsar.cirr.clip_loader import build_clip_from_config
from bfsar.cirr.cirr_dataloader_builders import build_cirr_query_dataloader, collate_cirr_query
from bfsar.cirr.metrics import compute_cirr_metrics, rank_by_similarity
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger(__name__)


class CIRRCLIPGalleryDataset(Dataset):
    """Gallery images with standard single-crop CLIP preprocessing."""

    def __init__(self, split: str, image_resolution: int = 224):
        split_json = cirr_paths.split_json(split)
        import json

        with open(split_json, "r") as f:
            self.name_to_relpath = json.load(f)
        self.image_names = sorted(self.name_to_relpath.keys())
        self.image_root = cirr_paths.CIRR_IMAGE_ROOT
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
        print(f"[CIRRCLIPGalleryDataset] split={split} gallery={len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        path = cirr_paths.image_path(self.name_to_relpath, name, self.image_root)
        image = Image.open(path).convert("RGB")
        return self.transform(image), name, index


def build_clip_gallery_dataloader(split: str, batch_size: int, workers: int, image_resolution: int = 224):
    dataset = CIRRCLIPGalleryDataset(split=split, image_resolution=image_resolution)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return loader, len(dataset)


@torch.no_grad()
def encode_gallery_clip(clip, gallery_loader, device) -> Tuple[torch.Tensor, List[str]]:
    clip.eval()
    feats, names = [], []
    for images, batch_names, _ in tqdm(gallery_loader, desc="encode gallery (CLIP)"):
        images = images.to(device, non_blocking=True)
        emb = clip.encode_image(images)
        emb = F.normalize(emb, dim=-1)
        feats.append(emb.cpu())
        names.extend(list(batch_names))
    return torch.cat(feats, dim=0), names


@torch.no_grad()
def encode_queries_clip(clip, query_loader, device) -> Tuple[torch.Tensor, list, list, list]:
    clip.eval()
    feats = []
    reference_names, target_names, group_members = [], [], []

    for text_ids, text_mask, ref_names, tgt_names, groups, _ in tqdm(
        query_loader, desc="encode queries (CLIP)"
    ):
        text_ids = text_ids.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        emb = clip.encode_text(text_ids, mask=text_mask)
        emb = F.normalize(emb, dim=-1)
        feats.append(emb.cpu())
        reference_names.extend(list(ref_names))
        target_names.extend(list(tgt_names))
        for members in groups:
            group_members.append(list(members))
    return torch.cat(feats, dim=0), reference_names, target_names, group_members


def _similarity_matrix(
    text_feats: torch.Tensor,
    gallery_feats: torch.Tensor,
    query_batch: int = 64,
    gallery_batch: int = 256,
) -> np.ndarray:
    n_q = text_feats.size(0)
    n_g = gallery_feats.size(0)
    sim = np.zeros((n_q, n_g), dtype=np.float32)
    for qs in range(0, n_q, query_batch):
        qe = min(qs + query_batch, n_q)
        q = text_feats[qs:qe]
        row = []
        for gs in range(0, n_g, gallery_batch):
            ge = min(gs + gallery_batch, n_g)
            g = gallery_feats[gs:ge]
            row.append((q @ g.t()).numpy())
        sim[qs:qe] = np.concatenate(row, axis=1)
    return sim


@torch.no_grad()
def evaluate_cirr_clip_baseline(
    config,
    split: str = "val",
    query_batch: int = 64,
    gallery_batch: int = 256,
) -> Dict[str, float]:
    device = getattr(config, "device", "cuda:0")
    tokenizer = ClipTokenizer()
    clip, _ = build_clip_from_config(config)
    clip.load_state_dict(_.state_dict, strict=False)
    clip = clip.to(device).eval()

    query_loader, _, _ = build_cirr_query_dataloader(config, tokenizer, split=split)
    gallery_loader, _ = build_clip_gallery_dataloader(
        split=split,
        batch_size=config.batch_size_val,
        workers=config.workers,
        image_resolution=config.image_resolution,
    )

    gallery_feats, index_names = encode_gallery_clip(clip, gallery_loader, device)
    text_feats, ref_names, tgt_names, group_members = encode_queries_clip(
        clip, query_loader, device
    )

    sim = _similarity_matrix(text_feats, gallery_feats, query_batch, gallery_batch)
    sorted_names = rank_by_similarity(sim, index_names)
    metrics = compute_cirr_metrics(sorted_names, ref_names, tgt_names, group_members)
    logger.info(
        "CLIP baseline (%s): R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f | "
        "G-R@1=%.2f G-R@2=%.2f G-R@3=%.2f",
        split,
        metrics["recall_at1"], metrics["recall_at5"], metrics["recall_at10"], metrics["recall_at50"],
        metrics["group_recall_at1"], metrics["group_recall_at2"], metrics["group_recall_at3"],
    )
    return metrics
