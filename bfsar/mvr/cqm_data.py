"""Utilities for BLIP-2 twin Q-Former CIRR rerank (index features + dataset wiring)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bfsar.cirr import cirr_paths
from bfsar.mvr.cvs_infer import (
    CR_SRC,
    ASSET_ROOT,
    _ensure_cirr_image_symlinks,
    _import_blip_modules,
)

logger = logging.getLogger("bfsar_cqm")


@torch.no_grad()
def extract_blip2_index_features(
    dataset,
    model,
    batch_size: int = 8,
) -> Tuple[torch.Tensor, List[str]]:
    """Extract gallery ViT tokens for BLIP-2 reranker (dynamic N×768)."""
    from bfsar.mvr.cvs_infer import _import_blip_modules as _imp

    mods = _imp()
    collate_fn = mods["collate_fn"]
    device = next(model.parameters()).device
    num_workers = int(os.environ.get("CIR_NUM_WORKERS", "4"))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    n = len(dataset)
    index_names: List[str] = []
    index_features = None
    offset = 0
    model.eval()
    for names, images in tqdm(loader, desc=f"blip2 index feats {getattr(dataset, 'split', '')}"):
        images = images.to(device, non_blocking=True)
        feats = model.img_embed(images).float()
        if index_features is None:
            index_features = torch.empty(
                (n, feats.shape[1], feats.shape[2]),
                dtype=feats.dtype,
                device=device,
            )
        bs = feats.shape[0]
        index_features[offset : offset + bs] = feats
        offset += bs
        index_names.extend(names)
    return index_features, index_names


def get_cirr_preprocess(target_ratio: float = 1.25):
    """Same targetpad transform as CR BLIP pipeline (384); resized inside model to 224."""
    mods = _import_blip_modules()
    return mods["targetpad_transform"](target_ratio, 384)


def import_cirr_dataset():
    _ensure_cirr_image_symlinks()
    mods = _import_blip_modules()
    return mods["CIRRDataset"], mods["collate_fn"]
