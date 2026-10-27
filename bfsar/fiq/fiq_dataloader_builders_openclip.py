"""DataLoader builders for Fashion-IQ BFSAR + OpenCLIP image normalization."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from bfsar.fiq.dataloader_openclip import (
    FIQGalleryDatasetOpenCLIP,
    FIQTrainDatasetOpenCLIP,
)
from bfsar.fiq.fiq_dataloader_builders import (
    _categories_from_args,
    _maybe_distributed_sampler,
    build_fiq_query_dataloader,
    collate_fiq_query,
)


def build_fiq_train_dataloader(args, tokenizer):
    dataset = FIQTrainDatasetOpenCLIP(
        split="train",
        tokenizer=tokenizer,
        max_words=args.max_words,
        num_crops=args.max_frames,
        image_resolution=args.image_resolution,
        categories=_categories_from_args(args),
        image_root=getattr(args, "fiq_image_root", None),
    )
    sampler = _maybe_distributed_sampler(dataset, True, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, len(dataset), sampler


def build_fiq_gallery_dataloader(args, split="val", category="dress"):
    dataset = FIQGalleryDatasetOpenCLIP(
        split=split,
        category=category,
        num_crops=args.max_frames,
        image_resolution=args.image_resolution,
        image_root=getattr(args, "fiq_image_root", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader, len(dataset), dataset


__all__ = [
    "build_fiq_train_dataloader",
    "build_fiq_gallery_dataloader",
    "build_fiq_query_dataloader",
    "collate_fiq_query",
]
