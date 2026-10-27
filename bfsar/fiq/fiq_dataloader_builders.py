"""DataLoader builders for Fashion-IQ bfsar."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, DistributedSampler

from bfsar.fiq.dataloader import (
    FIQTrainDataset,
    FIQQueryDataset,
    FIQGalleryDataset,
)


def _maybe_distributed_sampler(dataset, shuffle, args):
    if getattr(args, "distributed", 0) and torch.cuda.is_available():
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def _categories_from_args(args):
    raw = getattr(args, "fiq_categories", None)
    if not raw:
        return None
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return list(raw)


def build_fiq_train_dataloader(args, tokenizer):
    dataset = FIQTrainDataset(
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


def collate_fiq_query(batch):
    """Keep reference/target/category as lists (default collate would stack strings)."""
    text_ids = torch.stack([item[0] for item in batch])
    text_mask = torch.stack([item[1] for item in batch])
    reference_names = [item[2] for item in batch]
    target_names = [item[3] for item in batch]
    categories = [item[4] for item in batch]
    idx = torch.tensor([item[5] for item in batch], dtype=torch.long)
    return text_ids, text_mask, reference_names, target_names, categories, idx


def build_fiq_query_dataloader(args, tokenizer, split="val", category=None):
    if category is None:
        cats = _categories_from_args(args)
    else:
        cats = [category]
    dataset = FIQQueryDataset(
        split=split,
        tokenizer=tokenizer,
        max_words=args.max_words,
        categories=cats,
        image_root=getattr(args, "fiq_image_root", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fiq_query,
    )
    return loader, len(dataset), dataset


def build_fiq_gallery_dataloader(args, split="val", category="dress"):
    dataset = FIQGalleryDataset(
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
