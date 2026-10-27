"""DataLoader builders for CIRR bfsar."""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import DataLoader, DistributedSampler

from bfsar.cirr.dataloader import (
    CIRRTrainDataset,
    CIRRQueryDataset,
    CIRRGalleryDataset,
)


def _maybe_distributed_sampler(dataset, shuffle, args):
    if getattr(args, "distributed", 0) and torch.cuda.is_available():
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


def build_cirr_train_dataloader(args, tokenizer):
    dataset = CIRRTrainDataset(
        split="train",
        tokenizer=tokenizer,
        max_words=args.max_words,
        num_crops=args.max_frames,
        image_resolution=args.image_resolution,
        caption_json=getattr(args, "cirr_caption_json", None),
        split_json=getattr(args, "cirr_split_json", None),
        llm_json=getattr(args, "cirr_llm_json", None),
        image_root=getattr(args, "cirr_image_root", None),
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


def build_cirr_val_trainstyle_dataloader(args, tokenizer):
    """In-batch retrieval val (same tensor layout as MSRVTT val)."""
    dataset = CIRRTrainDataset(
        split="val",
        tokenizer=tokenizer,
        max_words=args.max_words,
        num_crops=args.max_frames,
        image_resolution=args.image_resolution,
        caption_json=getattr(args, "cirr_caption_json", None),
        split_json=getattr(args, "cirr_split_json", None),
        llm_json=getattr(args, "cirr_llm_json", None),
        image_root=getattr(args, "cirr_image_root", None),
    )
    # val uses eval transforms
    dataset.transform_train = dataset.transform_eval
    sampler = _maybe_distributed_sampler(dataset, False, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader, len(dataset)


def collate_cirr_query(batch):
    """Keep per-query group_members as list[list[str]] (default collate transposes them)."""
    text_ids = torch.stack([item[0] for item in batch])
    text_mask = torch.stack([item[1] for item in batch])
    reference_names = [item[2] for item in batch]
    target_names = [item[3] for item in batch]
    group_members = [list(item[4]) for item in batch]
    idx = torch.tensor([item[5] for item in batch], dtype=torch.long)
    return text_ids, text_mask, reference_names, target_names, group_members, idx


def build_cirr_query_dataloader(args, tokenizer, split="val"):
    dataset = CIRRQueryDataset(
        split=split,
        tokenizer=tokenizer,
        max_words=args.max_words,
        caption_json=getattr(args, "cirr_caption_json", None),
        llm_json=getattr(args, "cirr_llm_json", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_cirr_query,
    )
    return loader, len(dataset), dataset


def build_cirr_gallery_dataloader(args, split="val"):
    dataset = CIRRGalleryDataset(
        split=split,
        num_crops=args.max_frames,
        image_resolution=args.image_resolution,
        split_json=getattr(args, "cirr_split_json", None),
        image_root=getattr(args, "cirr_image_root", None),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader, len(dataset), dataset
