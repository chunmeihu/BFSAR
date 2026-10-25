"""DataLoader builders for CIRR HBI + OpenCLIP image normalization."""

from __future__ import annotations

from torch.utils.data import DataLoader

from HBI.cirr.cirr_dataloader_builders import (
    _maybe_distributed_sampler,
    build_cirr_query_dataloader,
    collate_cirr_query,
)
from HBI.cirr.dataloader_cirr_hbi_openclip import (
    CIRRHBIGalleryDatasetOpenCLIP,
    CIRRHBITrainDatasetOpenCLIP,
)


def build_cirr_train_dataloader(args, tokenizer):
    dataset = CIRRHBITrainDatasetOpenCLIP(
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
    dataset = CIRRHBITrainDatasetOpenCLIP(
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


def build_cirr_gallery_dataloader(args, split="val"):
    dataset = CIRRHBIGalleryDatasetOpenCLIP(
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


__all__ = [
    "build_cirr_train_dataloader",
    "build_cirr_val_trainstyle_dataloader",
    "build_cirr_gallery_dataloader",
    "build_cirr_query_dataloader",
    "collate_cirr_query",
]
