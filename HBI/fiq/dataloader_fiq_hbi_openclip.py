"""Fashion-IQ HBI datasets with OpenCLIP LAION-2B image normalization (mean/std=0.5)."""

from __future__ import annotations

from typing import Optional, Sequence

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from HBI.fiq.dataloader_fiq_hbi import FIQHBIGalleryDataset, FIQHBITrainDataset
from HBI.fiq.openclip_backbone import OPENCLIP_MEAN, OPENCLIP_STD


def _train_transform(image_resolution: int) -> Compose:
    return Compose(
        [
            RandomResizedCrop(image_resolution, scale=(0.5, 1.0)),
            RandomHorizontalFlip(),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(OPENCLIP_MEAN, OPENCLIP_STD),
        ]
    )


def _eval_transform(image_resolution: int) -> Compose:
    return Compose(
        [
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(OPENCLIP_MEAN, OPENCLIP_STD),
        ]
    )


class FIQHBITrainDatasetOpenCLIP(FIQHBITrainDataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        max_words: int = 77,
        num_crops: int = 6,
        image_resolution: int = 224,
        categories: Optional[Sequence[str]] = None,
        image_root: Optional[str] = None,
    ):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            max_words=max_words,
            num_crops=num_crops,
            image_resolution=image_resolution,
            categories=categories,
            image_root=image_root,
        )
        self.transform_train = _train_transform(image_resolution)
        self.transform_eval = _eval_transform(image_resolution)


class FIQHBIGalleryDatasetOpenCLIP(FIQHBIGalleryDataset):
    def __init__(
        self,
        split: str,
        category: str,
        num_crops: int = 6,
        image_resolution: int = 224,
        image_root: Optional[str] = None,
    ):
        super().__init__(
            split=split,
            category=category,
            num_crops=num_crops,
            image_resolution=image_resolution,
            image_root=image_root,
        )
        self.transform = _eval_transform(image_resolution)
