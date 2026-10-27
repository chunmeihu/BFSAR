"""CIRR BFSAR datasets with OpenCLIP LAION-2B image normalization (mean/std=0.5)."""

from __future__ import annotations

from typing import Optional

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

from bfsar.cirr.dataloader import CIRRGalleryDataset, CIRRTrainDataset
from bfsar.fiq.openclip_backbone import OPENCLIP_MEAN, OPENCLIP_STD


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


class CIRRTrainDatasetOpenCLIP(CIRRTrainDataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        max_words: int = 32,
        num_crops: int = 12,
        image_resolution: int = 224,
        caption_json: Optional[str] = None,
        split_json: Optional[str] = None,
        llm_json: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        super().__init__(
            split=split,
            tokenizer=tokenizer,
            max_words=max_words,
            num_crops=num_crops,
            image_resolution=image_resolution,
            caption_json=caption_json,
            split_json=split_json,
            llm_json=llm_json,
            image_root=image_root,
        )
        self.transform_train = _train_transform(image_resolution)
        self.transform_eval = _eval_transform(image_resolution)


class CIRRGalleryDatasetOpenCLIP(CIRRGalleryDataset):
    def __init__(
        self,
        split: str,
        num_crops: int = 12,
        image_resolution: int = 224,
        split_json: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        super().__init__(
            split=split,
            num_crops=num_crops,
            image_resolution=image_resolution,
            split_json=split_json,
            image_root=image_root,
        )
        self.transform = _eval_transform(image_resolution)
