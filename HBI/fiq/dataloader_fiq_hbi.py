"""Fashion-IQ dataset for HBI (ModifiedImageCaption -> target multi-crop)."""

from __future__ import annotations

import json
from typing import Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
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

from HBI.cirr.image_multi_crop import multi_crop_pil
from HBI.fiq import fiq_paths


class FIQHBITrainDataset(Dataset):
    """Train pairs: MCoT-MVS ModifiedImageCaption + target image multi-crop CLS inputs."""

    SPECIAL_TOKEN = {
        "CLS_TOKEN": "<|startoftext|>",
        "SEP_TOKEN": "<|endoftext|>",
        "MASK_TOKEN": "[MASK]",
        "UNK_TOKEN": "[UNK]",
        "PAD_TOKEN": "[PAD]",
    }

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
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.num_crops = num_crops
        self.image_resolution = image_resolution
        self.image_root = image_root
        self.categories = list(categories or fiq_paths.CATEGORIES)

        self.samples = []
        n_empty_mod = 0
        n_missing_img = 0
        for cat in self.categories:
            llm_path = fiq_paths.llm_json(cat, split)
            with open(llm_path, "r") as f:
                rows = json.load(f)
            for row_i, row in enumerate(rows):
                candidate = str(row["candidate"])
                target = str(row["target"])
                caption = row.get("ModifiedImageCaption", "")
                if not str(caption).strip():
                    n_empty_mod += 1
                img_path = fiq_paths.image_path(target, self.image_root)
                if img_path is None:
                    n_missing_img += 1
                    continue
                self.samples.append(
                    {
                        "idx": len(self.samples),
                        "category": cat,
                        "caption": str(caption),
                        "candidate": candidate,
                        "target_name": target,
                        "image_path": img_path,
                        "src_index": row_i,
                    }
                )

        self.n_empty_mod = n_empty_mod
        self.n_missing_img = n_missing_img
        print(
            f"[FIQHBITrainDataset] split={split} cats={self.categories} "
            f"pairs={len(self.samples)} empty_mod={n_empty_mod} missing_img={n_missing_img}"
        )
        if not self.samples:
            roots = list(fiq_paths.iter_image_roots(self.image_root))
            raise FileNotFoundError(
                "No Fashion-IQ target images found for estimator training. "
                f"Looked in: {roots}. Put images under data/fashioniq/images "
                "or set --fiq_image_root / FIQ_IMAGE_ROOT."
            )

        self.transform_train = Compose([
            RandomResizedCrop(image_resolution, scale=(0.5, 1.0)),
            RandomHorizontalFlip(),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.transform_eval = Compose([
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
        return len(self.samples)

    def _tokenize(self, text: str):
        words = self.tokenizer.tokenize(text)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        max_len = self.max_words - 1
        if len(words) > max_len:
            words = words[:max_len]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        return torch.LongTensor(input_ids), torch.LongTensor(input_mask)

    def _load_crops(self, image_path: str, train: bool):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        if w < 1 or h < 1:
            raise ValueError(f"Invalid image size {w}x{h}: {image_path}")
        if min(w, h) < 2:
            scale = max(2 / w, 2 / h)
            image = image.resize(
                (max(2, int(w * scale)), max(2, int(h * scale))),
                Image.BICUBIC,
            )
        crops = multi_crop_pil(image, self.num_crops)
        transform = self.transform_train if train else self.transform_eval
        tensors = torch.stack([transform(c) for c in crops], dim=0)
        crop_mask = torch.ones(self.num_crops, dtype=torch.long)
        return tensors, crop_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        text_ids, text_mask = self._tokenize(sample["caption"])
        crops, crop_mask = self._load_crops(sample["image_path"], train=True)
        return text_ids, text_mask, crops, crop_mask, sample["idx"]


class FIQHBIQueryDataset(Dataset):
    """Val/test queries: ModifiedImageCaption + candidate/target ids (no CIRR img_set)."""

    SPECIAL_TOKEN = FIQHBITrainDataset.SPECIAL_TOKEN

    def __init__(
        self,
        split: str,
        tokenizer,
        max_words: int = 77,
        categories: Optional[Sequence[str]] = None,
        image_root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.image_root = image_root
        self.categories = list(categories or fiq_paths.CATEGORIES)

        self.queries = []
        n_empty_mod = 0
        for cat in self.categories:
            llm_path = fiq_paths.llm_json(cat, split)
            with open(llm_path, "r") as f:
                rows = json.load(f)
            for row_i, row in enumerate(rows):
                candidate = str(row["candidate"])
                target = str(row["target"])
                caption = str(row.get("ModifiedImageCaption", "") or "")
                if not caption.strip():
                    n_empty_mod += 1
                self.queries.append(
                    {
                        "idx": len(self.queries),
                        "category": cat,
                        "caption": caption,
                        "reference_name": candidate,
                        "target_name": target,
                        "src_index": row_i,
                    }
                )
        self.n_empty_mod = n_empty_mod
        print(
            f"[FIQHBIQueryDataset] split={split} cats={self.categories} "
            f"queries={len(self.queries)} empty_mod={n_empty_mod}"
        )

    def __len__(self):
        return len(self.queries)

    def _tokenize(self, text: str):
        words = self.tokenizer.tokenize(text)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        max_len = self.max_words - 1
        if len(words) > max_len:
            words = words[:max_len]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        return torch.LongTensor(input_ids), torch.LongTensor(input_mask)

    def __getitem__(self, index):
        q = self.queries[index]
        text_ids, text_mask = self._tokenize(q["caption"])
        return (
            text_ids,
            text_mask,
            q["reference_name"],
            q["target_name"],
            q["category"],
            q["idx"],
        )


class FIQHBIGalleryDataset(Dataset):
    """Category-specific classic gallery: split json names with multi-crop CLS inputs."""

    def __init__(
        self,
        split: str,
        category: str,
        num_crops: int = 6,
        image_resolution: int = 224,
        image_root: Optional[str] = None,
    ):
        self.num_crops = num_crops
        self.image_resolution = image_resolution
        self.image_root = image_root
        self.category = category
        self.split = split

        with open(fiq_paths.split_json(category, split), "r") as f:
            names = json.load(f)

        self.items = []
        n_missing = 0
        seen = set()
        for name in names:
            name = str(name)
            if name in seen:
                continue
            seen.add(name)
            path = fiq_paths.image_path(name, self.image_root)
            if path is None:
                n_missing += 1
                continue
            self.items.append((name, path))
        self.n_missing = n_missing
        print(
            f"[FIQHBIGalleryDataset] split={split} cat={category} "
            f"gallery={len(self.items)} missing_img={n_missing}"
        )
        if not self.items:
            roots = list(fiq_paths.iter_image_roots(self.image_root))
            raise FileNotFoundError(
                f"No Fashion-IQ gallery images for {category}/{split}. Looked in: {roots}"
            )

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
        return len(self.items)

    def __getitem__(self, index):
        name, path = self.items[index]
        image = Image.open(path).convert("RGB")
        w, h = image.size
        if min(w, h) < 2:
            scale = max(2 / w, 2 / h)
            image = image.resize(
                (max(2, int(w * scale)), max(2, int(h * scale))),
                Image.BICUBIC,
            )
        crops = multi_crop_pil(image, self.num_crops)
        tensors = torch.stack([self.transform(c) for c in crops], dim=0)
        crop_mask = torch.ones(self.num_crops, dtype=torch.long)
        return tensors, crop_mask, name, index
