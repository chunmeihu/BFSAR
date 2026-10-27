"""CIRR dataset for BFSAR text-to-image retrieval (ModifiedImageCaption -> target_hard)."""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop

from bfsar.cirr import cirr_paths
from bfsar.cirr.image_multi_crop import multi_crop_pil


class CIRRTrainDataset(Dataset):
    """Train pairs: LLM ModifiedImageCaption + target_hard multi-crop CLS inputs."""

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
        max_words: int = 32,
        num_crops: int = 12,
        image_resolution: int = 224,
        caption_json: Optional[str] = None,
        split_json: Optional[str] = None,
        llm_json: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.num_crops = num_crops
        self.image_resolution = image_resolution
        self.image_root = image_root or cirr_paths.CIRR_IMAGE_ROOT

        caption_json = caption_json or cirr_paths.caption_json(split)
        split_json = split_json or cirr_paths.split_json(split)
        llm_json = llm_json or cirr_paths.llm_json(split)

        with open(caption_json, "r") as f:
            self.triplets = json.load(f)
        with open(split_json, "r") as f:
            self.name_to_relpath = json.load(f)
        with open(llm_json, "r") as f:
            self.llm_caption = json.load(f)

        self.samples = []
        for idx, triplet in enumerate(self.triplets):
            ref = triplet["reference"]
            tgt = triplet["target_hard"]
            key = cirr_paths.pair_key(ref, tgt)
            if key not in self.llm_caption:
                continue
            caption = self.llm_caption[key]["ModifiedImageCaption"]
            img_path = cirr_paths.image_path(self.name_to_relpath, tgt, self.image_root)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Missing target image: {img_path}")
            self.samples.append(
                {
                    "idx": idx,
                    "caption": caption,
                    "target_name": tgt,
                    "image_path": img_path,
                }
            )

        print(f"[CIRRTrainDataset] split={split} pairs={len(self.samples)}")

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
            # Upsample tiny placeholders so grid crops and RandomResizedCrop are valid.
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


class CIRRQueryDataset(Dataset):
    """Val/test queries: ModifiedImageCaption only (+ metadata for CIRR metrics)."""

    SPECIAL_TOKEN = CIRRTrainDataset.SPECIAL_TOKEN

    def __init__(
        self,
        split: str,
        tokenizer,
        max_words: int = 32,
        caption_json: Optional[str] = None,
        llm_json: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words

        caption_json = caption_json or cirr_paths.caption_json(split)
        llm_json = llm_json or cirr_paths.llm_json(split)

        with open(caption_json, "r") as f:
            self.triplets = json.load(f)
        with open(llm_json, "r") as f:
            self.llm_caption = json.load(f)

        self.queries = []
        for idx, triplet in enumerate(self.triplets):
            ref = triplet["reference"]
            tgt = triplet["target_hard"]
            key = cirr_paths.pair_key(ref, tgt)
            caption = self.llm_caption[key]["ModifiedImageCaption"]
            group_members = triplet["img_set"]["members"]
            self.queries.append(
                {
                    "idx": idx,
                    "caption": caption,
                    "reference_name": ref,
                    "target_name": tgt,
                    "group_members": group_members,
                }
            )
        print(f"[CIRRQueryDataset] split={split} queries={len(self.queries)}")

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
            q["group_members"],
            q["idx"],
        )


class CIRRGalleryDataset(Dataset):
    """Classic gallery: all images in split with multi-crop tensors."""

    def __init__(
        self,
        split: str,
        num_crops: int = 12,
        image_resolution: int = 224,
        split_json: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        self.num_crops = num_crops
        self.image_root = image_root or cirr_paths.CIRR_IMAGE_ROOT

        split_json = split_json or cirr_paths.split_json(split)
        with open(split_json, "r") as f:
            self.name_to_relpath = json.load(f)

        self.image_names = sorted(self.name_to_relpath.keys())
        print(f"[CIRRGalleryDataset] split={split} gallery={len(self.image_names)}")

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
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        path = cirr_paths.image_path(self.name_to_relpath, name, self.image_root)
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
