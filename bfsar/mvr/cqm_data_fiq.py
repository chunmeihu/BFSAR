"""Fashion-IQ data for BLIP-2 QTR cqm (Stage-A ITC).

Uses official dual captions (not ModifiedImageCaption / not CIRR weights).
Caption randomization matches Candidate-Reranking-CIR Stage-I.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import PIL.Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

from bfsar.fiq import fiq_paths

PIL.Image.MAX_IMAGE_PIXELS = None


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    """Pad if aspect ratio exceeds target_ratio (CR / CIRR cqm preprocess)."""

    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:
            return image
        scaled_max_wh = max(w, h) / self.target_ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        return TF.pad(image, [hp, vp, hp, vp], 0, "constant")


def fiq_targetpad_preprocess(target_ratio: float = 1.25, dim: int = 384):
    """Same 384 targetpad as CIRR QTR; Blip2QTR resizes to 224 internally."""
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def join_fiq_captions(captions: Sequence[str]) -> str:
    parts = [str(c).strip(".?, ").strip() for c in (captions or []) if str(c).strip()]
    if len(parts) >= 2:
        return f"{parts[0].capitalize()} and {parts[1]}"
    if parts:
        return parts[0].capitalize()
    return " "


def generate_randomized_fiq_caption(cap_a: Sequence[str], cap_b: Sequence[str]) -> List[str]:
    """CR 4-way caption aug: A and B / B and A / A / B."""
    out = []
    for a, b in zip(cap_a, cap_b):
        a = str(a).strip(".?, ").strip()
        b = str(b).strip(".?, ").strip()
        r = random.random()
        if r < 0.25:
            out.append(f"{a.capitalize()} and {b}" if a else b.capitalize())
        elif r < 0.5:
            out.append(f"{b.capitalize()} and {a}" if b else a.capitalize())
        elif r < 0.75:
            out.append(a.capitalize() if a else " ")
        else:
            out.append(b.capitalize() if b else " ")
    return out


def fiq_qtr_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class FIQClassicImages(Dataset):
    def __init__(self, names: Sequence[str], preprocess):
        self.names = list(names)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        path = fiq_paths.image_path(name)
        if path is None:
            raise FileNotFoundError(f"Missing FIQ image: {name}")
        img = self.preprocess(PIL.Image.open(path).convert("RGB"))
        return name, img


class FIQQTRRelativeTrain(Dataset):
    """Train: (ref_img, tgt_img, cap_a, cap_b) mixed over dress/shirt/toptee."""

    def __init__(
        self,
        preprocess,
        categories: Optional[Sequence[str]] = None,
        split: str = "train",
    ):
        self.preprocess = preprocess
        self.categories = list(categories or fiq_paths.CATEGORIES)
        self.split = split
        self.samples = []
        n_missing = 0
        for cat in self.categories:
            with open(fiq_paths.caption_json(cat, split), "r", encoding="utf-8") as f:
                rows = json.load(f)
            for row in rows:
                cand = str(row["candidate"])
                tgt = str(row["target"])
                caps = row.get("captions") or ["", ""]
                cap_a = str(caps[0]) if len(caps) > 0 else ""
                cap_b = str(caps[1]) if len(caps) > 1 else ""
                if fiq_paths.image_path(cand) is None or fiq_paths.image_path(tgt) is None:
                    n_missing += 1
                    continue
                self.samples.append(
                    {
                        "category": cat,
                        "candidate": cand,
                        "target": tgt,
                        "cap_a": cap_a,
                        "cap_b": cap_b,
                    }
                )
        self.n_missing = n_missing
        print(
            f"[FIQQTRRelativeTrain] split={split} cats={self.categories} "
            f"pairs={len(self.samples)} missing_img={n_missing}"
        )
        if not self.samples:
            raise FileNotFoundError("No Fashion-IQ train pairs with both images found.")

    def __len__(self):
        return len(self.samples)

    def _load(self, name: str):
        path = fiq_paths.image_path(name)
        return self.preprocess(PIL.Image.open(path).convert("RGB"))

    def __getitem__(self, index):
        row = self.samples[index]
        try:
            ref = self._load(row["candidate"])
            tgt = self._load(row["target"])
        except Exception:
            return None
        return ref, tgt, row["cap_a"], row["cap_b"]


def load_fiq_queries(category: str, split: str = "val") -> Tuple[List[str], List[str], List[str]]:
    with open(fiq_paths.caption_json(category, split), "r", encoding="utf-8") as f:
        rows = json.load(f)
    refs = [str(r["candidate"]) for r in rows]
    tgts = []
    for i, r in enumerate(rows):
        if "target" in r and r["target"] is not None and str(r["target"]).strip():
            tgts.append(str(r["target"]))
        else:
            tgts.append(f"__q{i}__")
    caps = [join_fiq_captions(r.get("captions") or []) for r in rows]
    return refs, tgts, caps


def load_fiq_val_queries(category: str) -> Tuple[List[str], List[str], List[str]]:
    return load_fiq_queries(category, "val")


def default_topk_pt(category: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "output" / "fiq_vision_desc" / f"hbi_top200_val_{category}.pt"
