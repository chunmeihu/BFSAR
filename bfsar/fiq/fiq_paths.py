"""Default filesystem paths for Fashion-IQ.

Dataset lives under ``<repo>/data/fashioniq`` (captions, image_splits, llm_data, images).
Override with ``FIQ_DATA_ROOT`` / ``FIQ_IMAGE_ROOT`` if needed.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSET_ROOT = os.environ.get("ASSET_ROOT", os.path.expanduser("~"))

FIQ_DATA_ROOT = os.environ.get("FIQ_DATA_ROOT", os.path.join(REPO_ROOT, "data", "fashioniq"))
FIQ_CAPTIONS_DIR = os.path.join(FIQ_DATA_ROOT, "captions")
FIQ_SPLITS_DIR = os.path.join(FIQ_DATA_ROOT, "image_splits")
FIQ_LLM_DIR = os.path.join(FIQ_DATA_ROOT, "llm_data")
FIQ_IMAGES_ROOT = os.path.join(FIQ_DATA_ROOT, "images")
FIQ_VISION_DESC_DIR = os.path.join(FIQ_DATA_ROOT, "vision_desc")
FIQ_VISION_OUT_DIR = os.path.join(REPO_ROOT, "output", "fiq_vision_desc")

CLIP_IN_REPO = os.path.join(REPO_ROOT, "weights", "clip", "ViT-L-14.pt")
CLIP_VITL14_CHECKPOINT = os.environ.get(
    "CLIP_VITL14_CHECKPOINT",
    CLIP_IN_REPO if os.path.isfile(CLIP_IN_REPO) else os.path.join(ASSET_ROOT, "ViT-L-14.pt"),
)

CATEGORIES: Sequence[str] = ("dress", "shirt", "toptee")
IMAGE_EXTS: Sequence[str] = (".png", ".jpg", ".jpeg")


def caption_json(category: str, split: str) -> str:
    return os.path.join(FIQ_CAPTIONS_DIR, f"cap.{category}.{split}.json")


def split_json(category: str, split: str) -> str:
    return os.path.join(FIQ_SPLITS_DIR, f"split.{category}.{split}.json")


def llm_json(category: str, split: str) -> str:
    return os.path.join(FIQ_LLM_DIR, f"cap_llm.{category}.{split}.json")


def vision_dual_json(split: str) -> str:
    """Generated vision-side dual targets: val / test / train."""
    return os.path.join(FIQ_VISION_DESC_DIR, f"{split}_dual_targets.json")


def pair_key(candidate: str, target: str) -> str:
    return f"{candidate}_{target}"


def iter_image_roots(image_root: Optional[str] = None) -> Iterable[str]:
    env_root = os.environ.get("FIQ_IMAGE_ROOT")
    seen = set()
    for root in (image_root, env_root, FIQ_IMAGES_ROOT):
        if not root:
            continue
        root = os.path.realpath(root)
        if root in seen:
            continue
        seen.add(root)
        yield root


def image_path(image_name: str, image_root: Optional[str] = None) -> Optional[str]:
    """Resolve `{name}.png/.jpg` under images or an override root."""
    for root in iter_image_roots(image_root):
        for ext in IMAGE_EXTS:
            path = os.path.join(root, f"{image_name}{ext}")
            if os.path.isfile(path):
                return path
    return None


def count_images_in_roots(image_root: Optional[str] = None) -> int:
    n = 0
    for root in iter_image_roots(image_root):
        if not os.path.isdir(root):
            continue
        with os.scandir(root) as it:
            for entry in it:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTS:
                    n += 1
    return n
