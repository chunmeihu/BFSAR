"""Default filesystem paths for CIRR data.

Images and official annotations live in one folder, ``data/cirr/cirr_dataset``:

  cirr_dataset/train|dev|test1     images from the official release
  cirr_dataset/cirr/captions
  cirr_dataset/cirr/image_splits
  llm_data/                         target captions included in this repo
  vision_desc/                      vision-side captions included in this repo

``img_raw`` is only a fallback for an older local copy whose images were
stored outside ``cirr_dataset``. Override with ``CIRR_DATA_ROOT``,
``CIRR_IMAGE_ROOT``, or ``CIRR_DATASET_ROOT`` if needed.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSET_ROOT = os.environ.get("ASSET_ROOT", os.path.expanduser("~"))

CIRR_DATA_ROOT = os.environ.get("CIRR_DATA_ROOT", os.path.join(REPO_ROOT, "data", "cirr"))
CIRR_DATASET_ROOT = os.environ.get(
    "CIRR_DATASET_ROOT", os.path.join(CIRR_DATA_ROOT, "cirr_dataset")
)
CIRR_CAPTIONS_DIR = os.path.join(CIRR_DATASET_ROOT, "cirr", "captions")
CIRR_SPLITS_DIR = os.path.join(CIRR_DATASET_ROOT, "cirr", "image_splits")
CIRR_LLM_DIR = os.environ.get("CIRR_LLM_DIR", os.path.join(CIRR_DATA_ROOT, "llm_data"))

# Local vision-side descriptions (generated dual targets) under this repo.
CIRR_VISION_DESC_DIR = os.path.join(CIRR_DATA_ROOT, "vision_desc")
CIRR_VISION_OUT_DIR = os.path.join(REPO_ROOT, "output", "cirr_vision_desc")

def _default_image_root() -> str:
    """Official release keeps images inside cirr_dataset. Fall back to img_raw."""
    chosen = os.environ.get("CIRR_IMAGE_ROOT")
    if chosen:
        return chosen
    official = CIRR_DATASET_ROOT
    legacy = os.path.join(CIRR_DATA_ROOT, "img_raw")
    if os.path.isdir(os.path.join(official, "dev")) or os.path.isdir(os.path.join(official, "train")):
        return official
    if os.path.isdir(legacy):
        return legacy
    return official


# Split json paths are relative to this root, e.g. ./dev/dev-244-0-img0.png
# or ./train/34/train-11041-2-img0.png.
CIRR_IMAGE_ROOT = _default_image_root()

CLIP_IN_REPO = os.path.join(REPO_ROOT, "weights", "clip", "ViT-L-14.pt")
CLIP_VITL14_CHECKPOINT = os.environ.get(
    "CLIP_VITL14_CHECKPOINT",
    CLIP_IN_REPO if os.path.isfile(CLIP_IN_REPO) else os.path.join(ASSET_ROOT, "ViT-L-14.pt"),
)


def caption_json(split: str) -> str:
    return os.path.join(CIRR_CAPTIONS_DIR, f"cap.rc2.{split}.json")


def split_json(split: str) -> str:
    return os.path.join(CIRR_SPLITS_DIR, f"split.rc2.{split}.json")


def llm_json(split: str) -> str:
    return os.path.join(CIRR_LLM_DIR, f"llm_info_cirr_{split}.json")


def vision_dual_json(split: str) -> str:
    """Generated vision-side dual targets: val / test1."""
    return os.path.join(CIRR_VISION_DESC_DIR, f"{split}_dual_targets.json")


def image_path(name_to_relpath: dict, image_name: str, image_root: str = CIRR_IMAGE_ROOT) -> str:
    rel = name_to_relpath[image_name].lstrip("./")
    return os.path.join(image_root, rel)


def pair_key(reference: str, target_hard: str) -> str:
    return reference + target_hard


def test1_llm_key(pairid, reference: str) -> str:
    """Keys in llm_info_cirr_test1.json."""
    return f"{pairid}{reference}"
