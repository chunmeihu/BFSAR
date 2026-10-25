"""Default paths for Composer and Candidate-Reranking (CR) backends."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- Candidate-Reranking (vendored under HBI/rerank/cr) ----
CR_ROOT = Path(os.environ.get("GBCIR_CR_ROOT", REPO_ROOT / "HBI" / "rerank" / "cr"))
CR_SRC = CR_ROOT / "src"
CR_CONFIGS = CR_ROOT / "configs"
CR_WEIGHTS = Path(os.environ.get("GBCIR_CR_WEIGHTS", REPO_ROOT / "weights" / "cr"))

DEFAULT_CIRR_STAGE1 = CR_WEIGHTS / "stage1/CIRR/saved_models/blip_mean.pt"
DEFAULT_CIRR_STAGE2 = CR_WEIGHTS / "stage2/CIRR/saved_models/blip_mean.pt"
DEFAULT_FIQ_STAGE1 = CR_WEIGHTS / "stage1/fashionIQ/saved_models/blip.pt"
DEFAULT_FIQ_STAGE2 = CR_WEIGHTS / "stage2/fashionIQ/saved_models/blip.pt"

# HBI coarse Top-K in CR-compatible .pt format (used by composer/CR fusion).
DEFAULT_CIRR_TOPK_VAL = CR_WEIGHTS / "HBI_coarse/cirr_top_200_val.pt"
DEFAULT_CIRR_TOPK_TEST1 = CR_WEIGHTS / "HBI_coarse/cirr_top_200_test1.pt"

# ---- Composer (BLIP-2 QTR Stage-A); code under vision_desc, ckpts under output/ ----
COMPOSER_CIRR_DEFAULT = (
    REPO_ROOT / "output/cirr_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r1.pt"
)
COMPOSER_FIQ_DEFAULT = (
    REPO_ROOT / "output/fiq_rerank/blip2_qtr/qtr_v1_stageA_g32/qtr_stagea_best_pool_r10.pt"
)
