"""Checkpoint locations for composed query matching and candidate verification."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CVS_ROOT = Path(os.environ.get("BFSAR_CVS_ROOT", REPO_ROOT / "bfsar" / "mvr" / "cvs"))
CVS_SRC = CVS_ROOT / "src"
CVS_CONFIGS = CVS_ROOT / "configs"
CVS_WEIGHTS = Path(os.environ.get("BFSAR_CVS_WEIGHTS", REPO_ROOT / "weights" / "cvs"))

# Names still used by the vendored verification trainer.
CR_ROOT = CVS_ROOT
CR_SRC = CVS_SRC
CR_CONFIGS = CVS_CONFIGS
CR_WEIGHTS = CVS_WEIGHTS

DEFAULT_CIRR_STAGE1 = CVS_WEIGHTS / "cirr" / "stage1.pt"
DEFAULT_CIRR_STAGE2 = CVS_WEIGHTS / "cirr" / "stage2.pt"
DEFAULT_FIQ_STAGE1 = CVS_WEIGHTS / "fashioniq" / "stage1.pt"
DEFAULT_FIQ_STAGE2 = CVS_WEIGHTS / "fashioniq" / "stage2.pt"

DEFAULT_CIRR_TOPK_VAL = REPO_ROOT / "output" / "topk" / "cirr_top_200_val.pt"
DEFAULT_CIRR_TOPK_TEST1 = REPO_ROOT / "output" / "topk" / "cirr_top_200_test1.pt"

CQM_CIRR_DEFAULT = REPO_ROOT / "weights" / "cqm" / "cirr" / "cqm.pt"
CQM_FIQ_DEFAULT = REPO_ROOT / "weights" / "cqm" / "fashioniq" / "cqm.pt"
COMPOSER_CIRR_DEFAULT = CQM_CIRR_DEFAULT
COMPOSER_FIQ_DEFAULT = CQM_FIQ_DEFAULT
