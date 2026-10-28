"""Checkpoint locations for composed query matching and candidate verification."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Official Candidate-Reranking-CIR checkout. Readers clone it themselves.
CVS_ROOT = Path(os.environ.get("BFSAR_CVS_ROOT", REPO_ROOT / "third_party" / "Candidate-Reranking-CIR"))
CVS_SRC = CVS_ROOT / "src"
CVS_CONFIGS = CVS_ROOT / "configs"
CVS_WEIGHTS = Path(os.environ.get("BFSAR_CVS_WEIGHTS", REPO_ROOT / "weights" / "cvs"))

CR_ROOT = CVS_ROOT
CR_SRC = CVS_SRC
CR_CONFIGS = CVS_CONFIGS
CR_WEIGHTS = CVS_WEIGHTS


def ensure_candidate_rerank_checkout() -> Path:
    """Inference imports the official checkout; this repository does not ship that code."""
    marker = CVS_SRC / "blip_stage1.py"
    if marker.is_file():
        return CVS_ROOT
    raise FileNotFoundError(
        "Candidate verification uses the official Candidate-Reranking-CIR code, "
        "which is not included in this repository.\n"
        "Clone https://github.com/Cuberick-Orion/Candidate-Reranking-CIR into\n"
        f"  {REPO_ROOT / 'third_party' / 'Candidate-Reranking-CIR'}\n"
        "or set BFSAR_CVS_ROOT to that checkout. Its license is in that repository."
    )

DEFAULT_CIRR_STAGE1 = CVS_WEIGHTS / "cirr" / "stage1.pt"
DEFAULT_CIRR_STAGE2 = CVS_WEIGHTS / "cirr" / "stage2.pt"
DEFAULT_FIQ_STAGE1 = CVS_WEIGHTS / "fashioniq" / "stage1.pt"
DEFAULT_FIQ_STAGE2 = CVS_WEIGHTS / "fashioniq" / "stage2.pt"

DEFAULT_CIRR_TOPK_VAL = REPO_ROOT / "output" / "topk" / "cirr_top_200_val.pt"

CQM_CIRR_DEFAULT = REPO_ROOT / "weights" / "cqm" / "cirr" / "cqm.pt"
CQM_FIQ_DEFAULT = REPO_ROOT / "weights" / "cqm" / "fashioniq" / "cqm.pt"
COMPOSER_CIRR_DEFAULT = CQM_CIRR_DEFAULT
COMPOSER_FIQ_DEFAULT = CQM_FIQ_DEFAULT

BLIP2_ITM_DIR = REPO_ROOT / "weights" / "blip2" / "blip2-itm-vit-g"
BERT_DIR = REPO_ROOT / "weights" / "bert" / "bert-base-uncased"
BLIP2_ITM_HUB = "Salesforce/blip2-itm-vit-g"


def blip2_pretrained_id(requested: str | None = None) -> str:
    """Prefer the copy under weights/blip2 over the Hugging Face hub id."""
    local = BLIP2_ITM_DIR
    ready = (local / "config.json").is_file() and (local / "model.safetensors").is_file()
    if requested and Path(requested).is_dir():
        return str(Path(requested))
    if ready and requested in (None, "", BLIP2_ITM_HUB):
        return str(local)
    if requested:
        return requested
    return str(local) if ready else BLIP2_ITM_HUB
