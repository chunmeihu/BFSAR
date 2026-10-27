"""Rerank backends used after coarse BFSAR retrieval.

Sibling packages
  - ``bfsar.rerank.cqm`` — BLIP-2 QTR CQM (Stage-A ITC)
  - ``bfsar.mvr.cvs`` — Candidate-Reranking BLIP Stage-1/2 (vendored)

Inference entrypoints live under ``bfsar.{cirr,fiq}.vision_desc``; this package
holds shared paths and the CR model source so the repo is self-contained.
"""

from bfsar.rerank.paths import (  # noqa: F401
    COMPOSER_FIQ_DEFAULT,
    COMPOSER_CIRR_DEFAULT,
    CR_CONFIGS,
    CR_ROOT,
    CR_SRC,
    CR_WEIGHTS,
    DEFAULT_FIQ_STAGE1,
    DEFAULT_FIQ_STAGE2,
    DEFAULT_CIRR_STAGE1,
    DEFAULT_CIRR_STAGE2,
    DEFAULT_CIRR_TOPK_VAL,
    DEFAULT_CIRR_TOPK_TEST1,
)
