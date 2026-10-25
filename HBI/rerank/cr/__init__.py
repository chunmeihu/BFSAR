"""Vendored Candidate-Reranking (CR) BLIP Stage-1/2.

Upstream: https://github.com/Cuberick-Orion/Candidate-Reranking-CIR (MIT).
Code under ``src/`` and ``configs/``; checkpoints under ``weights/cr/``.

Load via ``HBI.cirr.vision_desc.hbi_blip_rerank.load_blip_models``.
"""

from HBI.rerank.paths import (  # noqa: F401
    CR_CONFIGS,
    CR_ROOT,
    CR_SRC,
    CR_WEIGHTS,
    DEFAULT_CIRR_STAGE1,
    DEFAULT_CIRR_STAGE2,
    DEFAULT_FIQ_STAGE1,
    DEFAULT_FIQ_STAGE2,
)
