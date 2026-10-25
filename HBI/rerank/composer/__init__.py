"""Composer rerank (BLIP-2 QTR Stage-A).

Model / train / eval code lives next to dataset pipelines:

  CIRR: ``HBI.cirr.vision_desc.model_cirr_blip2_qtr``
        ``HBI.cirr.vision_desc.train_cirr_blip2_qtr``
        ``HBI.cirr.vision_desc.cirr_blip2_qtr_rerank``

  FIQ:  ``HBI.fiq.vision_desc.train_fiq_blip2_qtr``
        ``HBI.fiq.vision_desc.fiq_blip2_qtr_rerank``

This package is the sibling entry of ``HBI.rerank.cr`` for discovery and paths.
"""

from HBI.rerank.paths import COMPOSER_CIRR_DEFAULT, COMPOSER_FIQ_DEFAULT  # noqa: F401
