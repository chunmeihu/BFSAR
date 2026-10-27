"""CQM rerank (BLIP-2 QTR Stage-A).

Model / train / eval code lives next to dataset pipelines:

  CIRR: ``bfsar.mvr.cqm_model``
        ``bfsar.mvr.train_cqm_cirr``
        ``bfsar.mvr.cqm_rerank_cirr``

  FIQ:  ``bfsar.mvr.train_cqm_fiq``
        ``bfsar.mvr.cqm_rerank_fiq``

This package is the sibling entry of ``bfsar.mvr.cvs`` for discovery and paths.
"""

from bfsar.rerank.paths import COMPOSER_CIRR_DEFAULT, COMPOSER_FIQ_DEFAULT  # noqa: F401
