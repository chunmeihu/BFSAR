"""Abort if ablation code would write into mainline output dirs."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATIONS_ROOT = REPO_ROOT / "output" / "ablations"

# Substrings that must never be written by ablation runners.
PROTECTED_SUBSTR = (
    "/output/fiq_retrieval_openclip/",
    "/output/fiq_retrieval/",
    "/output/fiq_estimator_openclip/",
    "/output/fiq_estimator/",
    "/output/cirr_retrieval_openclip/",
    "/output/cirr_retrieval/",
    "/output/cirr_estimator_openclip/",
    "/output/cirr_estimator/",
    "/output/fiq_vision_desc/refsim_fusion",
    "/output/fiq_vision_desc/sdr_dual",
    "/output/cirr_vision_desc/sdr_wti_native",
    "/output/cirr_vision_desc/sdr_dual",
    "/output/fiq_rerank/fusion_k",
    "/output/fiq_rerank/ksweep_",
    "/output/cirr_rerank/fusion_openclip_dual",
)


def assert_ablation_out_dir(out_dir: str | os.PathLike) -> Path:
    """Require OUT_DIR under output/ablations/; abort on mainline paths."""
    path = Path(out_dir).resolve()
    path_s = str(path).replace("\\", "/") + "/"
    abl_s = str(ABLATIONS_ROOT.resolve()).replace("\\", "/") + "/"
    if not path_s.startswith(abl_s):
        raise SystemExit(
            f"ABORT: ablation OUT_DIR must be under {ABLATIONS_ROOT}, got: {out_dir}"
        )
    for sub in PROTECTED_SUBSTR:
        if sub in path_s:
            raise SystemExit(
                f"ABORT: refusing to write into protected mainline path ({sub}): {out_dir}"
            )
    return path


def ensure_ablation_out_dir(out_dir: str | os.PathLike) -> Path:
    path = assert_ablation_out_dir(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
