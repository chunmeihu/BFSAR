"""OpenCLIP ViT-L-14 (laion2b_s32b_b82k) backbone hooks for CIRR bfsar.

Reuses the shared OpenCLIP→BFSAR conversion in ``bfsar.fiq.openclip_backbone``.
Patches CIRR dataloaders to LAION mean/std=0.5. Does not touch OpenAI CIRR dirs.
"""

from __future__ import annotations

import os

from bfsar.fiq.openclip_backbone import (  # noqa: F401
    DEFAULT_CACHE_DIR,
    DEFAULT_CONVERTED_PATH,
    OPENCLIP_MEAN,
    OPENCLIP_MODEL,
    OPENCLIP_PRETRAINED,
    OPENCLIP_STD,
    build_clip_from_config,
    export_openclip_state_dict,
    load_openclip_state_dict,
)

PROTECTED_OUTPUT_SUBSTR = (
    "/output/cirr_retrieval/",
    "/output/fiq_estimator/",
    "/output/fiq_estimator_crop6/",
    "/output/fiq_retrieval/",
    "/output/fiq_retrieval_crop6/",
)


def assert_isolated_output_dir(output_dir: str) -> None:
    path = os.path.abspath(output_dir).replace("\\", "/") + "/"
    for sub in PROTECTED_OUTPUT_SUBSTR:
        if sub in path:
            raise ValueError(
                f"Refusing to overwrite protected OpenAI-CLIP run dir: {output_dir}"
            )
    return


def install_openclip_cirr_patches() -> None:
    """Install the OpenCLIP backbone before the retrieval model is imported."""
    import bfsar.cirr.cirr_dataloader_builders as builders
    import bfsar.cirr.clip_loader as clip_loader
    from bfsar.cirr.cirr_dataloader_builders_openclip import (
        build_cirr_gallery_dataloader,
        build_cirr_train_dataloader,
        build_cirr_val_trainstyle_dataloader,
    )

    clip_loader.build_clip_from_config = build_clip_from_config
    builders.build_cirr_train_dataloader = build_cirr_train_dataloader
    builders.build_cirr_gallery_dataloader = build_cirr_gallery_dataloader
    builders.build_cirr_val_trainstyle_dataloader = build_cirr_val_trainstyle_dataloader
