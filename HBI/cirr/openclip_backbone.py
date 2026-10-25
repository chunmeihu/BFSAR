"""OpenCLIP ViT-L-14 (laion2b_s32b_b82k) backbone hooks for CIRR HBI.

Reuses the shared OpenCLIP→HBI conversion in ``HBI.fiq.openclip_backbone``.
Patches CIRR dataloaders to LAION mean/std=0.5. Does not touch OpenAI CIRR dirs.
"""

from __future__ import annotations

import os

from HBI.fiq.openclip_backbone import (  # noqa: F401
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
    # Ablation runners write under output/ablations/ (no 'openclip' substring required).
    if "/output/ablations/" in path:
        return
    if "openclip" not in path.lower():
        raise ValueError(
            f"CIRR OpenCLIP runs must write under an *openclip* output dir, got: {output_dir}"
        )


def install_openclip_cirr_patches() -> None:
    """Must run before importing HBICIRR / main_cirr_hbi_*.py.

    Also patches ``CIRRUniqueRefHBIDataset`` (Path_ref S_r) to LAION mean/std=0.5.
    Gallery loaders are already OpenCLIP via ``cirr_dataloader_builders_openclip``.
    """
    import HBI.cirr.cirr_dataloader_builders as builders
    import HBI.cirr.clip_loader as clip_loader
    from HBI.cirr.cirr_dataloader_builders_openclip import (
        build_cirr_gallery_dataloader,
        build_cirr_train_dataloader,
        build_cirr_val_trainstyle_dataloader,
    )

    clip_loader.build_clip_from_config = build_clip_from_config
    builders.build_cirr_train_dataloader = build_cirr_train_dataloader
    builders.build_cirr_gallery_dataloader = build_cirr_gallery_dataloader
    builders.build_cirr_val_trainstyle_dataloader = build_cirr_val_trainstyle_dataloader

    # Path_ref S_r uses a separate dataset with hardcoded OpenAI Normalize.
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )

    import HBI.cirr.vision_desc.eval_hbi_triple_composed_relative_fuse as fuse_mod

    _orig_init = fuse_mod.CIRRUniqueRefHBIDataset.__init__

    def _openclip_ref_init(self, ref_names, num_crops=12, image_resolution=224):
        _orig_init(self, ref_names, num_crops=num_crops, image_resolution=image_resolution)
        self.transform = Compose(
            [
                Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_resolution),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(OPENCLIP_MEAN, OPENCLIP_STD),
            ]
        )

    fuse_mod.CIRRUniqueRefHBIDataset.__init__ = _openclip_ref_init
