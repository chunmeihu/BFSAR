"""OpenCLIP ViT-L-14 (laion2b_s32b_b82k) as the HBI CLIP backbone.

Does not modify HBI/cirr/clip_loader.py or the OpenAI ViT-L/14 FIQ runs.
OpenCLIP LAION-2B uses nn.GELU (not QuickGELU) and Inception mean/std 0.5.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Tuple

import torch
from torch import nn

from HBI.models.module_clip import CLIP, convert_weights, QuickGELU as VisualQuickGELU
from HBI.models.module_transformer import QuickGELU as TextQuickGELU

OPENCLIP_MODEL = "ViT-L-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b82k"
OPENCLIP_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
OPENCLIP_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CACHE_DIR = os.path.join(_ROOT, "weights", "open_clip")
DEFAULT_CONVERTED_PATH = os.path.join(
    DEFAULT_CACHE_DIR, "hbi_format", "ViT-L-14-laion2B-s32B-b82K.pt"
)

PROTECTED_OUTPUT_SUBSTR = (
    "/output/fiq_estimator/",
    "/output/fiq_estimator_crop6/",
    "/output/fiq_retrieval/",
    "/output/fiq_retrieval_crop6/",
)


def assert_isolated_output_dir(output_dir: str) -> None:
    path = os.path.abspath(output_dir).replace("\\", "/") + "/"
    if "cirr" in path.lower():
        raise ValueError(f"FIQ OpenCLIP run must not write under a cirr path: {output_dir}")
    for sub in PROTECTED_OUTPUT_SUBSTR:
        if sub in path:
            raise ValueError(
                f"Refusing to overwrite original FIQ OpenAI-CLIP run dir: {output_dir}"
            )


def replace_quick_gelu(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (VisualQuickGELU, TextQuickGELU)):
            setattr(module, name, nn.GELU())
            n += 1
        else:
            n += replace_quick_gelu(child)
    return n


def _meta_from_state_dict(state_dict, model_path: str):
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks"))
    )
    return SimpleNamespace(
        state_dict=state_dict,
        model_path=model_path,
        embed_dim=embed_dim,
        context_length=context_length,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        image_resolution=image_resolution,
        vision_width=vision_width,
        vision_layers=vision_layers,
        vision_patch_size=vision_patch_size,
        vocab_size=vocab_size,
        transformer_layers=transformer_layers,
    )


def export_openclip_state_dict(
    cache_dir: str = DEFAULT_CACHE_DIR,
    converted_path: str = DEFAULT_CONVERTED_PATH,
) -> str:
    """Load OpenCLIP once and dump an HBI-compatible state_dict (same keys as OpenAI CLIP)."""
    import open_clip

    os.makedirs(os.path.dirname(converted_path), exist_ok=True)
    if os.path.isfile(converted_path):
        return converted_path
    model, _, _ = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL,
        pretrained=OPENCLIP_PRETRAINED,
        cache_dir=cache_dir,
    )
    state_dict = {k.replace("module.", ""): v.detach().cpu() for k, v in model.state_dict().items()}
    tmp_path = converted_path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, converted_path)
    return converted_path


def load_openclip_state_dict(
    cache_dir: str = DEFAULT_CACHE_DIR,
    converted_path: str = DEFAULT_CONVERTED_PATH,
):
    if os.path.isfile(converted_path):
        state_dict = torch.load(converted_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        return state_dict, converted_path

    path = export_openclip_state_dict(cache_dir=cache_dir, converted_path=converted_path)
    state_dict = torch.load(path, map_location="cpu")
    return state_dict, path


def build_clip_from_config(config):
    cache_dir = getattr(config, "openclip_cache_dir", None) or DEFAULT_CACHE_DIR
    converted_path = getattr(config, "openclip_converted_path", None) or DEFAULT_CONVERTED_PATH
    state_dict, model_path = load_openclip_state_dict(cache_dir, converted_path)
    meta = _meta_from_state_dict(state_dict, model_path)

    clip = CLIP(
        meta.embed_dim,
        meta.image_resolution,
        meta.vision_layers,
        meta.vision_width,
        meta.vision_patch_size,
        meta.context_length,
        meta.vocab_size,
        meta.transformer_width,
        meta.transformer_heads,
        meta.transformer_layers,
    )
    n_gelu = replace_quick_gelu(clip)
    if n_gelu != 36:
        raise RuntimeError(
            f"Expected 36 QuickGELU→GELU swaps for ViT-L/14 (24 visual + 12 text), got {n_gelu}"
        )

    if torch.cuda.is_available():
        convert_weights(clip)

    meta.state_dict = state_dict
    meta.clip_backend = "openclip"
    meta.openclip_tag = f"{OPENCLIP_MODEL}/{OPENCLIP_PRETRAINED}"
    print(
        f"[openclip_backbone] loaded {meta.openclip_tag} from {model_path} "
        f"(GELU swaps={n_gelu}, mean/std=0.5)",
        flush=True,
    )
    return clip, meta


def install_openclip_hbi_patches() -> None:
    """Must run before importing HBICIRR / main_fiq_hbi_*.py."""
    import HBI.cirr.clip_loader as clip_loader
    import HBI.fiq.fiq_dataloader_builders as builders
    from HBI.fiq.fiq_dataloader_builders_openclip import (
        build_fiq_gallery_dataloader,
        build_fiq_train_dataloader,
    )

    clip_loader.build_clip_from_config = build_clip_from_config
    builders.build_fiq_train_dataloader = build_fiq_train_dataloader
    builders.build_fiq_gallery_dataloader = build_fiq_gallery_dataloader
