"""Load CLIP checkpoint for CIRR HBI (supports external ViT-L/14 path)."""

import os
from types import SimpleNamespace

import torch

from HBI.models.module_clip import CLIP, convert_weights, _PT_NAME


def resolve_clip_path(config) -> str:
    clip_checkpoint = getattr(config, "clip_checkpoint", None)
    if clip_checkpoint and os.path.isfile(clip_checkpoint):
        return clip_checkpoint

    backbone = getattr(config, "base_encoder", "ViT-L/14")
    if backbone not in _PT_NAME:
        raise ValueError(f"Unknown base_encoder {backbone!r}, expected one of {list(_PT_NAME)}")

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_path = os.path.join(models_dir, _PT_NAME[backbone])
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"CLIP weights not found at {model_path}. "
            f"Set --clip_checkpoint (e.g. weights/clip/ViT-L-14.pt)."
        )
    return model_path


def load_clip_state_dict(config):
    model_path = resolve_clip_path(config)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    return state_dict, model_path


def build_clip_from_config(config):
    state_dict, model_path = load_clip_state_dict(config)

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

    clip = CLIP(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
    )

    if torch.cuda.is_available():
        convert_weights(clip)

    meta = SimpleNamespace(
        state_dict=state_dict,
        model_path=model_path,
        embed_dim=embed_dim,
        context_length=context_length,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        image_resolution=image_resolution,
    )
    return clip, meta
