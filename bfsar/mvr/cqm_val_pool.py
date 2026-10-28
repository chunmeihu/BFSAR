"""Build the validation candidate lists that composed-query training reads.

The lists are the top 200 images from the fine-grained retriever on the
validation split. Training writes them once and reuses the files. The
checkpoint kept for inference is the best validation epoch, not the last one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
OPENCLIP = REPO / "weights" / "openclip" / "ViT-L-14-laion2B-s32B-b82K.pt"
FGSA_CIRR = REPO / "weights" / "fgsa" / "cirr" / "pytorch_model.bin"
FGSA_FIQ = REPO / "weights" / "fgsa" / "fashioniq" / "pytorch_model.bin"


def _device(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return device
    return "cpu"


def _openclip() -> str:
    if OPENCLIP.is_file():
        return str(OPENCLIP)
    from bfsar.fiq.openclip_backbone import export_openclip_state_dict

    return export_openclip_state_dict()


def _fgsa_namespace(ckpt: Path, split: str, dataset: str):
    return SimpleNamespace(
        base_encoder="ViT-L/14",
        clip_checkpoint=_openclip(),
        agg_module="None",
        interaction="fgsa",
        num_hidden_layers=4,
        estimator="None",
        kl=2.0,
        lambda_b=0.0,
        v_rate0=0.25,
        v_rate1=0.5,
        t_rate0=0.25,
        t_rate1=0.5,
        rate=[1.0, 1.0, 1.0],
        fgsa_levels=1,
        max_frames=12,
        max_words=77,
        cross_num_hidden_layers=4,
        distributed=0,
        image_encode_microbatch=8,
        batch_size_val=8,
        query_batch=16,
        gallery_batch=32,
        workers=2,
        local_rank=0,
        image_resolution=224,
        gradient_checkpointing=1,
        split=split,
        recompute=False,
        ckpt=str(ckpt),
        fgsa_levels_eval=1,
        cirr_image_root=None,
        fiq_image_root=None,
        fiq_categories="dress,shirt,toptee",
        dual_json=None,
        dataset=dataset,
    )


def _load_fgsa(ns, ckpt: Path, device: str):
    from bfsar.cirr.modeling_fgsa import FGSAModel
    from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

    model = FGSAModel(ns)
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()
    return model, ClipTokenizer()


def _release(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_cirr_val_topk(dest: Path, k: int = 200, device: str = "cuda:0") -> Path:
    dest = Path(dest)
    if dest.is_file():
        return dest
    if not FGSA_CIRR.is_file():
        raise FileNotFoundError(
            f"Missing {FGSA_CIRR}. Train fine-grained semantic alignment on CIRR first."
        )
    device = _device(device)
    from bfsar.cirr import cirr_paths
    from bfsar.dpic.eval_cirr import coarse_matrix, encode_branches, _pack_from_npz
    from bfsar.mvr.cvs_infer import build_full_val_topk_pt, save_topk_pt

    coarse_dir = REPO / "output" / "dpic" / "cirr" / "val"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    npz = coarse_dir / "branch_sims.npz"
    ns = _fgsa_namespace(FGSA_CIRR, "val", "cirr")
    ns.dual_json = cirr_paths.vision_dual_json("val")
    if not npz.is_file():
        model, tokenizer = _load_fgsa(ns, FGSA_CIRR, device)
        try:
            encode_branches(model, ns, tokenizer, device, coarse_dir)
        finally:
            _release(model)
    pack = _pack_from_npz(npz)
    sim = coarse_matrix(pack)
    with open(cirr_paths.split_json("val"), "r", encoding="utf-8") as f:
        official = list(json.load(f).keys())
    col = {name: i for i, name in enumerate(pack["index_names"])}
    missing = [name for name in official if name not in col]
    if missing:
        raise RuntimeError(f"CIRR val gallery is missing {len(missing)} split images")
    sim = sim[:, [col[name] for name in official]]
    payload = build_full_val_topk_pt(sim, pack["refs"], pack["tgts"], official, k)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return Path(save_topk_pt(payload, dest))


def ensure_fiq_val_topk(dest_dir: Path, k: int = 200, device: str = "cuda:0") -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cats = ("dress", "shirt", "toptee")
    if all((dest_dir / f"coarse_top200_val_{cat}.pt").is_file() for cat in cats):
        return dest_dir
    if not FGSA_FIQ.is_file():
        raise FileNotFoundError(
            f"Missing {FGSA_FIQ}. Train fine-grained semantic alignment on FashionIQ first."
        )
    device = _device(device)
    from bfsar.dpic.eval_fiq import encode_category
    from bfsar.fiq import fiq_paths
    from bfsar.mvr.cvs_fiq import align_to_official, build_topk_payload
    from bfsar.mvr.fuse_fiq import _dual_from_dpic_npz

    coarse_dir = REPO / "output" / "dpic" / "fashioniq" / "val"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    ns = _fgsa_namespace(FGSA_FIQ, "val", "fashioniq")
    ns.dual_json = fiq_paths.vision_dual_json("val")
    need_encode = any(not (coarse_dir / f"scores_{cat}.npz").is_file() for cat in cats)
    if need_encode:
        model, tokenizer = _load_fgsa(ns, FGSA_FIQ, device)
        try:
            for cat in cats:
                encode_category(model, ns, tokenizer, cat, device, coarse_dir)
        finally:
            _release(model)
    for cat in cats:
        pt = dest_dir / f"coarse_top200_val_{cat}.pt"
        if pt.is_file():
            continue
        packed = _dual_from_dpic_npz(coarse_dir / f"scores_{cat}.npz")
        sim, index_names, refs, tgts = align_to_official(packed, cat, split="val")
        payload = build_topk_payload(sim, index_names, refs, tgts, cat, k, split="val")
        torch.save(payload, pt)
    return dest_dir
