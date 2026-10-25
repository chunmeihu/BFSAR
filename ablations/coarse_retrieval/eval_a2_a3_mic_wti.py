#!/usr/bin/env python3
"""Eval A2/A3 MIC WTI from an ablation checkpoint (FIQ Original-Split / CIRR val)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir


def _find_best_ckpt(ckpt_dir: Path) -> Path:
    if ckpt_dir.is_file():
        return ckpt_dir
    bests = sorted(ckpt_dir.glob("pytorch_model.bin.best.*"))
    if bests:
        # prefer highest epoch suffix
        def ep(p: Path):
            try:
                return int(p.name.rsplit(".", 1)[-1])
            except ValueError:
                return -1
        return max(bests, key=ep)
    plain = sorted(ckpt_dir.glob("pytorch_model.bin.*"))
    plain = [p for p in plain if ".best." not in p.name]
    if plain:
        return max(plain, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"no checkpoint under {ckpt_dir}")


def eval_fiq(ckpt: Path, out: Path, device: str, wti_levels: int, batch: int):
    from HBI.fiq.openclip_backbone import install_openclip_hbi_patches
    install_openclip_hbi_patches()
    from HBI.cirr.modeling_hbi_cirr import HBICIRR
    from HBI.fiq import fiq_paths
    from HBI.fiq.eval_fiq_hbi import evaluate_fiq_val
    from HBI.fiq.fiq_dataloader_builders import (
        build_fiq_gallery_dataloader,
        build_fiq_query_dataloader,
    )
    from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

    args = SimpleNamespace(
        base_encoder="ViT-L/14",
        clip_checkpoint=fiq_paths.CLIP_VITL14_CHECKPOINT,
        agg_module="None",
        interaction="wti",
        num_hidden_layers=4,
        estimator="None",
        kl=2.0,
        skl=0.0,
        v_rate0=0.25,
        v_rate1=0.5,
        t_rate0=0.25,
        t_rate1=0.5,
        rate=[1.0, 1.0, 1.0],
        wti_levels=wti_levels,
        max_frames=12,
        max_words=32,
        cross_num_hidden_layers=4,
        distributed=0,
        image_encode_microbatch=12,
        batch_size_val=batch,
        workers=4,
        fiq_image_root=None,
        fiq_categories="dress,shirt,toptee",
        local_rank=0,
        image_resolution=224,
    )
    model = HBICIRR(args)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    tokenizer = ClipTokenizer()
    cats = ("dress", "shirt", "toptee")
    loaders = {}
    for cat in cats:
        q, _, _ = build_fiq_query_dataloader(args, tokenizer, split="val", category=cat)
        g, _, _ = build_fiq_gallery_dataloader(args, split="val", category=cat)
        loaders[cat] = (q, g)
    metrics = evaluate_fiq_val(model, loaders, device, query_batch=batch, gallery_batch=batch)
    payload = {
        "id": out.name.split("_")[0],
        "name": "mic_wti",
        "dataset": "fiq",
        "ckpt": str(ckpt),
        "wti_levels": wti_levels,
        "mean": {
            "recall_at10": metrics["recall_at10"],
            "recall_at50": metrics["recall_at50"],
        },
        "per": {
            c: {
                "recall_at10": metrics[f"{c}_recall_at10"],
                "recall_at50": metrics[f"{c}_recall_at50"],
                "n_queries": metrics.get(f"{c}_n_queries"),
            }
            for c in cats
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["mean"], indent=2), flush=True)


def eval_cirr(ckpt: Path, out: Path, device: str, wti_levels: int, batch: int):
    from HBI.cirr.openclip_backbone import install_openclip_cirr_patches
    install_openclip_cirr_patches()
    from HBI.cirr.eval_cirr_hbi import evaluate_cirr_val
    from HBI.cirr.cirr_dataloader_builders import (
        build_cirr_gallery_dataloader,
        build_cirr_query_dataloader,
    )
    from HBI.cirr.modeling_hbi_cirr import HBICIRR
    from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

    clip_ckpt = str(ROOT / "weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt")
    args = SimpleNamespace(
        base_encoder="ViT-L/14",
        clip_checkpoint=clip_ckpt,
        agg_module="None",
        interaction="wti",
        num_hidden_layers=4,
        estimator="None",
        kl=2.0,
        skl=0.0,
        v_rate0=0.25,
        v_rate1=0.5,
        t_rate0=0.25,
        t_rate1=0.5,
        rate=[1.0, 1.0, 1.0],
        wti_levels=wti_levels,
        max_frames=12,
        max_words=32,
        cross_num_hidden_layers=4,
        distributed=0,
        image_encode_microbatch=12,
        cirr_image_root=None,
        batch_size_val=batch,
        workers=4,
        local_rank=0,
        image_resolution=224,
    )
    model = HBICIRR(args)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    tokenizer = ClipTokenizer()
    q, _, _ = build_cirr_query_dataloader(args, tokenizer, split="val")
    g, _, _ = build_cirr_gallery_dataloader(args, split="val")
    metrics = evaluate_cirr_val(model, q, g, device, query_batch=batch, gallery_batch=batch)
    payload = {
        "id": out.name.split("_")[0],
        "name": "mic_wti",
        "dataset": "cirr",
        "ckpt": str(ckpt),
        "wti_levels": wti_levels,
        "metrics": {
            "recall_at1": metrics["recall_at1"],
            "recall_at5": metrics["recall_at5"],
            "recall_at10": metrics["recall_at10"],
            "recall_at50": metrics["recall_at50"],
            "group_recall_at1": metrics["group_recall_at1"],
            "group_recall_at2": metrics["group_recall_at2"],
            "group_recall_at3": metrics["group_recall_at3"],
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["metrics"], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("fiq", "cirr"), required=True)
    p.add_argument("--ckpt", required=True, help="Checkpoint file or training OUT_DIR")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--wti_levels", type=int, default=3, choices=[1, 3])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    out = ensure_ablation_out_dir(args.out_dir)
    ckpt = _find_best_ckpt(Path(args.ckpt))
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"eval {args.dataset} ckpt={ckpt} wti_levels={args.wti_levels} → {out}", flush=True)
    if args.dataset == "fiq":
        eval_fiq(ckpt, out, device, args.wti_levels, args.batch_size)
    else:
        eval_cirr(ckpt, out, device, args.wti_levels, args.batch_size)


if __name__ == "__main__":
    main()
