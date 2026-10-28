#!/usr/bin/env python3
"""FashionIQ coarse retrieval: entity-only FGSA and dual-path scores.

Both paths use (alpha, beta) = (0.4, 0.2). The path mix weight is 0.5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.fiq.openclip_backbone import install_openclip_patches

install_openclip_patches()

from bfsar.cirr.eval_fgsa import _similarity_matrix, encode_gallery  # noqa: E402
from bfsar.cirr.modeling_fgsa import FGSAModel  # noqa: E402
from bfsar.dpic.encode_cirr import encode_three  # noqa: E402
from bfsar.fiq import fiq_paths  # noqa: E402
from bfsar.fiq.fiq_dataloader_builders import build_fiq_gallery_dataloader  # noqa: E402
from bfsar.dpic.norm import zscore_rows  # noqa: E402
from bfsar.dpic.refsim import ref_gallery_cosine  # noqa: E402
from bfsar.dpic.encode_fiq import (  # noqa: E402
    FIQTripleVisionDataset,
    collate_fn,
)
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402
from bfsar.dpic.calibrate import r_at, score_path, tgt_ranks  # noqa: E402

CATS = ("dress", "shirt", "toptee")
RECALL_KS = (10, 50, 100, 200)
FIQ_ALPHA, FIQ_BETA, FIQ_WREF = 0.4, 0.2, 0.5
OPENCLIP_CKPT = ROOT / "weights/openclip/ViT-L-14-laion2B-s32B-b82K.pt"
DEFAULT_CKPT = ROOT / "weights/fgsa/fashioniq/pytorch_model.bin"
DEFAULT_OUT = ROOT / "output/dpic/fashioniq"


def _find_best_ckpt(ckpt_dir: Path) -> Path:
    if ckpt_dir.is_file():
        return ckpt_dir
    bests = sorted(ckpt_dir.glob("pytorch_model.bin.best.*"))
    if bests:
        def ep(p: Path):
            try:
                return int(p.name.rsplit(".", 1)[-1])
            except ValueError:
                return -1
        return max(bests, key=ep)
    raise FileNotFoundError(f"no checkpoint under {ckpt_dir}")


def _metrics_from_ranks(rk: np.ndarray) -> Dict[str, float]:
    return {f"recall_at{k}": r_at(rk, k) for k in RECALL_KS}


def _pack_from_npz(path: Path) -> Dict:
    d = np.load(path, allow_pickle=True)
    idx = [str(x) for x in d["index_names"].tolist()]
    refs = [str(x) for x in d["refs"].tolist()]
    tgts = [str(x) for x in d["tgts"].tolist()]
    name = {n: i for i, n in enumerate(idx)}
    return {
        "zd": zscore_rows(d["modified"]),
        "zm": zscore_rows(d["relative"]),
        "zr_ref": zscore_rows(d["ref"]),
        "zr_vis": zscore_rows(d["vision"]),
        "s_modified": np.asarray(d["modified"], dtype=np.float32),
        "s_relative": np.asarray(d["relative"], dtype=np.float32),
        "s_vision": np.asarray(d["vision"], dtype=np.float32),
        "s_ref": np.asarray(d["ref"], dtype=np.float32),
        "ref_i": np.array([name.get(r, -1) for r in refs], dtype=np.int32),
        "tgt_i": np.array([name[t] for t in tgts], dtype=np.int32),
        "n_queries": len(refs),
        "n_gallery": len(idx),
    }


def _eval_sim(pack: Dict, sim: np.ndarray) -> Dict:
    rk = tgt_ranks(sim, pack["ref_i"], pack["tgt_i"])
    return {
        **_metrics_from_ranks(rk),
        "n_queries": int(pack["n_queries"]),
        "n_gallery": int(pack["n_gallery"]),
    }


def _mean_over(per: Dict[str, Dict], keys=RECALL_KS) -> Dict[str, float]:
    return {
        f"recall_at{k}": float(np.mean([per[c][f"recall_at{k}"] for c in CATS]))
        for k in keys
    }


@torch.no_grad()
def encode_category(model, args, tokenizer, category: str, device, out: Path) -> Path:
    npz = out / f"scores_{category}.npz"
    if npz.is_file() and not args.recompute:
        print(f"reuse {npz}", flush=True)
        return npz

    dataset = FIQTripleVisionDataset(
        str(args.dual_json),
        tokenizer,
        category=category,
        max_words=args.max_words,
        skip_empty=True,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    gallery_loader, n_g, _ = build_fiq_gallery_dataloader(args, split=args.split, category=category)
    print(
        f"encode {category}: gallery={n_g} queries={len(dataset)} "
        f"fgsa_levels={args.fgsa_levels}",
        flush=True,
    )
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    variants, refs, tgts, _groups = encode_three(model, loader, device)

    sim = {}
    for key in ("vision", "modified", "relative"):
        feats, cls, masks = variants[key]
        print(f"  FGSA {category}/{key} {feats.size(0)}x{gallery_feats.size(0)}", flush=True)
        sim[key] = _similarity_matrix(
            model, feats, cls, masks, gallery_feats, gallery_masks, device,
            args.query_batch, args.gallery_batch,
        )
    sim_ref = ref_gallery_cosine(gallery_feats.cpu(), gallery_masks.cpu(), index_names, refs)
    np.savez_compressed(
        npz,
        vision=sim["vision"],
        modified=sim["modified"],
        relative=sim["relative"],
        ref=sim_ref,
        index_names=np.array(index_names, dtype=object),
        refs=np.array(refs, dtype=object),
        tgts=np.array(tgts, dtype=object),
    )
    print(f"wrote {npz}", flush=True)
    del gallery_feats, gallery_masks, variants, sim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return npz


def fuse_all(packs: Dict[str, Dict]) -> Dict:
    singles = {
        "mic_only_wti": lambda d: d["s_modified"],
        "relative_only_wti": lambda d: d["s_relative"],
        "vision_only_wti": lambda d: d["s_vision"],
        "ref_only": lambda d: d["s_ref"],
        "rasd": lambda d: score_path(d["zd"], d["zm"], d["zr_ref"], FIQ_ALPHA, FIQ_BETA),
        "dasd": lambda d: score_path(d["zd"], d["zm"], d["zr_vis"], FIQ_ALPHA, FIQ_BETA),
        "dual_full": lambda d: (
            FIQ_WREF * zscore_rows(score_path(d["zd"], d["zm"], d["zr_ref"], FIQ_ALPHA, FIQ_BETA))
            + (1.0 - FIQ_WREF) * zscore_rows(score_path(d["zd"], d["zm"], d["zr_vis"], FIQ_ALPHA, FIQ_BETA))
        ).astype(np.float32),
    }
    out = {}
    for name, fn in singles.items():
        per = {c: _eval_sim(packs[c], fn(packs[c])) for c in CATS}
        out[name] = {"per": per, "mean": _mean_over(per)}
    return out


def parse_args():
    p = argparse.ArgumentParser(description="FashionIQ dual-path coarse retrieval")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="val prints recall. test has no local target labels.")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--out_dir", default=None)
    p.add_argument("--dual_json", default=None)
    p.add_argument("--fgsa_levels", type=int, default=1, choices=[1, 3])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--query_batch", type=int, default=16)
    p.add_argument("--gallery_batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--memory_fraction", type=float, default=0.40,
                   help="Cap this process to this fraction of the visible GPU.")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--encode_only", action="store_true")
    p.add_argument("--exp_id", default="A3_dual")
    p.add_argument("--exp_name", default="entity_only_dual")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.dual_json:
        args.dual_json = fiq_paths.vision_dual_json(args.split)
    if not args.out_dir:
        args.out_dir = str(DEFAULT_OUT / args.split)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = _find_best_ckpt(Path(args.ckpt))
    if not Path(args.dual_json).is_file():
        raise FileNotFoundError(args.dual_json)
    if not OPENCLIP_CKPT.is_file():
        raise FileNotFoundError(OPENCLIP_CKPT)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != "cpu" and args.memory_fraction > 0:
        idx = torch.device(device).index or 0
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), idx)
        print(
            f"memory_fraction={args.memory_fraction:.2f} on {device} "
            f"(leave headroom for CIRR A2)",
            flush=True,
        )

    ns = SimpleNamespace(
        base_encoder="ViT-L/14",
        clip_checkpoint=str(OPENCLIP_CKPT),
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
        fgsa_levels=args.fgsa_levels,
        max_frames=12,
        max_words=args.max_words,
        cross_num_hidden_layers=4,
        distributed=0,
        image_encode_microbatch=8,
        batch_size_val=args.batch_size_val,
        query_batch=args.query_batch,
        gallery_batch=args.gallery_batch,
        workers=args.workers,
        fiq_image_root=None,
        fiq_categories="dress,shirt,toptee",
        local_rank=0,
        image_resolution=224,
        gradient_checkpointing=1,
        split=args.split,
        dual_json=args.dual_json,
        recompute=args.recompute,
    )

    print(f"load A3 ckpt={ckpt} fgsa_levels={args.fgsa_levels} → {out}", flush=True)
    model = FGSAModel(ns)
    state = torch.load(str(ckpt), map_location="cpu")
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(cleaned, strict=False)
    print(f"load_state_dict miss={len(miss)} unexp={len(unexp)}", flush=True)
    model = model.to(device).eval()
    tokenizer = ClipTokenizer()

    npz_paths = []
    for cat in CATS:
        npz_paths.append(encode_category(model, ns, tokenizer, cat, device, out))

    if args.encode_only or args.split != "val":
        if args.split != "val":
            note = {
                "split": args.split,
                "dataset": "fashioniq",
                "recall": None,
                "note": "This split has no local target labels. Scores are the scores_*.npz files.",
            }
            (out / "metrics.json").write_text(json.dumps(note, indent=2) + "\n")
            print(f"split={args.split}: no local labels, wrote scores under {out}", flush=True)
        else:
            print("encode_only: skip fusion", flush=True)
        return

    packs = {cat: _pack_from_npz(out / f"scores_{cat}.npz") for cat in CATS}
    fused = fuse_all(packs)
    dual = fused["dual_full"]
    payload = {
        "id": args.exp_id,
        "name": args.exp_name,
        "dataset": "fiq",
        "split": args.split,
        "gallery": "original_split_large",
        "ckpt": str(ckpt),
        "fgsa_levels": args.fgsa_levels,
        "params": {"alpha": FIQ_ALPHA, "beta": FIQ_BETA, "w": FIQ_WREF},
        "note": (
            "Banzhaf-guided entity-only FGSA (A3) replaces 3-level BFSAR matching; "
            "dual RASD+DASD uses mainline α/β/w, not retuned."
        ),
        "mean": dual["mean"],
        "per": dual["per"],
        "branches": {
            k: {"mean": v["mean"], "per": v["per"]} for k, v in fused.items()
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def line(tag, m):
        print(
            f"  {tag:22s}  R@10={m['recall_at10']:6.2f}  R@50={m['recall_at50']:6.2f}  "
            f"R@100={m['recall_at100']:6.2f}  R@200={m['recall_at200']:6.2f}",
            flush=True,
        )

    print("\n=== FIQ Original-Split large gallery | A3 entity-only Banzhaf + dual ===", flush=True)
    print(f"ckpt={ckpt}", flush=True)
    print(f"α={FIQ_ALPHA} β={FIQ_BETA} w={FIQ_WREF} fgsa_levels={args.fgsa_levels}", flush=True)
    for cat in CATS:
        p = packs[cat]
        print(f"  {cat}: n_queries={p['n_queries']} n_gallery={p['n_gallery']}", flush=True)
    print("\nmean:", flush=True)
    for name in ("mic_only_wti", "relative_only_wti", "vision_only_wti", "ref_only",
                 "rasd", "dasd", "dual_full"):
        line(name, fused[name]["mean"])
    print("\nper-category dual:", flush=True)
    for cat in CATS:
        line(cat, dual["per"][cat])
    print(f"\nwrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
