#!/usr/bin/env python3
"""CIRR coarse retrieval: entity-only FGSA and dual-path scores.

RASD uses (alpha, beta) = (0.5, 0.2). DASD uses (0.2, 0.8). The path mix weight is 0.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr.openclip_backbone import install_openclip_cirr_patches

install_openclip_cirr_patches()

from bfsar.cirr import cirr_paths  # noqa: E402
from bfsar.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader  # noqa: E402
from bfsar.cirr.eval_fgsa import _similarity_matrix, encode_gallery  # noqa: E402
from bfsar.cirr.modeling_fgsa import FGSAModel  # noqa: E402
from bfsar.dpic.encode_cirr import (  # noqa: E402
    CIRRTripleVisionDataset,
    collate_fn,
    encode_three,
)
from bfsar.dpic.norm import zscore_rows  # noqa: E402
from bfsar.dpic.refsim import ref_gallery_cosine  # noqa: E402
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402
from bfsar.dpic.calibrate import (  # noqa: E402
    cirr_full_metrics,
    fuse_paths,
    score_path,
)

ALPHA_REF, BETA_REF = 0.5, 0.2
ALPHA_VIS, BETA_VIS = 0.2, 0.8
W_REF = 0.2
OPENCLIP_CKPT = ROOT / "weights/openclip/ViT-L-14-laion2B-s32B-b82K.pt"
DEFAULT_CKPT = ROOT / "weights/fgsa/cirr/pytorch_model.bin"
DEFAULT_OUT = ROOT / "output/dpic/cirr"


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


def _pack_from_npz(path: Path) -> Dict:
    d = np.load(path, allow_pickle=True)
    index_names = [str(x) for x in d["index_names"].tolist()]
    refs = [str(x) for x in d["refs"].tolist()]
    tgts = [str(x) for x in d["tgts"].tolist()]
    return {
        "zd": zscore_rows(d["sim_modified"]),
        "zm": zscore_rows(d["sim_relative"]),
        "zr_ref": zscore_rows(d["sim_ref"]),
        "zr_vis": zscore_rows(d["sim_vision"]),
        "s_modified": np.asarray(d["sim_modified"], dtype=np.float32),
        "s_relative": np.asarray(d["sim_relative"], dtype=np.float32),
        "s_vision": np.asarray(d["sim_vision"], dtype=np.float32),
        "s_ref": np.asarray(d["sim_ref"], dtype=np.float32),
        "index_names": index_names,
        "refs": refs,
        "tgts": tgts,
        "groups": [list(g) for g in d["groups"].tolist()],
        "pairids": [int(x) for x in d["pairids"].tolist()] if "pairids" in d.files else [],
    }


@torch.no_grad()
def encode_branches(model, args, tokenizer, device, out: Path) -> Path:
    npz = out / "branch_sims.npz"
    if npz.is_file() and not args.recompute:
        print(f"reuse {npz}", flush=True)
        return npz

    dataset = CIRRTripleVisionDataset(
        str(args.dual_json), tokenizer, max_words=args.max_words, max_samples=None,
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
    file_split = "test1" if args.split in ("test", "test1") else "val"
    gallery_loader, n_g, _ = build_cirr_gallery_dataloader(args, split=file_split)
    print(
        f"encode CIRR {args.split}: gallery={n_g} queries={len(dataset)} fgsa_levels={args.fgsa_levels}",
        flush=True,
    )
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    variants, refs, tgts, groups = encode_three(model, loader, device)

    sim = {}
    for key in ("vision", "modified", "relative"):
        feats, cls, masks = variants[key]
        print(f"  FGSA {key} {feats.size(0)}x{gallery_feats.size(0)}", flush=True)
        sim[key] = _similarity_matrix(
            model, feats, cls, masks, gallery_feats, gallery_masks, device,
            args.query_batch, args.gallery_batch,
        )
    sim_ref = ref_gallery_cosine(gallery_feats.cpu(), gallery_masks.cpu(), index_names, refs)
    np.savez_compressed(
        npz,
        sim_modified=sim["modified"].astype(np.float32),
        sim_relative=sim["relative"].astype(np.float32),
        sim_vision=sim["vision"].astype(np.float32),
        sim_ref=sim_ref.astype(np.float32),
        refs=np.array(refs, dtype=object),
        tgts=np.array(tgts, dtype=object),
        groups=np.array(groups, dtype=object),
        pairids=np.asarray([int(q["pairid"]) for q in dataset.queries], dtype=np.int64),
        index_names=np.array(index_names, dtype=object),
        dual_json=str(args.dual_json),
        init_model=str(args.ckpt),
        ref_sim_mode="pool_gram",
        fgsa_levels=int(args.fgsa_levels),
    )
    print(f"wrote {npz}", flush=True)
    del gallery_feats, gallery_masks, variants, sim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return npz


def coarse_matrix(pack: Dict) -> np.ndarray:
    """Gallery scores used by multi-view reranking: w·RASD + (1-w)·DASD."""
    rasd = score_path(pack["zd"], pack["zm"], pack["zr_ref"], ALPHA_REF, BETA_REF)
    dasd = score_path(pack["zd"], pack["zm"], pack["zr_vis"], ALPHA_VIS, BETA_VIS)
    return fuse_paths(rasd, dasd, W_REF)


def write_coarse_sim(pack: Dict, path: Path) -> None:
    dual = coarse_matrix(pack)
    np.savez_compressed(
        path,
        sim=np.asarray(dual, dtype=np.float32),
        refs=np.array(pack["refs"], dtype=object),
        tgts=np.array(pack["tgts"], dtype=object),
        index_names=np.array(pack["index_names"], dtype=object),
        pairids=np.asarray(pack.get("pairids") or [], dtype=np.int64),
    )


def fuse_all(pack: Dict) -> Dict:
    rasd = score_path(pack["zd"], pack["zm"], pack["zr_ref"], ALPHA_REF, BETA_REF)
    dasd = score_path(pack["zd"], pack["zm"], pack["zr_vis"], ALPHA_VIS, BETA_VIS)
    dual = fuse_paths(rasd, dasd, W_REF)
    return {
        "mic_only_wti": cirr_full_metrics(pack["s_modified"], pack),
        "relative_only_wti": cirr_full_metrics(pack["s_relative"], pack),
        "vision_only_wti": cirr_full_metrics(pack["s_vision"], pack),
        "ref_only": cirr_full_metrics(pack["s_ref"], pack),
        "rasd": cirr_full_metrics(rasd, pack),
        "dasd": cirr_full_metrics(dasd, pack),
        "dual_full": cirr_full_metrics(dual, pack),
    }


def parse_args():
    p = argparse.ArgumentParser(description="CIRR dual-path coarse retrieval")
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="val prints recall. test is CIRR test1 and has no local labels.")
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
    p.add_argument("--memory_fraction", type=float, default=0.90)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--encode_only", action="store_true")
    p.add_argument("--exp_id", default="A3_dual")
    p.add_argument("--exp_name", default="entity_only_dual")
    return p.parse_args()


def main():
    args = parse_args()
    file_split = "test1" if args.split == "test" else "val"
    if not args.dual_json:
        args.dual_json = cirr_paths.vision_dual_json(file_split)
    if not args.out_dir:
        args.out_dir = str(DEFAULT_OUT / args.split)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = _find_best_ckpt(Path(args.ckpt))
    args.ckpt = str(ckpt)
    if not Path(args.dual_json).is_file():
        raise FileNotFoundError(args.dual_json)
    if not OPENCLIP_CKPT.is_file():
        raise FileNotFoundError(OPENCLIP_CKPT)

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != "cpu" and args.memory_fraction > 0:
        idx = torch.device(device).index or 0
        torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), idx)
        print(f"memory_fraction={args.memory_fraction:.2f} on {device}", flush=True)

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
        cirr_image_root=None,
        local_rank=0,
        image_resolution=224,
        gradient_checkpointing=1,
        split=args.split,
        dual_json=args.dual_json,
        recompute=args.recompute,
        ckpt=str(ckpt),
        fgsa_levels_eval=args.fgsa_levels,
    )

    print(f"load ckpt={ckpt} fgsa_levels={args.fgsa_levels} → {out}", flush=True)
    model = FGSAModel(ns)
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    miss, unexp = model.load_state_dict(cleaned, strict=False)
    print(f"load_state_dict miss={len(miss)} unexp={len(unexp)}", flush=True)
    model = model.to(device).eval()
    tokenizer = ClipTokenizer()

    encode_branches(model, ns, tokenizer, device, out)
    if args.encode_only:
        print("encode_only: skip fusion", flush=True)
        return

    pack = _pack_from_npz(out / "branch_sims.npz")
    coarse_path = out / "coarse_sim.npz"
    write_coarse_sim(pack, coarse_path)
    print(f"wrote {coarse_path}", flush=True)
    labeled = all(t and t in set(pack["index_names"]) for t in pack["tgts"])
    if not labeled:
        note = {
            "split": args.split,
            "dataset": "cirr",
            "recall": None,
            "note": "This split has no local target labels. Scores are in branch_sims.npz.",
        }
        (out / "metrics.json").write_text(json.dumps(note, indent=2) + "\n")
        print(f"split={args.split}: no local labels, wrote scores to {out / 'branch_sims.npz'}", flush=True)
        return

    fused = fuse_all(pack)
    dual = fused["dual_full"]
    payload = {
        "id": args.exp_id,
        "name": args.exp_name,
        "dataset": "cirr",
        "split": args.split,
        "ckpt": str(ckpt),
        "fgsa_levels": args.fgsa_levels,
        "params": {
            "alpha_ref": ALPHA_REF,
            "beta_ref": BETA_REF,
            "alpha_vis": ALPHA_VIS,
            "beta_vis": BETA_VIS,
            "w": W_REF,
        },
        "note": (
            "entity-only FGSA replaces 3-level BFSAR matching; dual RASD+DASD "
            "uses mainline CIRR A7 α/β/w, not retuned."
        ),
        "metrics": dual,
        "branches": fused,
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def line(tag, m):
        print(
            f"  {tag:22s}  R@1={m['recall_at1']:6.2f}  R@5={m['recall_at5']:6.2f}  "
            f"R@10={m['recall_at10']:6.2f}  R@50={m['recall_at50']:6.2f}  "
            f"G-R@1={m['group_recall_at1']:6.2f}",
            flush=True,
        )

    print(f"\n=== CIRR {args.split} | dual-path coarse ===", flush=True)
    print(f"ckpt={ckpt}", flush=True)
    print(
        f"α_ref={ALPHA_REF} β_ref={BETA_REF} α_vis={ALPHA_VIS} β_vis={BETA_VIS} "
        f"w={W_REF} fgsa_levels={args.fgsa_levels}",
        flush=True,
    )
    print(f"  n_queries={len(pack['refs'])} n_gallery={len(pack['index_names'])}", flush=True)
    for name in (
        "mic_only_wti", "relative_only_wti", "vision_only_wti", "ref_only",
        "rasd", "dasd", "dual_full",
    ):
        line(name, fused[name])
    print(f"\nwrote {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
