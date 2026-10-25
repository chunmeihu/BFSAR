#!/usr/bin/env python3
"""CIRR test1: OpenCLIP HBI branch sims for dual-path SDR coarse.

Writes modified / relative / vision WTI + FIQ-aligned ref_gallery_cosine S_r.
Must run under GPU with OpenCLIP patches installed (via wrapper or --openclip).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("cirr_openclip_test1_branches")

DEFAULT_CKPT = str(ROOT / "output/cirr_retrieval_openclip/pytorch_model.bin.best.9")
DEFAULT_DUAL = str(ROOT / "data/cirr/vision_desc/test1_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output/cirr_vision_desc/sdr_wti_native_openclip_test1/branch_sims.npz")
DEFAULT_CLIP = str(ROOT / "weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DUAL)
    p.add_argument("--output_npz", type=str, default=DEFAULT_OUT)
    p.add_argument("--clip_checkpoint", type=str, default=DEFAULT_CLIP)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--recompute", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out = Path(args.output_npz)
    if "openclip" not in str(out).lower():
        raise SystemExit(f"output_npz must contain openclip: {out}")
    if out.is_file() and not args.recompute:
        logger.info("Cache exists: %s", out)
        return

    from HBI.cirr.openclip_backbone import install_openclip_cirr_patches

    install_openclip_cirr_patches()

    from HBI.cirr import cirr_paths
    from HBI.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader
    from HBI.cirr.eval_cirr_hbi import encode_gallery, _similarity_matrix
    from HBI.cirr.modeling_hbi_cirr import HBICIRR
    from HBI.cirr.vision_desc.cirr_triple_test1 import CIRRTripleTest1Dataset, load_test1_triplets
    from HBI.cirr.vision_desc.eval_hbi_triple_vision import collate_fn, encode_three
    from HBI.cirr.vision_desc.hbi_blip_rerank import align_gallery_index_to_cirr
    from HBI.fiq.vision_desc.eval_fiq_hbi_refsim_fusion import ref_gallery_cosine
    from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

    dual_path = Path(args.dual_json)
    if not dual_path.is_file():
        raise FileNotFoundError(dual_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for test1 OpenCLIP branch encode")

    ns = SimpleNamespace(
        device=device,
        init_model=args.init_model,
        dual_json=str(dual_path),
        max_samples=args.max_samples,
        max_words=args.max_words,
        batch_size_val=args.batch_size_val,
        workers=args.workers,
        query_batch=args.query_batch,
        gallery_batch=args.gallery_batch,
        base_encoder="ViT-L/14",
        clip_checkpoint=args.clip_checkpoint,
        agg_module="None",
        interaction="wti",
        num_hidden_layers=4,
        estimator="None",
        max_frames=12,
        image_resolution=224,
        gradient_checkpointing=1,
        image_encode_microbatch=12,
        v_rate0=0.25,
        v_rate1=0.5,
        t_rate0=0.25,
        t_rate1=0.5,
        rate=[1.0, 1.0, 1.0],
        kl=2.0,
        skl=1.0,
        cirr_image_root=None,
    )

    n_cap = len(load_test1_triplets())
    dataset = CIRRTripleTest1Dataset(
        dual_json=str(dual_path),
        tokenizer=ClipTokenizer(),
        max_words=args.max_words,
        max_samples=args.max_samples,
    )
    if len(dataset.queries) != n_cap:
        raise RuntimeError(
            f"test1 query count {len(dataset.queries)} != caption triplets {n_cap}; "
            "submission row order would break"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = HBICIRR(ns)
    state = torch.load(args.init_model, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded %s missing=%d unexpected=%d", args.init_model, len(missing), len(unexpected))
    model = model.to(device).eval()

    gallery_loader, _, _ = build_cirr_gallery_dataloader(ns, split="test1")
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    variants, refs, _tgts, groups = encode_three(model, loader, device)

    sim = {}
    for key in ("vision", "modified", "relative"):
        feats, cls, masks = variants[key]
        logger.info("WTI %s ...", key)
        sim[key] = _similarity_matrix(
            model, feats, cls, masks, gallery_feats, gallery_masks, device,
            args.query_batch, args.gallery_batch,
        )

    logger.info("S_r = ref_gallery_cosine (FIQ-aligned) ...")
    sim_ref = ref_gallery_cosine(gallery_feats.cpu(), gallery_masks.cpu(), index_names, refs)
    logger.info("S_r shape=%s mean=%.4f", sim_ref.shape, float(sim_ref.mean()))

    orig_names = list(index_names)
    for key in sim:
        sim[key], _ = align_gallery_index_to_cirr(sim[key], orig_names, split="test1")
    sim_ref, index_names = align_gallery_index_to_cirr(sim_ref, orig_names, split="test1")

    pairids = [q["pairid"] for q in dataset.queries]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sim_modified=sim["modified"].astype(np.float32),
        sim_relative=sim["relative"].astype(np.float32),
        sim_vision=sim["vision"].astype(np.float32),
        sim_ref=sim_ref.astype(np.float32),
        refs=np.array(refs, dtype=object),
        pairids=np.array(pairids, dtype=np.int64),
        groups=np.array(groups, dtype=object),
        index_names=np.array(index_names, dtype=object),
        dual_json=str(dual_path),
        init_model=str(args.init_model),
        ref_sim_mode="hbi_pool_gram",
        split="test1",
    )
    logger.info("Wrote %s n=%d gallery=%d", out, len(pairids), len(index_names))


if __name__ == "__main__":
    main()
