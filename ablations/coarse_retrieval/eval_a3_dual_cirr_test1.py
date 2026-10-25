#!/usr/bin/env python3
"""CIRR test1: entity-only WTI encode + MIC / dual submission JSONs.

No public labels. Writes recall + recall_subset under output/ablations/cirr_coarse_test1/.
wti_levels=1. Dual α/β/w frozen to mainline A7 (not retuned).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir
from HBI.cirr.openclip_backbone import install_openclip_cirr_patches

install_openclip_cirr_patches()

from HBI.cirr.cirr_dataloader_builders import build_cirr_gallery_dataloader  # noqa: E402
from HBI.cirr.eval_cirr_hbi import _similarity_matrix, encode_gallery  # noqa: E402
from HBI.cirr.modeling_hbi_cirr import HBICIRR  # noqa: E402
from HBI.cirr.vision_desc.cirr_triple_test1 import CIRRTripleTest1Dataset, load_test1_triplets  # noqa: E402
from HBI.cirr.vision_desc.eval_hbi_triple_vision import collate_fn, encode_three  # noqa: E402
from HBI.cirr.vision_desc.hbi_blip_rerank import (  # noqa: E402
    align_gallery_index_to_cirr,
    generate_hbi_coarse_test_submission_files,
)
from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import zscore_rows  # noqa: E402
from HBI.fiq.vision_desc.eval_fiq_hbi_refsim_fusion import ref_gallery_cosine  # noqa: E402
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402
from HBI.xr_desc.eval_sdr_dual_sr_ref_vision import fuse_w, sdr_wti  # noqa: E402

ALPHA_REF, BETA_REF = 0.5, 0.2
ALPHA_VIS, BETA_VIS = 0.2, 0.8
W_REF = 0.2
OPENCLIP_CKPT = ROOT / "weights/open_clip/hbi_format/ViT-L-14-laion2B-s32B-b82K.pt"
DEFAULT_DUAL = ROOT / "data/cirr/vision_desc/test1_dual_targets.json"
DEFAULT_COARSE_ROOT = ROOT / "output/ablations/cirr_coarse_test1"


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


def _zip_pair(out_dir: Path, name: str, recall: Path, subset: Path) -> Path:
    zpath = out_dir / f"{name}_cirr_test1.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(recall, arcname=recall.name)
        zf.write(subset, arcname=subset.name)
    return zpath


def _write_sub(out_dir: Path, name: str, sim: np.ndarray, index_names, meta: dict) -> dict:
    ensure_ablation_out_dir(out_dir)
    paths = generate_hbi_coarse_test_submission_files(sim, index_names, name, out_dir)
    zpath = _zip_pair(out_dir, name, paths["recall"], paths["recall_subset"])
    payload = {
        **meta,
        "split": "test1",
        "submission_name": name,
        "recall_json": str(paths["recall"]),
        "recall_subset_json": str(paths["recall_subset"]),
        "zip": str(zpath),
        "n_queries": int(sim.shape[0]),
        "n_gallery": int(sim.shape[1]),
    }
    (out_dir / "submission_meta.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[OK] {name} zip={zpath}", flush=True)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dual_json", default=str(DEFAULT_DUAL))
    p.add_argument("--wti_levels", type=int, default=1, choices=[1, 3])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--query_batch", type=int, default=16)
    p.add_argument("--gallery_batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--memory_fraction", type=float, default=0.90)
    p.add_argument("--coarse_root", default=str(DEFAULT_COARSE_ROOT))
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--encode_only", action="store_true")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out = ensure_ablation_out_dir(args.out_dir)
    ckpt = _find_best_ckpt(Path(args.ckpt))
    npz = out / "branch_sims.npz"
    dual_path = Path(args.dual_json)
    if not dual_path.is_file():
        raise FileNotFoundError(dual_path)

    if npz.is_file() and not args.recompute:
        print(f"reuse {npz}", flush=True)
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            raise RuntimeError("CUDA required for CIRR test1 encode")
        if args.memory_fraction > 0:
            idx = torch.device(device).index or 0
            torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), idx)

        ns = SimpleNamespace(
            base_encoder="ViT-L/14",
            clip_checkpoint=str(OPENCLIP_CKPT),
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
            wti_levels=args.wti_levels,
            cross_num_hidden_layers=4,
            max_frames=12,
            max_words=args.max_words,
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
        )
        print(f"encode test1 ckpt={ckpt} wti_levels={args.wti_levels} → {npz}", flush=True)
        model = HBICIRR(ns)
        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        miss, unexp = model.load_state_dict(cleaned, strict=False)
        print(f"load_state_dict miss={len(miss)} unexp={len(unexp)}", flush=True)
        model = model.to(device).eval()

        n_cap = len(load_test1_triplets())
        dataset = CIRRTripleTest1Dataset(
            dual_json=str(dual_path),
            tokenizer=ClipTokenizer(),
            max_words=args.max_words,
        )
        if len(dataset.queries) != n_cap:
            raise RuntimeError(f"query count {len(dataset.queries)} != {n_cap}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        gallery_loader, n_g, _ = build_cirr_gallery_dataloader(ns, split="test1")
        print(f"gallery={n_g} queries={len(dataset.queries)}", flush=True)
        gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
        variants, refs, _tgts, groups = encode_three(model, loader, device)
        sim = {}
        for key in ("vision", "modified", "relative"):
            feats, cls, masks = variants[key]
            print(f"  WTI {key}", flush=True)
            sim[key] = _similarity_matrix(
                model, feats, cls, masks, gallery_feats, gallery_masks, device,
                args.query_batch, args.gallery_batch,
            )
        sim_ref = ref_gallery_cosine(gallery_feats.cpu(), gallery_masks.cpu(), index_names, refs)
        orig = list(index_names)
        for key in sim:
            sim[key], _ = align_gallery_index_to_cirr(sim[key], orig, split="test1")
        sim_ref, index_names = align_gallery_index_to_cirr(sim_ref, orig, split="test1")
        pairids = [q["pairid"] for q in dataset.queries]
        np.savez_compressed(
            npz,
            sim_modified=sim["modified"].astype(np.float32),
            sim_relative=sim["relative"].astype(np.float32),
            sim_vision=sim["vision"].astype(np.float32),
            sim_ref=sim_ref.astype(np.float32),
            refs=np.array(refs, dtype=object),
            pairids=np.array(pairids, dtype=np.int64),
            groups=np.array(groups, dtype=object),
            index_names=np.array(index_names, dtype=object),
            dual_json=str(dual_path),
            init_model=str(ckpt),
            ref_sim_mode="hbi_pool_gram",
            split="test1",
            wti_levels=int(args.wti_levels),
        )
        print(f"wrote {npz}", flush=True)
        del gallery_feats, gallery_masks, variants, sim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.encode_only:
        return

    d = np.load(npz, allow_pickle=True)
    index_names = [str(x) for x in d["index_names"].tolist()]
    zd = zscore_rows(d["sim_modified"])
    zm = zscore_rows(d["sim_relative"])
    zr_ref = zscore_rows(d["sim_ref"])
    zr_vis = zscore_rows(d["sim_vision"])
    dual = fuse_w(
        sdr_wti(zd, zm, zr_ref, ALPHA_REF, BETA_REF),
        sdr_wti(zd, zm, zr_vis, ALPHA_VIS, BETA_VIS),
        W_REF,
    ).astype(np.float32)
    tag = args.tag
    coarse_root = ensure_ablation_out_dir(args.coarse_root)
    mic_meta = _write_sub(
        coarse_root / f"{tag}_mic_only_wti",
        f"ablation_{tag}_mic_only_wti",
        d["sim_modified"].astype(np.float32),
        index_names,
        {"id": tag, "branch": "sim_modified", "ckpt": str(d["init_model"])},
    )
    dual_meta = _write_sub(
        coarse_root / f"{tag}_dual_full",
        f"ablation_{tag}_dual_full",
        dual,
        index_names,
        {
            "id": tag,
            "alpha_ref": ALPHA_REF,
            "beta_ref": BETA_REF,
            "alpha_vis": ALPHA_VIS,
            "beta_vis": BETA_VIS,
            "w_ref": W_REF,
            "ckpt": str(d["init_model"]),
        },
    )
    index = {
        "tag": tag,
        "split": "test1",
        "ckpt": str(ckpt),
        "encode_dir": str(out),
        "branch_npz": str(npz),
        "mic": mic_meta,
        "dual": dual_meta,
        "note": "test1 labels hidden; upload the two zips to CodaLab.",
    }
    (out / "submission_index.json").write_text(json.dumps(index, indent=2) + "\n")
    (coarse_root / f"{tag}_submission_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {out / 'submission_index.json'}", flush=True)


if __name__ == "__main__":
    main()
