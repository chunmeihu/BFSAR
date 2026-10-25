#!/usr/bin/env python3
"""Dump CIRR OpenCLIP dual-path SDR-HBI coarse sim + Top-K .pt for Composer/CR.

Uses indep-αβ best from sdr_dual_sr_ref_vision_openclip_val (R@1≈46.49):
  Path_ref: α=0.5, β=0.2   (S_r = ref_gallery_cosine)
  Path_vis: α=0.2, β=0.8   (S_r = vision_side WTI)
  Fuse:     w_ref=0.2 · z(path_ref) + 0.8 · z(path_vis)

CPU-only. Writes sim_hbi.npz (export/fusion compatible) + cirr_top_{K}_val.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.cirr.vision_desc.hbi_blip_rerank import (  # noqa: E402
    align_gallery_index_to_cirr,
    build_full_val_topk_pt,
    save_topk_pt,
)
from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import zscore_rows  # noqa: E402
from HBI.xr_desc.eval_sdr_dual_sr_ref_vision import fuse_w, sdr_wti  # noqa: E402

logger = logging.getLogger("dump_cirr_openclip_dual")

DEFAULT_BRANCH = str(ROOT / "output/cirr_vision_desc/sdr_wti_native_openclip_val/branch_sims.npz")
DEFAULT_METRICS = str(ROOT / "output/cirr_vision_desc/sdr_dual_sr_ref_vision_openclip_val/metrics.json")
DEFAULT_OUT = str(ROOT / "output/cirr_rerank/fusion_openclip_dual_k200_val")

# indep best (fallback if metrics.json missing)
DEFAULT_PARAMS = {
    "alpha_ref": 0.5,
    "beta_ref": 0.2,
    "alpha_vis": 0.2,
    "beta_vis": 0.8,
    "w_ref": 0.2,
}


def load_indep_params(metrics_json: Path) -> dict:
    if not metrics_json.is_file():
        logger.warning("missing %s; using hardcoded indep best", metrics_json)
        return dict(DEFAULT_PARAMS)
    d = json.loads(metrics_json.read_text())
    row = d["indep_ab_best"]
    return {k: float(row[k]) for k in DEFAULT_PARAMS}


def dual_sim_from_branch(npz_path: Path, params: dict) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    zd = zscore_rows(d["sim_modified"])
    zm = zscore_rows(d["sim_relative"])
    zr_ref = zscore_rows(d["sim_ref"])
    zr_vis = zscore_rows(d["sim_vision"])
    s_ref = sdr_wti(zd, zm, zr_ref, params["alpha_ref"], params["beta_ref"])
    s_vis = sdr_wti(zd, zm, zr_vis, params["alpha_vis"], params["beta_vis"])
    fused = fuse_w(s_ref, s_vis, params["w_ref"]).astype(np.float32)
    return {
        "sim_hbi": fused,
        "refs": [str(x) for x in d["refs"].tolist()],
        "tgts": [str(x) for x in d["tgts"].tolist()],
        "index_names": [str(x) for x in d["index_names"].tolist()],
        "init_model": str(d["init_model"]) if "init_model" in d.files else "",
        "dual_json": str(d["dual_json"]) if "dual_json" in d.files else "",
        "ref_sim_mode": str(d["ref_sim_mode"]) if "ref_sim_mode" in d.files else "",
    }


def coarse_metrics(sim: np.ndarray, refs, tgts, index_names) -> dict:
    name = {n: i for i, n in enumerate(index_names)}
    ref_i = np.array([name[r] for r in refs], dtype=np.int32)
    tgt_i = np.array([name[t] for t in tgts], dtype=np.int32)
    s = np.array(sim, dtype=np.float64, copy=True)
    rows = np.arange(len(refs))
    s[rows, ref_i] = -1e9
    rk = (s > s[rows, tgt_i][:, None]).sum(axis=1)
    out = {}
    for k in (1, 5, 10, 50, 100, 200):
        out[f"recall_at{k}"] = 100.0 * float(np.mean(rk < k))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--branch_npz", type=str, default=DEFAULT_BRANCH)
    p.add_argument("--metrics_json", type=str, default=DEFAULT_METRICS)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--k_save", type=int, default=200)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out = Path(args.out_dir)
    if "openclip" not in str(out).lower() and "dual" not in str(out).lower():
        raise SystemExit(f"out_dir should be isolated openclip dual path, got {out}")
    out.mkdir(parents=True, exist_ok=True)

    params = load_indep_params(Path(args.metrics_json))
    logger.info("dual params %s", params)
    pack = dual_sim_from_branch(Path(args.branch_npz), params)
    m = coarse_metrics(pack["sim_hbi"], pack["refs"], pack["tgts"], pack["index_names"])
    logger.info(
        "coarse (dual subset n=%d) R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f",
        len(pack["refs"]), m["recall_at1"], m["recall_at5"], m["recall_at10"], m["recall_at50"],
    )

    sim_path = out / "sim_hbi.npz"
    np.savez_compressed(
        sim_path,
        sim_hbi=pack["sim_hbi"],
        refs=np.array(pack["refs"], dtype=object),
        tgts=np.array(pack["tgts"], dtype=object),
        index_names=np.array(pack["index_names"], dtype=object),
        dual_params=json.dumps(params),
        init_model=pack["init_model"],
        dual_json=pack["dual_json"],
        ref_sim_mode=pack["ref_sim_mode"],
        branch_npz=str(args.branch_npz),
        source="openclip_dual_indep_best",
    )
    logger.info("Wrote %s", sim_path)

    sim, index_names = align_gallery_index_to_cirr(pack["sim_hbi"], pack["index_names"])
    payload = build_full_val_topk_pt(
        sim, pack["refs"], pack["tgts"], index_names, k_save=args.k_save,
    )
    topk_pt = out / f"cirr_top_{args.k_save}_val.pt"
    save_topk_pt(payload, topk_pt)
    hit = float(payload["labels"].any(axis=1).mean() * 100)
    n_miss = len(payload.get("missing_hbi_keys") or [])
    logger.info(
        "Wrote %s | target-in-top-%d=%.2f%% | missing_hbi_keys=%d (alpha fallback)",
        topk_pt, args.k_save, hit, n_miss,
    )

    meta = {
        "exp": "cirr_openclip_dual_coarse_dump",
        "params": params,
        "branch_npz": str(args.branch_npz),
        "sim_npz": str(sim_path),
        "topk_pt": str(topk_pt),
        "k_save": args.k_save,
        "n_dual_queries": len(pack["refs"]),
        "n_val_triplets": int(payload["labels"].shape[0]),
        "missing_hbi_keys": n_miss,
        "target_in_topk_pct": hit,
        "coarse_dual_subset": m,
        "init_model": pack["init_model"],
        "ref_sim_mode": pack["ref_sim_mode"],
    }
    (out / "dump_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
