#!/usr/bin/env python3
"""Materialize CIRR test1 dual-path coarse sim + Top-K .pt (frozen val indep-best).

Default params match val OpenCLIP dual indep best:
  α_ref=0.5 β_ref=0.2 / α_vis=0.2 β_vis=0.8 / w_ref=0.2
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

from HBI.cirr.vision_desc.hbi_blip_rerank import build_test1_topk_pt, save_topk_pt  # noqa: E402
from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import zscore_rows  # noqa: E402
from HBI.xr_desc.eval_sdr_dual_sr_ref_vision import fuse_w, sdr_wti  # noqa: E402

logger = logging.getLogger("dump_cirr_openclip_dual_test1")

DEFAULT_BRANCH = str(ROOT / "output/cirr_vision_desc/sdr_wti_native_openclip_test1/branch_sims.npz")
DEFAULT_OUT = str(ROOT / "output/cirr_rerank/fusion_openclip_dual_k200_test1")
DEFAULT_PARAMS = {
    "alpha_ref": 0.5,
    "beta_ref": 0.2,
    "alpha_vis": 0.2,
    "beta_vis": 0.8,
    "w_ref": 0.2,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--branch_npz", type=str, default=DEFAULT_BRANCH)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--k_save", type=int, default=200)
    p.add_argument("--alpha_ref", type=float, default=DEFAULT_PARAMS["alpha_ref"])
    p.add_argument("--beta_ref", type=float, default=DEFAULT_PARAMS["beta_ref"])
    p.add_argument("--alpha_vis", type=float, default=DEFAULT_PARAMS["alpha_vis"])
    p.add_argument("--beta_vis", type=float, default=DEFAULT_PARAMS["beta_vis"])
    p.add_argument("--w_ref", type=float, default=DEFAULT_PARAMS["w_ref"])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out = Path(args.out_dir)
    if "openclip" not in str(out).lower() or "dual" not in str(out).lower():
        raise SystemExit(f"out_dir must isolate openclip dual: {out}")
    out.mkdir(parents=True, exist_ok=True)

    params = {
        "alpha_ref": args.alpha_ref,
        "beta_ref": args.beta_ref,
        "alpha_vis": args.alpha_vis,
        "beta_vis": args.beta_vis,
        "w_ref": args.w_ref,
    }
    d = np.load(args.branch_npz, allow_pickle=True)
    for key in ("sim_modified", "sim_relative", "sim_vision", "sim_ref", "pairids", "index_names"):
        if key not in d.files:
            raise KeyError(f"branch npz missing {key}: {args.branch_npz}")
    if str(d.get("ref_sim_mode", "")) != "hbi_pool_gram":
        logger.warning("ref_sim_mode=%s (expected hbi_pool_gram)", d.get("ref_sim_mode", ""))

    zd = zscore_rows(d["sim_modified"])
    zm = zscore_rows(d["sim_relative"])
    zr_ref = zscore_rows(d["sim_ref"])
    zr_vis = zscore_rows(d["sim_vision"])
    s_ref = sdr_wti(zd, zm, zr_ref, params["alpha_ref"], params["beta_ref"])
    s_vis = sdr_wti(zd, zm, zr_vis, params["alpha_vis"], params["beta_vis"])
    fused = fuse_w(s_ref, s_vis, params["w_ref"]).astype(np.float32)

    pairids = [int(x) for x in d["pairids"].tolist()]
    index_names = [str(x) for x in d["index_names"].tolist()]
    refs = [str(x) for x in d["refs"].tolist()] if "refs" in d.files else [""] * len(pairids)

    sim_path = out / "sim_hbi.npz"
    np.savez_compressed(
        sim_path,
        sim_hbi=fused,
        pairids=np.array(pairids, dtype=np.int64),
        refs=np.array(refs, dtype=object),
        index_names=np.array(index_names, dtype=object),
        dual_params=json.dumps(params),
        init_model=str(d["init_model"]) if "init_model" in d.files else "",
        dual_json=str(d["dual_json"]) if "dual_json" in d.files else "",
        ref_sim_mode=str(d["ref_sim_mode"]) if "ref_sim_mode" in d.files else "",
        branch_npz=str(args.branch_npz),
        source="openclip_dual_indep_best_test1",
        split="test1",
    )
    logger.info("Wrote %s shape=%s", sim_path, fused.shape)

    payload = build_test1_topk_pt(fused, index_names, k_save=args.k_save)
    topk_pt = out / f"cirr_top_{args.k_save}_test1.pt"
    save_topk_pt(payload, topk_pt)
    logger.info("Wrote %s shape=%s", topk_pt, payload["sorted_index_names"].shape)

    meta = {
        "exp": "cirr_openclip_dual_test1_dump",
        "params": params,
        "branch_npz": str(args.branch_npz),
        "sim_npz": str(sim_path),
        "topk_pt": str(topk_pt),
        "k_save": args.k_save,
        "n_queries": len(pairids),
        "n_gallery": len(index_names),
        "weights_source": "val OpenCLIP dual indep-best (not retuned on test1)",
    }
    (out / "dump_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
