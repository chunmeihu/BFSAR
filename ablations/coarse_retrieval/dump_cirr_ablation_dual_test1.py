#!/usr/bin/env python3
"""Dump CIRR test1 ablation dual sim + Top-K for Composer/CR (frozen A7 α/β/w)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir
from HBI.cirr.vision_desc.dump_cirr_openclip_dual_test1 import DEFAULT_PARAMS  # noqa: E402
from HBI.cirr.vision_desc.hbi_blip_rerank import build_test1_topk_pt, save_topk_pt  # noqa: E402
from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import zscore_rows  # noqa: E402
from HBI.xr_desc.eval_sdr_dual_sr_ref_vision import fuse_w, sdr_wti  # noqa: E402

logger = logging.getLogger("dump_cirr_ablation_test1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--branch_npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_save", type=int, default=200)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    branch = Path(args.branch_npz)
    if not branch.is_file():
        raise SystemExit(f"ABORT: missing {branch}")
    out = ensure_ablation_out_dir(args.out_dir)
    params = dict(DEFAULT_PARAMS)
    d = np.load(branch, allow_pickle=True)
    for key in ("sim_modified", "sim_relative", "sim_vision", "sim_ref", "pairids", "index_names"):
        if key not in d.files:
            raise KeyError(f"missing {key} in {branch}")
    zd = zscore_rows(d["sim_modified"])
    zm = zscore_rows(d["sim_relative"])
    fused = fuse_w(
        sdr_wti(zd, zm, zscore_rows(d["sim_ref"]), params["alpha_ref"], params["beta_ref"]),
        sdr_wti(zd, zm, zscore_rows(d["sim_vision"]), params["alpha_vis"], params["beta_vis"]),
        params["w_ref"],
    ).astype(np.float32)
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
        branch_npz=str(branch),
        source="ablation_dual_frozen_a7_test1",
        split="test1",
        tag=args.tag,
    )
    payload = build_test1_topk_pt(fused, index_names, k_save=args.k_save)
    topk_pt = out / f"cirr_top_{args.k_save}_test1.pt"
    save_topk_pt(payload, topk_pt)
    meta = {
        "exp": "cirr_ablation_dual_test1_dump",
        "tag": args.tag,
        "params": params,
        "branch_npz": str(branch),
        "sim_npz": str(sim_path),
        "topk_pt": str(topk_pt),
        "k_save": args.k_save,
        "n_queries": len(pairids),
        "n_gallery": len(index_names),
    }
    (out / "dump_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
