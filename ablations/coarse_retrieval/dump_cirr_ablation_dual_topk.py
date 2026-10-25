#!/usr/bin/env python3
"""Dump CIRR ablation dual-path coarse sim + Top-K for Composer/CR.

Reads output/ablations/cirr_coarse/*_dual/branch_sims.npz, fuses with frozen
mainline A7 α/β/w (not retuned), writes sim_hbi.npz + cirr_top_{K}_val.pt
under output/ablations/cirr_rerank/.
"""

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
from HBI.cirr.vision_desc.dump_cirr_openclip_dual_sim import (  # noqa: E402
    DEFAULT_PARAMS,
    coarse_metrics,
    dual_sim_from_branch,
)
from HBI.cirr.vision_desc.hbi_blip_rerank import (  # noqa: E402
    align_gallery_index_to_cirr,
    build_full_val_topk_pt,
    save_topk_pt,
)

logger = logging.getLogger("dump_cirr_ablation_dual")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--branch_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--k_save", type=int, default=200)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    branch = Path(args.branch_npz)
    if not branch.is_file():
        raise SystemExit(f"ABORT: missing {branch}")
    out = ensure_ablation_out_dir(args.out_dir)
    params = dict(DEFAULT_PARAMS)
    logger.info("tag=%s dual params %s", args.tag, params)

    pack = dual_sim_from_branch(branch, params)
    m = coarse_metrics(pack["sim_hbi"], pack["refs"], pack["tgts"], pack["index_names"])
    logger.info(
        "coarse dual n=%d R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f G skipped here",
        len(pack["refs"]),
        m["recall_at1"],
        m["recall_at5"],
        m["recall_at10"],
        m["recall_at50"],
    )

    sim_path = out / "sim_hbi.npz"
    np.savez_compressed(
        sim_path,
        sim_hbi=pack["sim_hbi"],
        refs=np.array(pack["refs"], dtype=object),
        tgts=np.array(pack["tgts"], dtype=object),
        index_names=np.array(pack["index_names"], dtype=object),
        dual_params=json.dumps(params),
        init_model=pack.get("init_model", ""),
        dual_json=pack.get("dual_json", ""),
        ref_sim_mode=pack.get("ref_sim_mode", ""),
        branch_npz=str(branch),
        source="ablation_dual_frozen_a7",
        tag=args.tag,
    )

    sim, index_names = align_gallery_index_to_cirr(pack["sim_hbi"], pack["index_names"])
    payload = build_full_val_topk_pt(
        sim, pack["refs"], pack["tgts"], index_names, k_save=args.k_save,
    )
    topk_pt = out / f"cirr_top_{args.k_save}_val.pt"
    save_topk_pt(payload, topk_pt)
    hit = float(payload["labels"].any(axis=1).mean() * 100)
    n_miss = len(payload.get("missing_hbi_keys") or [])
    meta = {
        "exp": "cirr_ablation_dual_coarse_dump",
        "tag": args.tag,
        "params": params,
        "branch_npz": str(branch),
        "sim_npz": str(sim_path),
        "topk_pt": str(topk_pt),
        "k_save": args.k_save,
        "n_dual_queries": len(pack["refs"]),
        "n_val_triplets": int(payload["labels"].shape[0]),
        "missing_hbi_keys": n_miss,
        "target_in_topk_pct": hit,
        "coarse_dual_subset": m,
    }
    (out / "dump_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
