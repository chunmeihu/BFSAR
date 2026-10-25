#!/usr/bin/env python3
"""CIRR ablation rerank table: coarse dual + frozen mainline 3-way fuse.

Headline fusion is the mainline CIRR val recipe α=0.2, β=0.35, γ=0.45
(zscore-per-query), not retuned per ablation. Grid-best is recorded only as a
reference and must not be used for A2e vs A3 vs A3t comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir
from HBI.cirr.vision_desc.eval_hbi_composer_cr_fusion import (  # noqa: E402
    compute_hbi,
    pool_from_scores,
    zscore_rows,
)

# Mainline CIRR OpenCLIP dual + Composer + CR (val K=200) zscore best.
FROZEN = {"alpha_hbi": 0.2, "beta_composer": 0.35, "gamma_cr": 0.45}


def _pool_keys(m: dict) -> dict:
    out = {}
    for k in (
        "pool_recall_at1",
        "pool_recall_at5",
        "pool_recall_at10",
        "pool_recall_at50",
        "pool_group_recall_at1",
        "pool_group_recall_at2",
        "pool_group_recall_at3",
    ):
        if k in m:
            out[k.replace("pool_", "")] = float(m[k])
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dual_metrics", default="")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_ablation_out_dir(args.out_dir)
    composer = np.load(out / "composer_itc_scores.npz", allow_pickle=True)
    cr = np.load(out / "cr_nlvr_scores.npz", allow_pickle=True)
    target_names = [str(x) for x in composer["target_names"].tolist()]
    reference_names = [str(x) for x in composer["reference_names"].tolist()]
    k_sorted = composer["k_sorted"]
    group_members = composer["group_members"].tolist()
    s_comp = composer["pred_logits"].astype(np.float32)
    s_cr = cr["pred_logits"].astype(np.float32)
    s_hbi, g_hbi, n_miss = compute_hbi(
        out / "sim_hbi.npz", reference_names, target_names, k_sorted, group_members,
    )
    singles = {
        "hbi": _pool_keys(pool_from_scores(s_hbi, k_sorted, target_names, g_hbi, group_members)),
        "composer": _pool_keys(pool_from_scores(s_comp, k_sorted, target_names, None, group_members)),
        "cr": _pool_keys(pool_from_scores(s_cr, k_sorted, target_names, None, group_members)),
    }
    h = zscore_rows(s_hbi)
    c = zscore_rows(s_comp)
    r = zscore_rows(s_cr)
    a, b, g = FROZEN["alpha_hbi"], FROZEN["beta_composer"], FROZEN["gamma_cr"]
    frozen_m = _pool_keys(
        pool_from_scores(a * h + b * c + g * r, k_sorted, target_names, None, group_members)
    )
    grid_best = None
    fusion_path = out / "fusion_metrics.json"
    if fusion_path.is_file():
        fusion = json.loads(fusion_path.read_text())
        grid_best = fusion.get("zscore", {}).get("best")

    coarse = None
    if args.dual_metrics and Path(args.dual_metrics).is_file():
        dm = json.loads(Path(args.dual_metrics).read_text())
        coarse = {
            "metrics": dm.get("metrics"),
            "params": dm.get("params"),
            "ckpt": dm.get("ckpt"),
            "id": dm.get("id"),
            "name": dm.get("name"),
        }

    report = {
        "tag": args.tag,
        "k_rerank": 200,
        "n_queries": len(target_names),
        "hbi_missing_rows": int(n_miss),
        "note": (
            "Headline fuse uses frozen mainline CIRR weights 0.20/0.35/0.45; "
            "not retuned. Grid-best is val-tuned and not for ablation comparison."
        ),
        "coarse_dual": coarse,
        "rerank_singles": singles,
        "fusion_frozen_mainline": {**FROZEN, **frozen_m},
        "fusion_grid_best_reference_only": grid_best,
    }
    dest = out / "rerank_final.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def row(tag, m):
        if not m:
            return
        print(
            f"  {tag:22s}  R@1={m.get('recall_at1', m.get('pool_recall_at1', 0)):6.2f}  "
            f"R@5={m.get('recall_at5', m.get('pool_recall_at5', 0)):6.2f}  "
            f"R@10={m.get('recall_at10', m.get('pool_recall_at10', 0)):6.2f}  "
            f"R@50={m.get('recall_at50', m.get('pool_recall_at50', 0)):6.2f}",
            flush=True,
        )

    print(f"=== CIRR {args.tag} dual coarse + rerank K=200 ===", flush=True)
    if coarse and coarse.get("metrics"):
        print("coarse dual:", flush=True)
        row("dual", coarse["metrics"])
    print("rerank singles:", flush=True)
    for name in ("hbi", "composer", "cr"):
        row(name, singles[name])
    print("frozen fuse 0.20/0.35/0.45:", flush=True)
    row("frozen", frozen_m)
    if grid_best:
        print("grid-best (reference only, not for comparison):", flush=True)
        row("grid", grid_best)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
