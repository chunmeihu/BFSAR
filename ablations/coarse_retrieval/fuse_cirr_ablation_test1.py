#!/usr/bin/env python3
"""CIRR test1 ablation fuse: frozen 0.25/0.40/0.35 → CodaLab JSON+zip.

Writes only under output/ablations/. Does not touch mainline submission_CIRR.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir
from HBI.cirr.vision_desc.eval_hbi_composer_cr_fusion import zscore_rows  # noqa: E402
from HBI.cirr.vision_desc.eval_hbi_composer_cr_fusion_test1 import (  # noqa: E402
    _load_scores,
    hbi_scores_test1,
    write_submissions,
)

FROZEN = {"alpha_hbi": 0.25, "beta_composer": 0.40, "gamma_cr": 0.35}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hbi_npz", required=True)
    p.add_argument("--submission_name", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--alpha_hbi", type=float, default=FROZEN["alpha_hbi"])
    p.add_argument("--beta_composer", type=float, default=FROZEN["beta_composer"])
    p.add_argument("--gamma_cr", type=float, default=FROZEN["gamma_cr"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ensure_ablation_out_dir(args.out_dir)
    composer_npz = out / "composer_itc_scores.npz"
    cr_npz = out / "cr_nlvr_scores.npz"
    if not composer_npz.is_file() or not cr_npz.is_file():
        raise SystemExit(f"ABORT: missing composer/cr scores under {out}")

    a, b, g = args.alpha_hbi, args.beta_composer, args.gamma_cr
    if min(a, b, g) < -1e-12:
        raise ValueError(f"weights must be non-negative, got {a}/{b}/{g}")
    if a + b + g <= 1e-12:
        raise ValueError(f"weights must have positive sum, got {a}+{b}+{g}")
    # Leave-one-out keeps remaining weights as-is (may sum ≠ 1); ranking is scale-invariant.

    comp = _load_scores(composer_npz)
    cr = _load_scores(cr_npz)
    if comp["pairids"] != cr["pairids"]:
        raise RuntimeError("Composer/CR pairid order mismatch")
    if not np.array_equal(comp["k_sorted"].astype(str), cr["k_sorted"].astype(str)):
        raise RuntimeError("Composer/CR top-K name order mismatch")

    hbi = hbi_scores_test1(
        Path(args.hbi_npz), comp["pairids"], comp["k_sorted"], comp["group_members_no_ref"]
    )
    fused = a * zscore_rows(hbi["pred_logits"]) + b * zscore_rows(comp["pred_logits"]) + g * zscore_rows(cr["pred_logits"])
    fused_g = (
        a * zscore_rows(hbi["group_logits"])
        + b * zscore_rows(comp["group_logits"])
        + g * zscore_rows(cr["group_logits"])
    )
    idx = np.argsort(-fused, axis=1)
    ranked50 = np.take_along_axis(comp["k_sorted"], idx, axis=1)
    g_idx = np.argsort(-fused_g, axis=1)
    g_arr = np.array(comp["group_members_no_ref"], dtype=object)
    ranked_g = np.take_along_axis(g_arr, g_idx, axis=1)

    paths = write_submissions(comp["pairids"], ranked50, ranked_g, out, args.submission_name)
    zpath = out / f"{args.submission_name}_cirr_test1.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(paths["recall"], arcname=paths["recall"].name)
        zf.write(paths["recall_subset"], arcname=paths["recall_subset"].name)

    rec = json.loads(paths["recall"].read_text())
    n = len([k for k in rec if k not in ("version", "metric")])
    meta = {
        "split": "test1",
        "tag": args.tag,
        "n_queries": n,
        "k_rerank": int(comp["k_sorted"].shape[1]),
        "weights_zscore": {"hbi": a, "composer": b, "cr": g},
        "weights_source": (
            f"zscore fuse α_hbi={a:.2f} β_composer={b:.2f} γ_cr={g:.2f} "
            "(A3t val K=200 simplex 0.25/0.40/0.35; leave-one-out keeps remaining, not renormalized)"
            if min(a, b, g) < 1e-12
            else (
                f"zscore fuse α_hbi={a:.2f} β_composer={b:.2f} γ_cr={g:.2f} "
                "(selected on A3t val K=200 simplex; applied frozen to test1)"
            )
        ),
        "hbi_missing_rows": int(hbi["missing"]),
        "hbi_npz": str(args.hbi_npz),
        "submission_name": args.submission_name,
        "recall_json": str(paths["recall"]),
        "recall_subset_json": str(paths["recall_subset"]),
        "zip": str(zpath),
        "note": "CIRR test1 labels are hidden; upload the zip to CodaLab.",
    }
    (out / "test1_fusion_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
