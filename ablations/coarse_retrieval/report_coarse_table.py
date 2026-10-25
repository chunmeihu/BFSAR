#!/usr/bin/env python3
"""Build FIQ / CIRR coarse ablation summary tables from metrics.json files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir

FIQ_ROOT = ROOT / "output/ablations/fiq_coarse"
CIRR_ROOT = ROOT / "output/ablations/cirr_coarse"

FIQ_DIRS = [
    "A1_zeroshot_mic_openclip",
    "A2_no_banzhaf",
    "A2e_no_banzhaf_entity",
    "A2e_no_banzhaf_entity_dual",
    "A3_entity_only",
    "A3_entity_only_dual",
    "A4_mic_only_wti",
    "A5_path_ref",
    "A6_path_vis",
    "A7_dual_full",
]
CIRR_DIRS = FIQ_DIRS + [
    "A3s_skl03_entity",
    "A3s_skl03_entity_dual",
    "A3t_skl01_entity",
    "A3t_skl01_entity_dual",
]


def _fiq_row(d: Path):
    m = json.loads((d / "metrics.json").read_text())
    mean = m.get("mean") or {}
    per = m.get("per") or {}
    row = {
        "id": m.get("id", d.name.split("_")[0]),
        "name": m.get("name", d.name),
        "mean_R@10": mean.get("recall_at10"),
        "mean_R@50": mean.get("recall_at50"),
    }
    for c in ("dress", "shirt", "toptee"):
        if c in per:
            row[f"{c}_R@10"] = per[c].get("recall_at10")
            row[f"{c}_R@50"] = per[c].get("recall_at50")
    return row


def _cirr_row(d: Path):
    m = json.loads((d / "metrics.json").read_text())
    met = m.get("metrics") or {}
    return {
        "id": m.get("id", d.name.split("_")[0]),
        "name": m.get("name", d.name),
        "R@1": met.get("recall_at1"),
        "R@5": met.get("recall_at5"),
        "R@10": met.get("recall_at10"),
        "R@50": met.get("recall_at50"),
        "G-R@1": met.get("group_recall_at1"),
        "G-R@2": met.get("group_recall_at2"),
        "G-R@3": met.get("group_recall_at3"),
    }


def _md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.2f}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    fiq_rows, cirr_rows = [], []
    for name in FIQ_DIRS:
        p = FIQ_ROOT / name / "metrics.json"
        if p.is_file():
            fiq_rows.append(_fiq_row(p.parent))
    for name in CIRR_DIRS:
        p = CIRR_ROOT / name / "metrics.json"
        if p.is_file():
            cirr_rows.append(_cirr_row(p.parent))

    fiq_cols = [
        "id", "name",
        "dress_R@10", "dress_R@50",
        "shirt_R@10", "shirt_R@50",
        "toptee_R@10", "toptee_R@50",
        "mean_R@10", "mean_R@50",
    ]
    cirr_cols = ["id", "name", "R@1", "R@5", "R@10", "R@50", "G-R@1", "G-R@2", "G-R@3"]

    summary = {
        "notes": {
            "fiq_metrics": "R@10 / R@50 per-cat + mean; Original-Split large gallery",
            "cirr_metrics": "R@1/5/10/50 + G-R@1/2/3",
            "fiq_sota_params": {"alpha": 0.4, "beta": 0.2, "w_ref": 0.5, "ckpt": "best.3"},
            "cirr_sota_params": {
                "alpha_ref": 0.5, "beta_ref": 0.2,
                "alpha_vis": 0.2, "beta_vis": 0.8, "w_ref": 0.2, "ckpt": "best.9",
            },
            "mainline_rerank_k": {"fiq": 300, "cirr": 200},
        },
        "fiq": fiq_rows,
        "cirr": cirr_rows,
    }

    out_fiq = ensure_ablation_out_dir(FIQ_ROOT)
    out_cirr = ensure_ablation_out_dir(CIRR_ROOT)
    (out_fiq / "summary_table.json").write_text(json.dumps({"fiq": fiq_rows, "notes": summary["notes"]}, indent=2) + "\n")
    (out_cirr / "summary_table.json").write_text(json.dumps({"cirr": cirr_rows, "notes": summary["notes"]}, indent=2) + "\n")
    both = ensure_ablation_out_dir(ROOT / "output/ablations")
    (both / "summary_both.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = []
    md.append("# Coarse ablation summary\n")
    md.append("## FashionIQ (R@10 / R@50)\n")
    md.append(_md_table(fiq_rows, fiq_cols) if fiq_rows else "_(no results yet)_")
    md.append("\n\n## CIRR (R@K + G-R@K)\n")
    md.append(_md_table(cirr_rows, cirr_cols) if cirr_rows else "_(no results yet)_")
    md.append("\n\nMainline rerank K note: FIQ=300, CIRR=200.\n")
    (both / "summary_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {both / 'summary_both.json'}", flush=True)


if __name__ == "__main__":
    main()
