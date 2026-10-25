#!/usr/bin/env python3
"""Pick CIRR Banzhaf SKL with the largest val dual gain vs A2e (no-Banzhaf).

Primary: dual R@1. Tie-break: G-R@1, then R@5.
A3t is included only after its val dual metrics exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.guards import ensure_ablation_out_dir

CIRR = ROOT / "output/ablations/cirr_coarse"
A2E_DUAL = CIRR / "A2e_no_banzhaf_entity_dual" / "metrics.json"
CANDIDATES = (
    ("A3", 1.0, CIRR / "A3_entity_only_dual" / "metrics.json"),
    ("A3s", 0.3, CIRR / "A3s_skl03_entity_dual" / "metrics.json"),
    ("A3t", 0.1, CIRR / "A3t_skl01_entity_dual" / "metrics.json"),
)
OUT = CIRR / "banzhaf_vs_a2e_pick.json"


def _dual(path: Path) -> dict:
    m = json.loads(path.read_text())
    met = m.get("metrics") or {}
    return {
        "r1": float(met["recall_at1"]),
        "r5": float(met["recall_at5"]),
        "r10": float(met.get("recall_at10") or 0.0),
        "gr1": float(met["group_recall_at1"]),
        "ckpt": m.get("ckpt"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--require-a3t", action="store_true")
    p.add_argument("--out", default=str(OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not A2E_DUAL.is_file():
        raise SystemExit(f"ABORT: missing A2e dual {A2E_DUAL}")
    a2e = _dual(A2E_DUAL)
    rows = []
    a3t_ready = False
    for tag, skl, path in CANDIDATES:
        if not path.is_file():
            continue
        if tag == "A3t":
            a3t_ready = True
        d = _dual(path)
        rows.append({
            "tag": tag,
            "skl": skl,
            "dual_R@1": d["r1"],
            "dual_R@5": d["r5"],
            "dual_R@10": d["r10"],
            "dual_G-R@1": d["gr1"],
            "delta_R@1": d["r1"] - a2e["r1"],
            "delta_R@5": d["r5"] - a2e["r5"],
            "delta_G-R@1": d["gr1"] - a2e["gr1"],
            "ckpt": d["ckpt"],
            "metrics": str(path),
        })
    if args.require_a3t and not a3t_ready:
        raise SystemExit("A3t val dual not ready; not picking yet")
    if not rows:
        raise SystemExit("ABORT: no Banzhaf dual metrics")
    rows.sort(key=lambda r: (r["delta_R@1"], r["delta_G-R@1"], r["delta_R@5"]), reverse=True)
    pick = rows[0]
    payload = {
        "split": "val",
        "criterion": "max dual R@1 vs A2e; tie-break G-R@1 then R@5",
        "a2e": {
            "tag": "A2e",
            "dual_R@1": a2e["r1"],
            "dual_R@5": a2e["r5"],
            "dual_G-R@1": a2e["gr1"],
        },
        "picked": pick["tag"],
        "test1_pair": [pick["tag"], "A2e"],
        "skipped_test1": [r["tag"] for r in rows if r["tag"] != pick["tag"]],
        "candidates": rows,
        "note": (
            "CIRR test1 full MIC+dual+rerank only for the picked Banzhaf SKL vs A2e. "
            "A3 skl=1 is not used on test1 unless it wins this comparison."
        ),
    }
    out = ensure_ablation_out_dir(Path(args.out).parent) / Path(args.out).name
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"PICK={pick['tag']} ΔR@1={pick['delta_R@1']:+.2f} ΔG-R@1={pick['delta_G-R@1']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
