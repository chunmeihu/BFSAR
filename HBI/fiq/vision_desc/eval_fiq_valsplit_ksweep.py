#!/usr/bin/env python3
"""Fashion-IQ VAL-Split (MCoT small gallery) fusion at coarse K=200/300/500.

Gallery = unique reference ∪ target in val-triplet order (MCoT validate.py).
Metrics: MCoT protocol = do not drop the reference. Frozen fusion 0.25/0.30/0.45.

CPU-only. Reads VAL-Split dual coarse + Composer/CR scores from --score_dir
(must be scored on that same VAL-Split Top-K dump, not Original-Split).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CATS = ("dress", "shirt", "toptee")
FROZEN = (0.25, 0.30, 0.45)
MCOT_PAPER = {
    "dress": (58.45, 78.92),
    "shirt": (64.08, 83.25),
    "toptee": (67.18, 83.86),
    "mean": (63.24, 82.01),
}


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def r_at(ranks: np.ndarray, k: int) -> float:
    return 100.0 * float(np.mean(ranks < k))


def load_npz_names(path: Path) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def pool_hbi(sim: np.ndarray, index_names: Sequence[str], k_sorted: np.ndarray) -> np.ndarray:
    name_to_i = {str(n): i for i, n in enumerate(index_names)}
    n_q, k = k_sorted.shape
    out = np.empty((n_q, k), dtype=np.float64)
    for i in range(n_q):
        out[i] = [sim[i, name_to_i[str(n)]] for n in k_sorted[i]]
    return out


def ranks_merged(
    fused: np.ndarray,
    pool_names: np.ndarray,
    sim: np.ndarray,
    index_names: Sequence[str],
    refs: Sequence[str],
    tgts: Sequence[str],
    drop_ref: bool,
) -> np.ndarray:
    order = np.argsort(-fused, axis=1)
    head = np.take_along_axis(pool_names.astype(object), order, axis=1)
    coarse = np.asarray(index_names, dtype=object)[np.argsort(-sim, axis=1)]
    n = len(tgts)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        seen = set()
        ranking: List[str] = []
        for n_ in list(head[i]) + list(coarse[i]):
            s = str(n_)
            if s in seen:
                continue
            if drop_ref and s == str(refs[i]):
                continue
            seen.add(s)
            ranking.append(s)
        try:
            ranks[i] = ranking.index(str(tgts[i]))
        except ValueError:
            ranks[i] = len(ranking)
    return ranks


def metrics(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "recall_at10": r_at(ranks, 10),
        "recall_at50": r_at(ranks, 50),
        "recall_at100": r_at(ranks, 100),
        "recall_at200": r_at(ranks, 200),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--score_dir",
        type=str,
        default=str(ROOT / "output/fiq_rerank/fusion_k500_openclip_dual_valsplit"),
    )
    p.add_argument("--ks", type=str, default="200,300,500")
    p.add_argument("--out_json", type=str, default="")
    args = p.parse_args()
    score_dir = Path(args.score_dir)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    by_k: Dict[int, Dict] = {k: {"per_category": {}, "mean": {}} for k in ks}

    for cat in CATS:
        fused = load_npz_names(score_dir / f"hbi_fused_{cat}.npz")
        sim = np.asarray(fused["sim"], dtype=np.float64)
        index_names = [str(x) for x in fused["index_names"].tolist()]
        refs = [str(x) for x in fused["refs"].tolist()]
        tgts = [str(x) for x in fused["tgts"].tolist()]
        comp = load_npz_names(score_dir / f"composer_{cat}.npz")
        cr = load_npz_names(score_dir / f"cr_{cat}.npz")
        k_sorted = np.asarray(comp["k_sorted"]).astype(str)
        if not np.array_equal(k_sorted, np.asarray(cr["k_sorted"]).astype(str)):
            raise RuntimeError(f"{cat}: composer/CR k_sorted mismatch")
        s_hbi = pool_hbi(sim, index_names, k_sorted)
        s_comp = np.asarray(comp["pred_logits"], dtype=np.float64)
        s_cr = np.asarray(cr["pred_logits"], dtype=np.float64)
        k_max = k_sorted.shape[1]
        for k in ks:
            if k > k_max:
                raise ValueError(f"K={k} > scored pool {k_max}")
            h = zscore_rows(s_hbi[:, :k])
            c = zscore_rows(s_comp[:, :k])
            r = zscore_rows(s_cr[:, :k])
            fused_s = FROZEN[0] * h + FROZEN[1] * c + FROZEN[2] * r
            rk_mcot = ranks_merged(
                fused_s, k_sorted[:, :k], sim, index_names, refs, tgts, drop_ref=False,
            )
            rk_hbi = ranks_merged(
                fused_s, k_sorted[:, :k], sim, index_names, refs, tgts, drop_ref=True,
            )
            by_k[k]["per_category"][cat] = {
                "n_queries": len(refs),
                "n_gallery": len(index_names),
                "coverage_at_k": float(np.asarray(comp["labels"])[:, :k].any(axis=1).mean() * 100.0),
                "mcot_no_drop": metrics(rk_mcot),
                "hbi_drop_ref": metrics(rk_hbi),
                "paper_mcot": {
                    "recall_at10": MCOT_PAPER[cat][0],
                    "recall_at50": MCOT_PAPER[cat][1],
                },
            }
            m = by_k[k]["per_category"][cat]["mcot_no_drop"]
            print(
                f"K={k} {cat}: gallery={len(index_names)}  "
                f"R@10/50={m['recall_at10']:.2f}/{m['recall_at50']:.2f}  "
                f"(MCoT paper {MCOT_PAPER[cat][0]:.2f}/{MCOT_PAPER[cat][1]:.2f})",
                flush=True,
            )

    for k in ks:
        per = by_k[k]["per_category"]
        for proto in ("mcot_no_drop", "hbi_drop_ref"):
            by_k[k]["mean"][proto] = {
                "recall_at10": float(np.mean([per[c][proto]["recall_at10"] for c in CATS])),
                "recall_at50": float(np.mean([per[c][proto]["recall_at50"] for c in CATS])),
            }
        m = by_k[k]["mean"]["mcot_no_drop"]
        print(
            f"K={k} MEAN: {m['recall_at10']:.2f}/{m['recall_at50']:.2f}  "
            f"(MCoT paper {MCOT_PAPER['mean'][0]:.2f}/{MCOT_PAPER['mean'][1]:.2f})",
            flush=True,
        )

    report = {
        "gallery": "VAL-Split = unique ref∪tgt in val-triplet order (MCoT validate.py lines 133-139)",
        "protocol": "mcot_no_drop is the fair comparison to published MCoT-MVS",
        "fusion": "frozen 0.25/0.30/0.45 zscore-per-query on VAL-Split coarse Top-K",
        "coarse": "OpenCLIP dual Path_ref+Path_vis α=0.4 β=0.2 w_ref=0.5, re-zscore on VAL gallery",
        "mcot_paper_valsplit": MCOT_PAPER,
        "by_k": {str(k): by_k[k] for k in ks},
    }
    out = Path(args.out_json) if args.out_json else score_dir / "valsplit_k200_300_500.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
