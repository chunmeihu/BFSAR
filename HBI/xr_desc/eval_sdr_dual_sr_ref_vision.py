#!/usr/bin/env python3
"""Dual-path 类SDR-HBI: both datasets, same recipe.

  Path_ref: S_d=MIC, S_m=relative, S_r=reference image
  Path_vis: S_d=MIC, S_m=relative, S_r=vision_side WTI
  Fuse:     w·z(S_ref) + (1-w)·z(S_vis)

Each dataset sweeps its own α/β/w. Cache-only, no GPU.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.fiq.vision_desc.eval_fiq_hbi_compose_fusion import load_wti_cache, zscore_rows  # noqa: E402

FIQ_WTI = ROOT / "output/fiq_vision_desc/refsim_fusion_v1"
FIQ_FEAT = ROOT / "output/fiq_vision_desc/compose_fusion_v1"
FIQ_OUT = ROOT / "output/fiq_vision_desc/sdr_dual_sr_ref_vision_v1"
CIRR_NPZ = ROOT / "output/cirr_vision_desc/sdr_wti_native_val/branch_sims.npz"
CIRR_OUT = ROOT / "output/cirr_vision_desc/sdr_dual_sr_ref_vision_val"

CATS = ("dress", "shirt", "toptee")
COMPOSE41 = {"recall_at10": 41.53580119474895, "recall_at50": 63.94058744698183}
CIRR_SOTA = {"recall_at1": 41.96022046489336, "recall_at50": 96.33357296908699}
ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4)
BETAS = (0.2, 0.4, 0.5, 0.6)
WS = (0.3, 0.4, 0.5, 0.6, 0.7)


def sdr_wti(zd, zm, zr, alpha, beta):
    anc = (1.0 - alpha) * zd + alpha * zr
    return ((1.0 + beta) * anc - beta * (zd - zm)).astype(np.float32)


def tgt_ranks(sim, ref_i, tgt_i):
    s = np.array(sim, dtype=np.float64, copy=True)
    n = s.shape[0]
    rows = np.arange(n)
    valid = ref_i >= 0
    s[rows[valid], ref_i[valid]] = -1e9
    return (s > s[rows, tgt_i][:, None]).sum(axis=1)


def r_at(rk, k):
    return 100.0 * float(np.mean(rk < k))


def fuse_w(a, b, w):
    return (w * zscore_rows(a) + (1.0 - w) * zscore_rows(b)).astype(np.float32)


def load_fiq() -> Dict[str, Dict]:
    data = {}
    for cat in CATS:
        w = load_wti_cache(FIQ_WTI, cat)
        f = np.load(FIQ_FEAT / f"feats_{cat}.npz", allow_pickle=True)
        idx = [str(x) for x in f["index_names"].tolist()]
        refs = [str(x) for x in f["refs"].tolist()]
        tgts = [str(x) for x in f["tgts"].tolist()]
        name = {n: i for i, n in enumerate(idx)}
        data[cat] = {
            "zd": zscore_rows(w["modified"]),
            "zm": zscore_rows(w["relative"]),
            "zr_ref": zscore_rows(w["ref"]),
            "zr_vis": zscore_rows(w["vision"]),
            "ref_i": np.array([name.get(r, -1) for r in refs], dtype=np.int32),
            "tgt_i": np.array([name[t] for t in tgts], dtype=np.int32),
        }
    return data


def load_cirr() -> Dict:
    d = np.load(CIRR_NPZ, allow_pickle=True)
    index_names = [str(x) for x in d["index_names"].tolist()]
    refs = [str(x) for x in d["refs"].tolist()]
    tgts = [str(x) for x in d["tgts"].tolist()]
    name = {n: i for i, n in enumerate(index_names)}
    return {
        "zd": zscore_rows(d["sim_modified"]),
        "zm": zscore_rows(d["sim_relative"]),
        "zr_ref": zscore_rows(d["sim_ref"]),
        "zr_vis": zscore_rows(d["sim_vision"]),
        "ref_i": np.array([name[r] for r in refs], dtype=np.int32),
        "tgt_i": np.array([name[t] for t in tgts], dtype=np.int32),
        "index_names": index_names,
        "refs": refs,
        "tgts": tgts,
        "groups": [list(g) for g in d["groups"].tolist()],
    }


def fiq_mean(data, sim_fn) -> Dict[str, float]:
    r10, r50 = [], []
    per = {}
    for c in CATS:
        d = data[c]
        rk = tgt_ranks(sim_fn(d), d["ref_i"], d["tgt_i"])
        per[c] = {"recall_at10": r_at(rk, 10), "recall_at50": r_at(rk, 50)}
        r10.append(per[c]["recall_at10"])
        r50.append(per[c]["recall_at50"])
    return {
        "recall_at10": float(np.mean(r10)),
        "recall_at50": float(np.mean(r50)),
        "per": per,
    }


def cirr_ks(d, sim) -> Dict[str, float]:
    rk = tgt_ranks(sim, d["ref_i"], d["tgt_i"])
    return {
        "recall_at1": r_at(rk, 1),
        "recall_at5": r_at(rk, 5),
        "recall_at10": r_at(rk, 10),
        "recall_at50": r_at(rk, 50),
    }


def _drop_per(m: Dict) -> Dict:
    return {k: v for k, v in m.items() if k != "per"}


def sweep_dataset(kind: str, pack) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Both paths share α/β; only fusion weight w is extra. Per-dataset grid search."""

    key = "recall_at10" if kind == "fiq" else "recall_at1"
    sec = "recall_at50"

    singles = {}
    for tag, a, b in (("paper_cirr", 0.05, 0.5), ("fiq_best", 0.3, 0.4), ("cirr_vis_best", 0.3, 0.6)):
        if kind == "fiq":
            singles[f"ref_only_{tag}"] = _drop_per(
                fiq_mean(pack, lambda d, a=a, b=b: sdr_wti(d["zd"], d["zm"], d["zr_ref"], a, b))
            )
            singles[f"vis_only_{tag}"] = _drop_per(
                fiq_mean(pack, lambda d, a=a, b=b: sdr_wti(d["zd"], d["zm"], d["zr_vis"], a, b))
            )
        else:
            singles[f"ref_only_{tag}"] = cirr_ks(
                pack, sdr_wti(pack["zd"], pack["zm"], pack["zr_ref"], a, b)
            )
            singles[f"vis_only_{tag}"] = cirr_ks(
                pack, sdr_wti(pack["zd"], pack["zm"], pack["zr_vis"], a, b)
            )

    shared = []
    if kind == "fiq":
        for a, b in product(ALPHAS, BETAS):
            per_cat = {
                c: (
                    zscore_rows(sdr_wti(pack[c]["zd"], pack[c]["zm"], pack[c]["zr_ref"], a, b)),
                    zscore_rows(sdr_wti(pack[c]["zd"], pack[c]["zm"], pack[c]["zr_vis"], a, b)),
                )
                for c in CATS
            }
            for w in WS:
                r10, r50 = [], []
                for c in CATS:
                    fused = (w * per_cat[c][0] + (1.0 - w) * per_cat[c][1]).astype(np.float32)
                    rk = tgt_ranks(fused, pack[c]["ref_i"], pack[c]["tgt_i"])
                    r10.append(r_at(rk, 10))
                    r50.append(r_at(rk, 50))
                shared.append({
                    "alpha_ref": a, "beta_ref": b, "alpha_vis": a, "beta_vis": b, "w_ref": w,
                    "recall_at10": float(np.mean(r10)),
                    "recall_at50": float(np.mean(r50)),
                })
            print(f"  FIQ α={a} β={b} done", flush=True)
    else:
        for a, b in product(ALPHAS, BETAS):
            z_ref = zscore_rows(sdr_wti(pack["zd"], pack["zm"], pack["zr_ref"], a, b))
            z_vis = zscore_rows(sdr_wti(pack["zd"], pack["zm"], pack["zr_vis"], a, b))
            for w in WS:
                fused = (w * z_ref + (1.0 - w) * z_vis).astype(np.float32)
                m = cirr_ks(pack, fused)
                shared.append({
                    "alpha_ref": a, "beta_ref": b, "alpha_vis": a, "beta_vis": b, "w_ref": w, **m,
                })
            print(f"  CIRR α={a} β={b} done", flush=True)

    shared.sort(key=lambda r: (-r[key], -r[sec]))
    return singles, shared, shared


def cirr_full_metrics(sim, pack) -> Dict:
    from HBI.cirr.vision_desc.eval_hbi_triple_vision import _metrics
    from HBI.cirr.vision_desc.eval_hbi_triple_vision_removed import metrics_row

    return metrics_row(_metrics(sim, pack["index_names"], pack["refs"], pack["tgts"], pack["groups"]))


def main():
    print("load FIQ …")
    fiq = load_fiq()
    print("sweep FIQ …")
    fiq_singles, fiq_shared, fiq_indep = sweep_dataset("fiq", fiq)
    FIQ_OUT.mkdir(parents=True, exist_ok=True)
    fiq_payload = {
        "exp": "sdr_dual_sr_ref_vision_fiq",
        "recipe": "path_ref: S_r=ref image; path_vis: S_r=vision WTI; both S_d=MIC S_m=relative; fuse w*z(ref)+(1-w)*z(vis)",
        "compose41": COMPOSE41,
        "singles": fiq_singles,
        "shared_ab_best": fiq_shared[0],
        "shared_ab_top10": fiq_shared[:10],
        "indep_ab_best": fiq_indep[0],
        "indep_ab_top10": fiq_indep[:10],
    }
    (FIQ_OUT / "metrics.json").write_text(json.dumps(fiq_payload, indent=2, ensure_ascii=False))

    print("load CIRR …")
    cirr = load_cirr()
    print("sweep CIRR …")
    cirr_singles, cirr_shared, cirr_indep = sweep_dataset("cirr", cirr)

    def materialize(row):
        s_ref = sdr_wti(cirr["zd"], cirr["zm"], cirr["zr_ref"], row["alpha_ref"], row["beta_ref"])
        s_vis = sdr_wti(cirr["zd"], cirr["zm"], cirr["zr_vis"], row["alpha_vis"], row["beta_vis"])
        return fuse_w(s_ref, s_vis, row["w_ref"])

    print("CIRR full metrics on best …")
    shared_full = cirr_full_metrics(materialize(cirr_shared[0]), cirr)
    indep_full = cirr_full_metrics(materialize(cirr_indep[0]), cirr)
    CIRR_OUT.mkdir(parents=True, exist_ok=True)
    cirr_payload = {
        "exp": "sdr_dual_sr_ref_vision_cirr",
        "recipe": "path_ref: S_r=ref image; path_vis: S_r=vision WTI; both S_d=MIC S_m=relative; fuse w*z(ref)+(1-w)*z(vis)",
        "sota_recomputed": CIRR_SOTA,
        "singles": cirr_singles,
        "shared_ab_best": {**cirr_shared[0], "full": shared_full},
        "shared_ab_top10": cirr_shared[:10],
        "indep_ab_best": {**cirr_indep[0], "full": indep_full},
        "indep_ab_top10": cirr_indep[:10],
    }
    (CIRR_OUT / "metrics.json").write_text(json.dumps(cirr_payload, indent=2, ensure_ascii=False))

    def fiq_line(name, m):
        print(f"  {name:42s} {m['recall_at10']:6.2f} / {m['recall_at50']:6.2f}")

    def cirr_line(name, m):
        r1 = m.get("recall_at1", m.get("full", {}).get("recall_at1"))
        r50 = m.get("recall_at50", m.get("full", {}).get("recall_at50"))
        print(f"  {name:42s} R@1={r1:6.2f}  R@50={r50:6.2f}")

    print("\n=== FIQ mean R@10 / R@50 ===")
    for k, v in fiq_singles.items():
        fiq_line(k, v)
    fiq_line("shared αβ best", fiq_shared[0])
    fiq_line("indep αβ best", fiq_indep[0])
    fiq_line("compose41", COMPOSE41)
    print(
        f"  shared params α={fiq_shared[0]['alpha_ref']} β={fiq_shared[0]['beta_ref']} w_ref={fiq_shared[0]['w_ref']}"
    )
    print(
        f"  indep params  α_ref={fiq_indep[0]['alpha_ref']} β_ref={fiq_indep[0]['beta_ref']} "
        f"α_vis={fiq_indep[0]['alpha_vis']} β_vis={fiq_indep[0]['beta_vis']} w_ref={fiq_indep[0]['w_ref']}"
    )

    print("\n=== CIRR val R@1 / R@50 ===")
    for k, v in cirr_singles.items():
        cirr_line(k, v)
    cirr_line("shared αβ best (full)", shared_full)
    cirr_line("indep αβ best (full)", indep_full)
    cirr_line("sota triple-removed", CIRR_SOTA)
    print(
        f"  shared params α={cirr_shared[0]['alpha_ref']} β={cirr_shared[0]['beta_ref']} w_ref={cirr_shared[0]['w_ref']}"
    )
    print(
        f"  indep params  α_ref={cirr_indep[0]['alpha_ref']} β_ref={cirr_indep[0]['beta_ref']} "
        f"α_vis={cirr_indep[0]['alpha_vis']} β_vis={cirr_indep[0]['beta_vis']} w_ref={cirr_indep[0]['w_ref']}"
    )
    print(f"wrote {FIQ_OUT / 'metrics.json'}")
    print(f"wrote {CIRR_OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
