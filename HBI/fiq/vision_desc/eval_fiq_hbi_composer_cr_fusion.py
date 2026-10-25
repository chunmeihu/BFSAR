#!/usr/bin/env python3
"""Fashion-IQ full rerank: 12-crop FIQ-best coarse + Composer ITC + FIQ CR.

Coarse weights are the 12-crop grid-best (0.3/0.4/0.3, μ=0), not CIRR-frozen.
Does not load CIRR composer/CR weights. Does not write under cirr/ paths.

Fusion (per-query z-score, CIRR recipe):
    s = 0.25 * s_coarse + 0.30 * s_Composer + 0.45 * s_CR
Also reports singles and a cheap val grid (not used to overwrite CIRR).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HBI.cirr.metrics_cirr_hbi import rank_by_similarity
from HBI.cirr.modeling_hbi_cirr import HBICIRR
from HBI.cirr.vision_desc.hbi_blip_rerank import load_blip_models, merge_rerank_with_coarse_gallery
from HBI.cirr.vision_desc.model_cirr_blip2_qtr import build_blip2_qtr, load_blip2_qtr_ckpt
from HBI.fiq import fiq_paths
from HBI.fiq.vision_desc.eval_fiq_hbi_cr_rerank import (
    align_to_official,
    build_topk_payload,
    encode_hbi_fused,
    recall_at_ks,
    rerank_category as cr_rerank_category,
)
from HBI.fiq.vision_desc.fiq_blip2_qtr_data import fiq_targetpad_preprocess
from HBI.fiq.vision_desc.fiq_blip2_qtr_rerank import rerank_category as qtr_rerank_category
from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from HBI.rerank.paths import COMPOSER_FIQ_DEFAULT, DEFAULT_FIQ_STAGE1, DEFAULT_FIQ_STAGE2

logger = logging.getLogger("fiq_hbi_composer_cr_fusion")

DEFAULT_HBI = str(ROOT / "output/fiq_retrieval/pytorch_model.bin.best.5")
DEFAULT_DUAL = str(ROOT / "data/fashioniq/vision_desc/val_dual_targets.json")
DEFAULT_COMPOSER = str(COMPOSER_FIQ_DEFAULT)
DEFAULT_STAGE1 = str(DEFAULT_FIQ_STAGE1)
DEFAULT_STAGE2 = str(DEFAULT_FIQ_STAGE2)
DEFAULT_OUT = str(ROOT / "output/fiq_rerank/fusion_k300_fiqbest")
FROZEN = (0.25, 0.30, 0.45)  # coarse / composer / CR from CIRR rerank recipe


def _abort_cirr(*paths: str) -> None:
    for p in paths:
        if "cirr" in os.path.abspath(str(p)).lower():
            raise ValueError(f"FIQ fusion must not use a cirr path: {p}")


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def pool_from_scores(scores: np.ndarray, labels: np.ndarray, ks=(10, 50, 100, 300)) -> Dict[str, float]:
    order = np.argsort(-np.asarray(scores), axis=1)
    ranked = np.take_along_axis(np.asarray(labels, dtype=bool), order, axis=1)
    out = {}
    k_max = ranked.shape[1]
    for k in ks:
        kk = min(int(k), k_max)
        out[f"pool_recall_at{k}"] = float(ranked[:, :kk].any(axis=1).mean() * 100.0)
    out["coverage"] = float(np.asarray(labels).any(axis=1).mean() * 100.0)
    return out


def _topk_pt(out_dir: Path, cat: str, k_save: int, split: str = "val") -> Path:
    return out_dir / f"hbi_top{k_save}_{split}_{cat}.pt"


def _hbi_npz(out_dir: Path, cat: str) -> Path:
    return out_dir / f"hbi_fused_{cat}.npz"


def _score_npz(out_dir: Path, tag: str, cat: str) -> Path:
    return out_dir / f"{tag}_{cat}.npz"


def dump_coarse(args, cats: Sequence[str], device) -> None:
    out_dir = Path(args.out_dir)
    split = getattr(args, "split", "val")
    tokenizer = ClipTokenizer()
    model = HBICIRR(args)
    state = torch.load(args.init_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded FIQ HBI %s missing=%d unexpected=%d", args.init_model, len(missing), len(unexpected))
    model = model.to(device).eval()
    for cat in cats:
        hbi = encode_hbi_fused(model, args, tokenizer, cat, device)
        np.savez_compressed(
            _hbi_npz(out_dir, cat),
            sim=hbi["sim"],
            index_names=np.array(hbi["index_names"], dtype=object),
            refs=np.array(hbi["refs"], dtype=object),
            tgts=np.array(hbi["tgts"], dtype=object),
        )
        sim, index_names, refs, tgts = align_to_official(hbi, cat, split=split)
        ranked = rank_by_similarity(sim, index_names)
        if split != "test":
            coarse = recall_at_ks(ranked, refs, tgts, ks=(10, 50, 100, 200, 300))
            logger.info(
                "FIQ-best coarse %s R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f R@300=%.2f",
                cat, coarse["recall_at10"], coarse["recall_at50"],
                coarse["recall_at100"], coarse["recall_at200"], coarse["recall_at300"],
            )
        else:
            coarse = {"n_queries": len(refs), "n_gallery": len(index_names), "split": split}
            logger.info("FIQ test coarse %s queries=%d gallery=%d (no labels)", cat, len(refs), len(index_names))
        payload = build_topk_payload(
            sim, index_names, refs, tgts, cat, int(args.k_save), split=split,
        )
        payload["source"] = "hbi_12crop_fiqbest_0.3_0.4_0.3_mu0"
        payload["weights"] = {
            "w_vision": args.w_vision,
            "w_modified": args.w_modified,
            "w_relative": args.w_relative,
            "mu": args.mu,
        }
        pt = _topk_pt(out_dir, cat, int(args.k_save), split=split)
        torch.save(payload, pt)
        meta = {
            "coarse": coarse,
            "topk_pt": str(pt),
            "n_queries": len(refs),
            "n_gallery": len(index_names),
            "split": split,
        }
        (out_dir / f"coarse_{cat}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del model
    _free_cuda()


def _parse_composer_ckpt_by_cat(raw: str | None) -> Dict[str, str]:
    """Parse 'dress=/a.pt,shirt=/b.pt' into {cat: path}."""
    out: Dict[str, str] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--composer_ckpt_by_cat entry must be cat=path, got: {part}")
        cat, path = part.split("=", 1)
        cat, path = cat.strip(), path.strip()
        if not cat or not path:
            raise ValueError(f"bad --composer_ckpt_by_cat entry: {part}")
        _abort_cirr(path)
        out[cat] = path
    return out


def score_composer(args, cats: Sequence[str], device) -> None:
    by_cat = _parse_composer_ckpt_by_cat(getattr(args, "composer_ckpt_by_cat", None))
    if by_cat:
        missing_cats = [c for c in cats if c not in by_cat]
        if missing_cats:
            raise ValueError(f"--composer_ckpt_by_cat missing cats: {missing_cats}")
        ckpt_plan = [(c, by_cat[c]) for c in cats]
    else:
        _abort_cirr(args.composer_ckpt)
        ckpt_plan = [(c, args.composer_ckpt) for c in cats]

    preprocess = fiq_targetpad_preprocess()
    from HBI.fiq.vision_desc import fiq_blip2_qtr_rerank as qtr_mod

    # Group consecutive cats that share the same ckpt so we load once when possible.
    i = 0
    while i < len(ckpt_plan):
        ckpt = ckpt_plan[i][1]
        group = [ckpt_plan[i][0]]
        j = i + 1
        while j < len(ckpt_plan) and ckpt_plan[j][1] == ckpt:
            group.append(ckpt_plan[j][0])
            j += 1
        _abort_cirr(ckpt)
        if not Path(ckpt).is_file():
            raise FileNotFoundError(ckpt)
        model = build_blip2_qtr(
            device,
            pretrained_id=args.pretrained_id,
            local_files_only=True,
            freeze_vision=True,
            stage="a",
        )
        missing, unexpected = load_blip2_qtr_ckpt(model, ckpt, device)
        logger.info(
            "Loaded FIQ Composer %s for %s missing=%d unexpected=%d",
            ckpt, group, len(missing), len(unexpected),
        )
        model.eval()
        for cat in group:
            pt = _topk_pt(Path(args.out_dir), cat, int(args.k_save), split=getattr(args, "split", "val"))
            if not pt.is_file():
                raise FileNotFoundError(pt)
            qtr_mod._GALLERY_CACHE.clear()
            out = qtr_rerank_category(
                model, cat, topk_pt=pt, k_rerank=int(args.k_rerank), score_mode="itc",
                preprocess=preprocess, gallery_batch=args.gallery_batch, num_workers=args.workers,
            )
            np.savez_compressed(
                _score_npz(Path(args.out_dir), "composer", cat),
                pred_logits=out["pred_logits"],
                k_sorted=np.asarray(out["k_sorted"], dtype=object),
                labels=np.asarray(out["labels"], dtype=bool),
                reference_names=np.asarray(out["reference_names"], dtype=object),
                target_names=np.asarray(out["target_names"], dtype=object),
            )
            meta = dict(out["metrics"])
            meta["composer_ckpt"] = ckpt
            (Path(args.out_dir) / f"composer_{cat}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8",
            )
            qtr_mod._GALLERY_CACHE.clear()
            _free_cuda()
        del model
        _free_cuda()
        i = j


def score_cr(args, cats: Sequence[str], device) -> None:
    _abort_cirr(args.stage1_ckpt, args.stage2_ckpt)
    model_stage1, model_stage2, preprocess, _mods = load_blip_models(
        Path(args.stage1_ckpt), Path(args.stage2_ckpt)
    )
    model_stage1 = model_stage1.to(device).eval()
    model_stage2 = model_stage2.to(device).eval()
    for cat in cats:
        pt = _topk_pt(Path(args.out_dir), cat, int(args.k_save), split=getattr(args, "split", "val"))
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        reranked, pred, pool_metrics = cr_rerank_category(
            payload, model_stage1, model_stage2, preprocess, cat,
            int(args.k_rerank), device, args.workers, args.cr_gallery_batch,
            cand_chunk=int(args.cr_cand_chunk),
            split=getattr(args, "split", "val"),
        )
        labels = payload["labels"]
        if torch.is_tensor(labels):
            labels = labels.numpy()
        labels = np.asarray(labels)[:, : int(args.k_rerank)]
        k_sorted = np.array(payload["sorted_index_names"], dtype=object)[:, : int(args.k_rerank)]
        np.savez_compressed(
            _score_npz(Path(args.out_dir), "cr", cat),
            pred_logits=np.asarray(pred, dtype=np.float32),
            k_sorted=k_sorted,
            labels=np.asarray(labels, dtype=bool),
            reference_names=np.asarray(payload["reference_names"], dtype=object),
            target_names=np.asarray(payload["target_names"], dtype=object),
            reranked=np.asarray(reranked, dtype=object),
        )
        (Path(args.out_dir) / f"cr_{cat}.json").write_text(
            json.dumps(pool_metrics, indent=2), encoding="utf-8",
        )
        logger.info(
            "FIQ %s CR | pool R@10=%.2f R@50=%.2f coverage=%.2f",
            cat, pool_metrics["pool_recall_at10"], pool_metrics["pool_recall_at50"],
            pool_metrics["coverage"],
        )
        _free_cuda()
    del model_stage1, model_stage2
    _free_cuda()


def _load_score(path: Path) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {
        "pred_logits": d["pred_logits"].astype(np.float32),
        "k_sorted": d["k_sorted"],
        "labels": d["labels"].astype(bool),
        "reference_names": [str(x) for x in d["reference_names"].tolist()],
        "target_names": [str(x) for x in d["target_names"].tolist()],
    }


def _hbi_pool_scores(out_dir: Path, cat: str, k_sorted: np.ndarray, refs: List[str], tgts: List[str]):
    """Gather coarse scores for the Top-K names, aligned to score-file query order.

    Score dumps (Composer/CR) may use compose-feat order / a subset of official val;
    do not force official caption order here.
    """
    raw = np.load(_hbi_npz(out_dir, cat), allow_pickle=True)
    sim_all = np.asarray(raw["sim"], dtype=np.float32)
    index_names = [str(x) for x in raw["index_names"].tolist()]
    hbi_refs = [str(x) for x in raw["refs"].tolist()]
    hbi_tgts = [str(x) for x in raw["tgts"].tolist()]
    key_to_i = {(r, t): i for i, (r, t) in enumerate(zip(hbi_refs, hbi_tgts))}
    try:
        row_idx = [key_to_i[(r, t)] for r, t in zip(refs, tgts)]
    except KeyError as e:
        raise RuntimeError(f"{cat}: score-file query missing from hbi_fused: {e}") from e
    sim = sim_all[row_idx]
    name_to_j = {str(n): i for i, n in enumerate(index_names)}
    n, k = k_sorted.shape
    if n != len(refs):
        raise RuntimeError(f"{cat}: k_sorted rows {n} != n_queries {len(refs)}")
    scores = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        cols = [name_to_j[str(nm)] for nm in k_sorted[i]]
        scores[i] = sim[i, cols]
    return scores, sim, index_names, list(refs), list(tgts)


def _full_metrics(scores: np.ndarray, k_sorted: np.ndarray, sim: np.ndarray, index_names, refs, tgts) -> Dict[str, float]:
    order = np.argsort(-np.asarray(scores), axis=1)
    reranked = np.take_along_axis(k_sorted, order, axis=1)
    coarse_sorted = rank_by_similarity(sim, index_names)
    n_g = coarse_sorted.shape[1]
    ref_exp = np.repeat(np.asarray(refs, dtype=object), n_g).reshape(len(refs), n_g)
    coarse_no_ref = coarse_sorted[coarse_sorted != ref_exp].reshape(len(refs), n_g - 1)
    merged = merge_rerank_with_coarse_gallery(coarse_no_ref, reranked)
    return recall_at_ks(merged, refs, tgts, ks=(10, 50, 100, 200, 300))


# Val-locked fusion weights (from K500 compose41 grid on val; do not retune on test).
TEST_PERCAT_WEIGHTS = {
    "dress": (0.1, 0.5, 0.4),
    "shirt": (0.1, 0.5, 0.4),
    "toptee": (0.1, 0.6, 0.3),
}


def _merged_ranking(
    scores: np.ndarray,
    k_sorted: np.ndarray,
    sim: np.ndarray,
    index_names,
    refs,
    top_n: int = 50,
) -> List[List[str]]:
    order = np.argsort(-np.asarray(scores), axis=1)
    reranked = np.take_along_axis(k_sorted, order, axis=1)
    coarse_sorted = rank_by_similarity(sim, index_names)
    n_g = coarse_sorted.shape[1]
    ref_exp = np.repeat(np.asarray(refs, dtype=object), n_g).reshape(len(refs), n_g)
    coarse_no_ref = coarse_sorted[coarse_sorted != ref_exp].reshape(len(refs), n_g - 1)
    merged = merge_rerank_with_coarse_gallery(coarse_no_ref, reranked)
    out = []
    for i in range(merged.shape[0]):
        ranking = [str(n) for n in merged[i] if str(n) != str(refs[i])]
        out.append(ranking[:top_n])
    return out


def write_codalab_predict(
    out_dir: Path,
    cats: Sequence[str],
    rankings: Dict[str, List[List[str]]],
    split: str = "test",
) -> Dict[str, str]:
    """Write Fashion-IQ CodaLab `{cat}.predict.json` (caption row + ranking top-50)."""
    submit_dir = out_dir / "codalab_submit"
    submit_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cat in cats:
        with open(fiq_paths.caption_json(cat, split), "r", encoding="utf-8") as f:
            rows = json.load(f)
        ranks = rankings[cat]
        if len(rows) != len(ranks):
            raise RuntimeError(f"{cat}: captions {len(rows)} != rankings {len(ranks)}")
        out_rows = []
        for row, ranking in zip(rows, ranks):
            item = {
                "candidate": row["candidate"],
                "captions": row.get("captions") or [],
                "ranking": list(ranking),
            }
            out_rows.append(item)
        path = submit_dir / f"{cat}.predict.json"
        path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        paths[cat] = str(path)
        logger.info("Wrote %s n=%d top50", path, len(out_rows))
    return paths


def fuse_test_submit(args, cats: Sequence[str]) -> Dict[str, Any]:
    """Fuse scores on test and write CodaLab predict JSONs (no recall metrics)."""
    out_dir = Path(args.out_dir)
    mode = getattr(args, "fuse_mode", None) or getattr(args, "fusion_mode", "percat")
    rankings: Dict[str, List[List[str]]] = {}
    weight_log = {}
    for cat in cats:
        comp = _load_score(_score_npz(out_dir, "composer", cat))
        cr = _load_score(_score_npz(out_dir, "cr", cat))
        s_hbi, sim, index_names, refs, tgts = _hbi_pool_scores(
            out_dir, cat, comp["k_sorted"], comp["reference_names"], comp["target_names"],
        )
        h = zscore_rows(s_hbi)
        c = zscore_rows(comp["pred_logits"])
        r = zscore_rows(cr["pred_logits"])
        if mode == "frozen":
            a, b, g = FROZEN
        else:
            a, b, g = TEST_PERCAT_WEIGHTS[cat]
        weight_log[cat] = {"a": a, "b": b, "g": g}
        fused = a * h + b * c + g * r
        rankings[cat] = _merged_ranking(fused, comp["k_sorted"], sim, index_names, refs, top_n=50)
        logger.info("FIQ test fuse %s weights=(%.2f,%.2f,%.2f) n=%d", cat, a, b, g, len(refs))

    paths = write_codalab_predict(out_dir, cats, rankings, split="test")
    # Also zip for CodaLab upload
    import zipfile

    zip_path = out_dir / "codalab_submit" / "fiq_test_predict.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for cat, p in paths.items():
            zf.write(p, arcname=f"{cat}.predict.json")
    meta = {
        "split": "test",
        "fusion_mode": mode,
        "weights": weight_log,
        "k_rerank": int(args.k_rerank),
        "predict_json": paths,
        "zip": str(zip_path),
        "note": "Upload zip or the three *.predict.json to FashionIQ CodaLab test phase.",
    }
    meta_path = out_dir / "codalab_submit" / "submission_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("CodaLab zip -> %s", zip_path)
    return meta


def fuse_all(args, cats: Sequence[str]) -> Dict[str, Any]:
    if getattr(args, "split", "val") == "test":
        return fuse_test_submit(args, cats)
    out_dir = Path(args.out_dir)
    a, b, g = FROZEN
    per_cat = {}
    for cat in cats:
        comp = _load_score(_score_npz(out_dir, "composer", cat))
        cr = _load_score(_score_npz(out_dir, "cr", cat))
        if comp["reference_names"] != cr["reference_names"] or comp["target_names"] != cr["target_names"]:
            raise RuntimeError(f"{cat}: composer/CR query order mismatch")
        if not np.array_equal(comp["k_sorted"].astype(str), cr["k_sorted"].astype(str)):
            raise RuntimeError(f"{cat}: composer/CR top-k order mismatch")
        s_hbi, sim, index_names, refs, tgts = _hbi_pool_scores(
            out_dir, cat, comp["k_sorted"], comp["reference_names"], comp["target_names"],
        )
        labels = comp["labels"]
        s_comp = comp["pred_logits"]
        s_cr = cr["pred_logits"]
        h = zscore_rows(s_hbi)
        c = zscore_rows(s_comp)
        r = zscore_rows(s_cr)
        fused = a * h + b * c + g * r

        singles_full = {
            "coarse": _full_metrics(s_hbi, comp["k_sorted"], sim, index_names, refs, tgts),
            "composer": _full_metrics(s_comp, comp["k_sorted"], sim, index_names, refs, tgts),
            "cr": _full_metrics(s_cr, comp["k_sorted"], sim, index_names, refs, tgts),
        }
        frozen_full = _full_metrics(fused, comp["k_sorted"], sim, index_names, refs, tgts)
        singles_pool = {
            "coarse": pool_from_scores(s_hbi, labels),
            "composer": pool_from_scores(s_comp, labels),
            "cr": pool_from_scores(s_cr, labels),
            "frozen_0.25_0.30_0.45": pool_from_scores(fused, labels),
        }

        grid_best_r10 = None
        grid_best_r50 = None
        grid_best_joint = None  # maximize R@10+R@50 (tie-break R@10)
        step = float(args.grid_step)
        alphas = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
        for aa in alphas:
            for bb in alphas:
                if aa + bb > 1.0 + 1e-9:
                    continue
                gg = float(np.round(1.0 - aa - bb, 4))
                m = _full_metrics(aa * h + bb * c + gg * r, comp["k_sorted"], sim, index_names, refs, tgts)
                rec = {
                    "alpha_coarse": float(aa),
                    "beta_composer": float(bb),
                    "gamma_cr": gg,
                    **m,
                }
                if grid_best_r10 is None or rec["recall_at10"] > grid_best_r10["recall_at10"]:
                    grid_best_r10 = rec
                if grid_best_r50 is None or rec["recall_at50"] > grid_best_r50["recall_at50"]:
                    grid_best_r50 = rec
                jkey = (rec["recall_at10"] + rec["recall_at50"], rec["recall_at10"], rec["recall_at50"])
                if grid_best_joint is None or jkey > (
                    grid_best_joint["recall_at10"] + grid_best_joint["recall_at50"],
                    grid_best_joint["recall_at10"],
                    grid_best_joint["recall_at50"],
                ):
                    grid_best_joint = rec
        # Prefer joint when it does not sacrifice >0.3 R@10 vs R@10-only best
        grid_best = grid_best_r10
        if (
            grid_best_joint is not None
            and grid_best_r10 is not None
            and grid_best_joint["recall_at10"] >= grid_best_r10["recall_at10"] - 0.3
            and (grid_best_joint["recall_at10"] + grid_best_joint["recall_at50"])
            > (grid_best_r10["recall_at10"] + grid_best_r10["recall_at50"])
        ):
            grid_best = grid_best_joint

        per_cat[cat] = {
            "n_queries": len(refs),
            "n_gallery": len(index_names),
            "singles_full": singles_full,
            "frozen_0.25_0.30_0.45_full": frozen_full,
            "singles_pool": singles_pool,
            "grid_best_r10": grid_best,
            "grid_best_r10_only": grid_best_r10,
            "grid_best_r50_only": grid_best_r50,
            "grid_best_joint": grid_best_joint,
        }
        logger.info(
            "FIQ %s full R@10/50 | coarse=%.2f/%.2f composer=%.2f/%.2f CR=%.2f/%.2f "
            "frozen0.25/0.30/0.45=%.2f/%.2f grid-best=%.2f/%.2f (a=%.2f b=%.2f g=%.2f)",
            cat,
            singles_full["coarse"]["recall_at10"], singles_full["coarse"]["recall_at50"],
            singles_full["composer"]["recall_at10"], singles_full["composer"]["recall_at50"],
            singles_full["cr"]["recall_at10"], singles_full["cr"]["recall_at50"],
            frozen_full["recall_at10"], frozen_full["recall_at50"],
            grid_best["recall_at10"], grid_best["recall_at50"],
            grid_best["alpha_coarse"], grid_best["beta_composer"], grid_best["gamma_cr"],
        )

    def _mean(path_keys: Sequence[str]) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for cat in cats:
            cur = per_cat[cat]
            for k in path_keys[:-1]:
                cur = cur[k]
            row = cur if path_keys[-1] is None else cur[path_keys[-1]]
            for mk, mv in row.items():
                if isinstance(mv, (int, float)) and (
                    mk.startswith("recall") or mk.startswith("pool") or mk == "coverage"
                ):
                    acc.setdefault(mk, []).append(float(mv))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    mean = {
        "coarse": _mean(["singles_full", "coarse"]),
        "composer": _mean(["singles_full", "composer"]),
        "cr": _mean(["singles_full", "cr"]),
        "frozen_0.25_0.30_0.45": _mean(["frozen_0.25_0.30_0.45_full"]),
        "grid_best_r10": _mean(["grid_best_r10"]),
        "pool": {
            "coarse": _mean(["singles_pool", "coarse"]),
            "composer": _mean(["singles_pool", "composer"]),
            "cr": _mean(["singles_pool", "cr"]),
            "frozen_0.25_0.30_0.45": _mean(["singles_pool", "frozen_0.25_0.30_0.45"]),
        },
    }
    logger.info(
        "FIQ mean full R@10/50 | coarse=%.2f/%.2f composer=%.2f/%.2f CR=%.2f/%.2f "
        "frozen=%.2f/%.2f",
        mean["coarse"]["recall_at10"], mean["coarse"]["recall_at50"],
        mean["composer"]["recall_at10"], mean["composer"]["recall_at50"],
        mean["cr"]["recall_at10"], mean["cr"]["recall_at50"],
        mean["frozen_0.25_0.30_0.45"]["recall_at10"],
        mean["frozen_0.25_0.30_0.45"]["recall_at50"],
    )
    report = {
        "k_save": int(args.k_save),
        "k_rerank": int(args.k_rerank),
        "coarse_weights": {
            "w_vision": args.w_vision,
            "w_modified": args.w_modified,
            "w_relative": args.w_relative,
            "mu": args.mu,
        },
        "fusion_frozen": {"alpha_coarse": a, "beta_composer": b, "gamma_cr": g, "norm": "zscore_per_query"},
        "composer_ckpt": args.composer_ckpt,
        "composer_ckpt_by_cat": args.composer_ckpt_by_cat,
        "init_model": args.init_model,
        "stage1_ckpt": args.stage1_ckpt,
        "stage2_ckpt": args.stage2_ckpt,
        "mean": mean,
        "per_category": per_cat,
    }
    out_json = out_dir / "fusion_metrics.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", out_json)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FIQ 12-crop best coarse + Composer + CR, K=300")
    p.add_argument("--only", choices=["coarse", "composer", "cr", "fuse", "all"], default="all")
    p.add_argument("--init_model", type=str, default=DEFAULT_HBI)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DUAL)
    p.add_argument("--composer_ckpt", type=str, default=DEFAULT_COMPOSER)
    p.add_argument(
        "--composer_ckpt_by_cat",
        type=str,
        default=None,
        help="Optional per-category ckpts: dress=/a.pt,shirt=/b.pt,toptee=/c.pt",
    )
    p.add_argument("--stage1_ckpt", type=str, default=DEFAULT_STAGE1)
    p.add_argument("--stage2_ckpt", type=str, default=DEFAULT_STAGE2)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--pretrained_id", type=str, default="Salesforce/blip2-itm-vit-g")
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument(
        "--fuse_mode",
        type=str,
        default="percat",
        choices=["percat", "frozen"],
        help="Test submit fusion: percat=K500 val grid locks; frozen=0.25/0.30/0.45",
    )
    p.add_argument("--k_save", type=int, default=300)
    p.add_argument("--k_rerank", type=int, default=300)
    p.add_argument("--w_vision", type=float, default=0.3)
    p.add_argument("--w_modified", type=float, default=0.4)
    p.add_argument("--w_relative", type=float, default=0.3)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--removed_agg", type=str, default="max")
    p.add_argument("--grid_step", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--query_batch", type=int, default=32)
    p.add_argument("--gallery_batch", type=int, default=64)
    p.add_argument("--cr_gallery_batch", type=int, default=16)
    p.add_argument("--cr_cand_chunk", type=int, default=50)
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=fiq_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--agg_module", type=str, default="None")
    p.add_argument("--interaction", type=str, default="wti")
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--estimator", type=str, default="None")
    p.add_argument("--max_frames", type=int, default=12)
    p.add_argument("--image_resolution", type=int, default=224)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--image_encode_microbatch", type=int, default=12)
    p.add_argument("--v_rate0", type=float, default=0.25)
    p.add_argument("--v_rate1", type=float, default=0.5)
    p.add_argument("--t_rate0", type=float, default=0.25)
    p.add_argument("--t_rate1", type=float, default=0.5)
    p.add_argument("--rate", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p.add_argument("--kl", type=float, default=2.0)
    p.add_argument("--skl", type=float, default=1.0)
    p.add_argument("--fiq_image_root", type=str, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    _abort_cirr(args.out_dir, args.init_model, args.composer_ckpt or "", args.stage1_ckpt, args.stage2_ckpt)
    for _p in _parse_composer_ckpt_by_cat(args.composer_ckpt_by_cat).values():
        _abort_cirr(_p)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    args.distributed = 0
    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    logger.info(
        "FIQ full rerank only=%s split=%s K=%d coarse w=(%.1f,%.1f,%.1f) μ=%.2f cats=%s",
        args.only, args.split, args.k_rerank, args.w_vision, args.w_modified, args.w_relative, args.mu, cats,
    )
    if args.only in ("coarse", "all"):
        dump_coarse(args, cats, device)
        if args.only == "coarse":
            return
    if args.only in ("composer", "all"):
        score_composer(args, cats, device)
        if args.only == "composer":
            return
    if args.only in ("cr", "all"):
        score_cr(args, cats, device)
        if args.only == "cr":
            return
    fuse_all(args, cats)


if __name__ == "__main__":
    main()
