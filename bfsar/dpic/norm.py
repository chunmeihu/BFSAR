#!/usr/bin/env python3
"""FIQ coarse: training-free feature composition (ref visual + text CLS).

q = normalize(α · v_ref + β · t_cls)
S_compose = cos(q, v_gallery_pooled)

Also reports crop–crop max ref↔cand similarity as a finer S_ref.
Fuses with cached text FGSA branches from refsim_fusion_v1 when present.

Does NOT retrain. Does NOT touch CIRR. Writes under
output/fiq_vision_desc/compose_fusion_v1/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.cirr.eval_fgsa import encode_gallery
from bfsar.cirr.modeling_fgsa import FGSAModel
from bfsar.dpic.encode_cirr import encode_three
from bfsar.fiq import fiq_paths
from bfsar.fiq.fiq_dataloader_builders import build_fiq_gallery_dataloader
from bfsar.dpic.encode_fiq import (
    FIQTripleVisionDataset,
    collate_fn,
    fiq_metrics_from_sim,
)
from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer

logger = logging.getLogger("fiq_compose_fusion")

DEFAULT_CKPT = str(ROOT / "output/fiq_retrieval/pytorch_model.bin.best.5")
DEFAULT_DUAL = str(ROOT / "data/fashioniq/vision_desc/val_dual_targets.json")
DEFAULT_OUT = str(ROOT / "output/fiq_vision_desc/compose_fusion_v1")
DEFAULT_SCORE_CACHE = str(ROOT / "output/fiq_vision_desc/refsim_fusion_v1")
BASE_W = (0.3, 0.4, 0.3)


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


def masked_mean_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(-1)
    return (feat * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def tgt_ranks(sim: np.ndarray, index_names: Sequence[str], refs: Sequence[str], tgts: Sequence[str]) -> np.ndarray:
    name_to_i = {n: i for i, n in enumerate(index_names)}
    ranks = np.empty(len(refs), dtype=np.int32)
    for i, (r, t) in enumerate(zip(refs, tgts)):
        s = sim[i].copy()
        ri = name_to_i.get(r)
        if ri is not None:
            s[ri] = -1e9
        order = np.argsort(-s, kind="mergesort")
        ranks[i] = int(np.where(order == name_to_i[t])[0][0])
    return ranks


def r_at(ranks: np.ndarray, k: int) -> float:
    return 100.0 * float(np.mean(ranks < k))


def compose_sim(
    v_ref: np.ndarray,
    t_cls: np.ndarray,
    gallery_pooled: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """v_ref, t_cls: N x D (L2-normalized preferred); gallery: G x D."""
    # ensure norm
    vr = v_ref / np.clip(np.linalg.norm(v_ref, axis=1, keepdims=True), 1e-8, None)
    tt = t_cls / np.clip(np.linalg.norm(t_cls, axis=1, keepdims=True), 1e-8, None)
    q = alpha * vr + beta * tt
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    return (q @ gallery_pooled.T).astype(np.float32)


def crop_max_ref_sim(
    gallery_feats: torch.Tensor,
    gallery_masks: torch.Tensor,
    index_names: Sequence[str],
    refs: Sequence[str],
) -> np.ndarray:
    """For each query: max cosine over (ref_crop, cand_crop) pairs."""
    # normalize crop tokens
    feat = F.normalize(gallery_feats.float(), dim=-1)  # G x K x D
    mask = gallery_masks.float()  # G x K
    name_to_i = {str(n): i for i, n in enumerate(index_names)}
    G, K, D = feat.shape
    # Flatten valid tokens per image is awkward; use masked -inf for invalid
    out = np.zeros((len(refs), G), dtype=np.float32)
    # Precompute all crop matrix: (G*K) x D with invalid zeroed
    flat = feat.view(G * K, D)
    for qi, ref in enumerate(tqdm(refs, desc="crop-max refsim", leave=False)):
        j = name_to_i.get(str(ref))
        if j is None:
            continue
        rf = feat[j]  # K x D
        rm = mask[j] > 0.5
        rf = rf[rm]  # Kr x D
        if rf.numel() == 0:
            continue
        # (Kr x D) @ (D x G*K) -> Kr x G*K, then max over Kr and over K per gallery
        logits = torch.matmul(rf, flat.T)  # Kr x (G*K)
        logits = logits.view(rf.size(0), G, K)
        # mask invalid cand crops
        logits = logits.masked_fill(mask.unsqueeze(0) < 0.5, -1e4)
        out[qi] = logits.amax(dim=(0, 2)).cpu().numpy()
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_model", type=str, default=DEFAULT_CKPT)
    p.add_argument("--dual_json", type=str, default=DEFAULT_DUAL)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--score_cache_dir", type=str, default=DEFAULT_SCORE_CACHE)
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_words", type=int, default=77)
    p.add_argument("--batch_size_val", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--base_encoder", type=str, default="ViT-L/14")
    p.add_argument("--clip_checkpoint", type=str, default=fiq_paths.CLIP_VITL14_CHECKPOINT)
    p.add_argument("--agg_module", type=str, default="None")
    p.add_argument("--interaction", type=str, default="fgsa")
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
    p.add_argument("--lambda_b", type=float, default=1.0)
    p.add_argument("--fiq_image_root", type=str, default=None)
    p.add_argument("--do_cropmax", type=int, default=1)
    p.add_argument("--encode_only", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def encode_text_cls(model, loader, device) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    if hasattr(model, "module"):
        model = model.module
    model.eval()
    buckets = {k: [] for k in ("vision", "modified", "relative")}
    refs, tgts = [], []
    for batch in tqdm(loader, desc="encode text cls"):
        (
            v_ids, v_mask, m_ids, m_mask, r_ids, r_mask,
            ref_names, tgt_names, _groups, _,
        ) = batch
        for key, ids, mask in (
            ("vision", v_ids, v_mask),
            ("modified", m_ids, m_mask),
            ("relative", r_ids, r_mask),
        ):
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            _feat, cls = model.get_text_feat(ids, mask)
            buckets[key].append(F.normalize(cls.float(), dim=-1).cpu())
        refs.extend(list(ref_names))
        tgts.extend(list(tgt_names))
    out = {k: torch.cat(v, 0).numpy().astype(np.float32) for k, v in buckets.items()}
    return out, refs, tgts


def encode_category(model, args, tokenizer, category: str, device, out_dir: Path):
    feat_path = out_dir / f"feats_{category}.npz"
    if feat_path.exists():
        logger.info("Reuse cached feats %s", feat_path)
        return

    dataset = FIQTripleVisionDataset(
        args.dual_json, tokenizer, category=category,
        max_words=args.max_words, max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size_val, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )
    gallery_loader, n_g, _ = build_fiq_gallery_dataloader(args, split="val", category=category)
    logger.info("FIQ %s gallery=%d queries=%d", category, n_g, len(dataset))
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    text_cls, refs, tgts = encode_text_cls(model, loader, device)

    pooled = F.normalize(masked_mean_pool(gallery_feats, gallery_masks), dim=-1).cpu().numpy().astype(np.float32)
    name_to_i = {str(n): i for i, n in enumerate(index_names)}
    v_ref = np.stack([pooled[name_to_i[str(r)]] for r in refs], axis=0).astype(np.float32)

    cropmax = None
    if args.do_cropmax:
        cropmax = crop_max_ref_sim(gallery_feats.cpu(), gallery_masks.cpu(), index_names, refs)

    np.savez_compressed(
        feat_path,
        gallery_pooled=pooled,
        v_ref=v_ref,
        cls_vision=text_cls["vision"],
        cls_modified=text_cls["modified"],
        cls_relative=text_cls["relative"],
        cropmax=cropmax if cropmax is not None else np.zeros((0,), dtype=np.float32),
        index_names=np.array(index_names, dtype=object),
        refs=np.array(refs, dtype=object),
        tgts=np.array(tgts, dtype=object),
    )
    logger.info("Wrote %s", feat_path)


def load_score_cache(cache_dir: Path, category: str) -> Optional[Dict[str, np.ndarray]]:
    p = cache_dir / f"scores_{category}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "vision": d["vision"],
        "modified": d["modified"],
        "relative": d["relative"],
        "ref": d["ref"],
        "index_names": [str(x) for x in d["index_names"].tolist()],
        "refs": [str(x) for x in d["refs"].tolist()],
        "tgts": [str(x) for x in d["tgts"].tolist()],
    }


def fuse_text_scores(parts: Dict[str, np.ndarray]) -> np.ndarray:
    wv, wm, wr = BASE_W
    # zscore then weighted (same as refsim adaptive baseline)
    return (
        wv * zscore_rows(parts["vision"])
        + wm * zscore_rows(parts["modified"])
        + wr * zscore_rows(parts["relative"])
    ).astype(np.float32)


def grid_category(category: str, out_dir: Path, cache_dir: Path) -> Dict:
    feat = np.load(out_dir / f"feats_{category}.npz", allow_pickle=True)
    gallery = feat["gallery_pooled"]
    v_ref = feat["v_ref"]
    idx = [str(x) for x in feat["index_names"].tolist()]
    refs = [str(x) for x in feat["refs"].tolist()]
    tgts = [str(x) for x in feat["tgts"].tolist()]
    cropmax = feat["cropmax"]
    has_cropmax = cropmax.ndim == 2 and cropmax.shape[0] == len(refs)

    cached = load_score_cache(cache_dir, category)
    text_z = None
    if cached is not None:
        assert cached["refs"] == refs and cached["tgts"] == tgts, f"query order mismatch {category}"
        text_z = fuse_text_scores(cached)
        logger.info("%s loaded FGSA cache", category)

    singles = {}
    # text-only CLS cosine (no ref) as sanity
    for name in ("vision", "modified", "relative"):
        t = feat[f"cls_{name}"]
        sim = (t @ gallery.T).astype(np.float32)
        rk = tgt_ranks(sim, idx, refs, tgts)
        singles[f"cls_{name}"] = {"r10": r_at(rk, 10), "r50": r_at(rk, 50)}
    # pure ref pooled
    sim_ref = (v_ref @ gallery.T).astype(np.float32)
    rk = tgt_ranks(sim_ref, idx, refs, tgts)
    singles["ref_pooled"] = {"r10": r_at(rk, 10), "r50": r_at(rk, 50)}
    if has_cropmax:
        rk = tgt_ranks(cropmax, idx, refs, tgts)
        singles["ref_cropmax"] = {"r10": r_at(rk, 10), "r50": r_at(rk, 50)}
    if text_z is not None:
        rk = tgt_ranks(text_z, idx, refs, tgts)
        singles["wti_text"] = {"r10": r_at(rk, 10), "r50": r_at(rk, 50)}

    # compose grids: alpha on ref, beta on text; scan alpha in [0,1], beta=1-alpha AND independent small grid
    compose_rows = []
    text_keys = ("relative", "modified", "vision")
    alphas = [i / 20 for i in range(0, 21)]  # 0..1 step 0.05
    for tname in text_keys:
        t = feat[f"cls_{tname}"]
        for a in alphas:
            b = 1.0 - a
            sim = compose_sim(v_ref, t, gallery, a, b)
            rk = tgt_ranks(sim, idx, refs, tgts)
            compose_rows.append({
                "text": tname, "alpha": a, "beta": b,
                "r10": r_at(rk, 10), "r50": r_at(rk, 50),
            })
    compose_rows.sort(key=lambda r: (-r["r10"], -r["r50"]))

    # fuse compose with FGSA text: zscore blend
    fuse_rows = []
    if text_z is not None:
        # pick a few strong compose recipes + full alpha sweep on best text
        best_per_text = {}
        for tname in text_keys:
            cand = [r for r in compose_rows if r["text"] == tname]
            best_per_text[tname] = cand[0]
        for tname, best in best_per_text.items():
            t = feat[f"cls_{tname}"]
            sim_c = compose_sim(v_ref, t, gallery, best["alpha"], best["beta"])
            zc = zscore_rows(sim_c)
            for w_c in [i / 20 for i in range(0, 11)]:  # 0..0.5
                fused = ((1.0 - w_c) * text_z + w_c * zc).astype(np.float32)
                rk = tgt_ranks(fused, idx, refs, tgts)
                fuse_rows.append({
                    "text": tname, "alpha": best["alpha"], "beta": best["beta"],
                    "w_compose": w_c, "r10": r_at(rk, 10), "r50": r_at(rk, 50),
                })
        # also sweep alpha while fusing with fixed w_compose
        for tname in ("modified", "relative"):
            t = feat[f"cls_{tname}"]
            for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                sim_c = compose_sim(v_ref, t, gallery, a, 1.0 - a)
                zc = zscore_rows(sim_c)
                for w_c in [0.15, 0.25, 0.35, 0.45]:
                    fused = ((1.0 - w_c) * text_z + w_c * zc).astype(np.float32)
                    rk = tgt_ranks(fused, idx, refs, tgts)
                    fuse_rows.append({
                        "text": tname, "alpha": a, "beta": 1.0 - a,
                        "w_compose": w_c, "r10": r_at(rk, 10), "r50": r_at(rk, 50),
                    })
        fuse_rows.sort(key=lambda r: (-r["r10"], -r["r50"]))

    # cropmax fuse with FGSA (compare to old pooled ref)
    crop_fuse = []
    if text_z is not None and has_cropmax:
        zc = zscore_rows(cropmax)
        zp = zscore_rows(sim_ref)
        for name, z in (("cropmax", zc), ("pooled", zp)):
            for w in [i / 20 for i in range(0, 11)]:
                fused = ((1.0 - w) * text_z + w * z).astype(np.float32)
                rk = tgt_ranks(fused, idx, refs, tgts)
                crop_fuse.append({"ref_kind": name, "w": w, "r10": r_at(rk, 10), "r50": r_at(rk, 50)})
        crop_fuse.sort(key=lambda r: (-r["r10"], -r["r50"]))

    return {
        "n_queries": len(refs),
        "n_gallery": len(idx),
        "singles": singles,
        "compose_best": compose_rows[0],
        "compose_top10": compose_rows[:10],
        "compose_best_per_text": {
            t: next(r for r in compose_rows if r["text"] == t) for t in text_keys
        },
        "fuse_best": fuse_rows[0] if fuse_rows else None,
        "fuse_top10": fuse_rows[:10],
        "crop_fuse_best": crop_fuse[0] if crop_fuse else None,
        "crop_fuse_top5": crop_fuse[:5],
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if "cirr" in os.path.abspath(str(out_dir)).lower():
        raise ValueError(f"Must not write under cirr path: {out_dir}")
    if "cirr" in os.path.abspath(args.init_model).lower():
        raise ValueError(f"Must not load CIRR ckpt: {args.init_model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    need_encode = [c for c in cats if not (out_dir / f"feats_{c}.npz").exists()]

    if need_encode:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        args.device = device
        args.distributed = 0
        tokenizer = ClipTokenizer()
        model = FGSAModel(args)
        state = torch.load(args.init_model, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        miss, unexp = model.load_state_dict(cleaned, strict=False)
        logger.info("Loaded %s miss=%d unexp=%d", args.init_model, len(miss), len(unexp))
        model = model.to(device).eval()
        for cat in need_encode:
            encode_category(model, args, tokenizer, cat, device, out_dir)
        del model
        torch.cuda.empty_cache()

    if args.encode_only:
        logger.info("encode_only done")
        return

    cache_dir = Path(args.score_cache_dir)
    per = {}
    for cat in cats:
        logger.info("=== grid %s ===", cat)
        per[cat] = grid_category(cat, out_dir, cache_dir)
        logger.info(
            "%s compose_best R@10=%.2f (%s a=%.2f) | fuse_best=%s | crop_fuse=%s",
            cat,
            per[cat]["compose_best"]["r10"],
            per[cat]["compose_best"]["text"],
            per[cat]["compose_best"]["alpha"],
            None if per[cat]["fuse_best"] is None else f"{per[cat]['fuse_best']['r10']:.2f}",
            None if per[cat]["crop_fuse_best"] is None else f"{per[cat]['crop_fuse_best']['r10']:.2f}",
        )

    def mean_field(path):
        vals = []
        for c in cats:
            cur = per[c]
            for k in path.split("."):
                if cur is None:
                    return None
                cur = cur.get(k) if isinstance(cur, dict) else None
            vals.append(cur)
        if any(v is None for v in vals):
            return None
        return float(np.mean(vals))

    # shared hyperparams: pick alpha/text/w that maximize mean across cats
    # re-evaluate shared from feats
    shared_compose = []
    alphas = [i / 20 for i in range(0, 21)]
    for tname in ("relative", "modified", "vision"):
        for a in alphas:
            r10s, r50s = [], []
            for c in cats:
                feat = np.load(out_dir / f"feats_{c}.npz", allow_pickle=True)
                sim = compose_sim(
                    feat["v_ref"], feat[f"cls_{tname}"], feat["gallery_pooled"], a, 1.0 - a,
                )
                idx = [str(x) for x in feat["index_names"].tolist()]
                refs = [str(x) for x in feat["refs"].tolist()]
                tgts = [str(x) for x in feat["tgts"].tolist()]
                rk = tgt_ranks(sim, idx, refs, tgts)
                r10s.append(r_at(rk, 10)); r50s.append(r_at(rk, 50))
            shared_compose.append({
                "text": tname, "alpha": a, "beta": 1.0 - a,
                "r10": float(np.mean(r10s)), "r50": float(np.mean(r50s)),
                "per_r10": {c: r10s[i] for i, c in enumerate(cats)},
            })
    shared_compose.sort(key=lambda r: (-r["r10"], -r["r50"]))

    shared_fuse = []
    for tname in ("modified", "relative"):
        for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            for w_c in [0.0, 0.15, 0.25, 0.35, 0.45]:
                r10s, r50s = [], []
                for c in cats:
                    feat = np.load(out_dir / f"feats_{c}.npz", allow_pickle=True)
                    cached = load_score_cache(cache_dir, c)
                    text_z = fuse_text_scores(cached)
                    sim_c = compose_sim(
                        feat["v_ref"], feat[f"cls_{tname}"], feat["gallery_pooled"], a, 1.0 - a,
                    )
                    fused = ((1.0 - w_c) * text_z + w_c * zscore_rows(sim_c)).astype(np.float32)
                    idx = [str(x) for x in feat["index_names"].tolist()]
                    refs = [str(x) for x in feat["refs"].tolist()]
                    tgts = [str(x) for x in feat["tgts"].tolist()]
                    rk = tgt_ranks(fused, idx, refs, tgts)
                    r10s.append(r_at(rk, 10)); r50s.append(r_at(rk, 50))
                shared_fuse.append({
                    "text": tname, "alpha": a, "w_compose": w_c,
                    "r10": float(np.mean(r10s)), "r50": float(np.mean(r50s)),
                    "per_r10": {c: r10s[i] for i, c in enumerate(cats)},
                })
    shared_fuse.sort(key=lambda r: (-r["r10"], -r["r50"]))

    # baselines from singles mean
    def mean_single(key):
        return float(np.mean([per[c]["singles"][key]["r10"] for c in cats if key in per[c]["singles"]]))

    out = {
        "exp": "fiq_compose_fusion_v1",
        "init_model": args.init_model,
        "formula": "q = normalize(α·v_ref + β·t_cls); S = cos(q, v_gallery)",
        "mean": {
            "wti_text_r10": mean_single("wti_text") if "wti_text" in per[cats[0]]["singles"] else None,
            "ref_pooled_r10": mean_single("ref_pooled"),
            "ref_cropmax_r10": mean_single("ref_cropmax") if "ref_cropmax" in per[cats[0]]["singles"] else None,
            "shared_compose_best": shared_compose[0],
            "shared_fuse_best": shared_fuse[0] if shared_fuse else None,
            "per_cat_compose_oracle_mean_r10": float(np.mean([per[c]["compose_best"]["r10"] for c in cats])),
            "per_cat_fuse_oracle_mean_r10": float(np.mean([per[c]["fuse_best"]["r10"] for c in cats])) if per[cats[0]]["fuse_best"] else None,
        },
        "shared_compose_top10": shared_compose[:10],
        "shared_fuse_top10": shared_fuse[:10],
        "per_category": per,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_dir / "metrics.json")
    logger.info(
        "MEAN wti_text=%.2f | compose_shared=%.2f | fuse_shared=%.2f | compose_percat_best=%.2f | fuse_percat_best=%s",
        out["mean"]["wti_text_r10"] or -1,
        shared_compose[0]["r10"],
        shared_fuse[0]["r10"] if shared_fuse else -1,
        out["mean"]["per_cat_compose_oracle_mean_r10"],
        f"{out['mean']['per_cat_fuse_oracle_mean_r10']:.2f}" if out["mean"]["per_cat_fuse_oracle_mean_r10"] else "n/a",
    )


if __name__ == "__main__":
    main()
