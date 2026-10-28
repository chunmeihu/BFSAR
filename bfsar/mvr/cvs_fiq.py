"""FashionIQ candidate-verification scores on a coarse top-K list."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.fiq import fiq_paths
from bfsar.fiq.eval_fgsa import compute_fiq_metrics
from bfsar.cirr.metrics import rank_by_similarity

logger = logging.getLogger("bfsar_cvs_fiq")


def recall_at_ks(
    sorted_index_names: np.ndarray,
    refs: Sequence[str],
    tgts: Sequence[str],
    ks: Sequence[int] = (10, 50, 100, 200),
) -> Dict[str, float]:
    base = compute_fiq_metrics(sorted_index_names, refs, tgts)
    n_queries = len(tgts)
    ranks = np.empty(n_queries, dtype=np.int64)
    for i in range(n_queries):
        ranking = [name for name in sorted_index_names[i] if name != refs[i]]
        try:
            ranks[i] = ranking.index(tgts[i])
        except ValueError:
            ranks[i] = len(ranking)
    out = dict(base)
    for k in ks:
        out[f"recall_at{k}"] = float((ranks < k).mean() * 100.0)
    return out


def cr_joined_caption(captions: Sequence[str]) -> str:
    parts = [str(c).strip(".?, ").strip() for c in (captions or []) if str(c).strip()]
    if len(parts) >= 2:
        return f"{parts[0].capitalize()} and {parts[1]}"
    if parts:
        return parts[0].capitalize()
    return " "


class FIQClassicImages(Dataset):
    def __init__(self, names: Sequence[str], preprocess):
        self.names = list(names)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        path = fiq_paths.image_path(name)
        if path is None:
            raise FileNotFoundError(f"Missing FIQ image: {name}")
        img = self.preprocess(Image.open(path).convert("RGB"))
        return name, img


def align_to_official(
    coarse: Dict, category: str, split: str = "val",
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    with open(fiq_paths.caption_json(category, split), "r", encoding="utf-8") as f:
        official = json.load(f)
    # Test (no target): rows already follow official caption order.
    if not official or "target" not in official[0]:
        if len(coarse["refs"]) != len(official):
            raise RuntimeError(
                f"{category}/{split}: queries {len(coarse['refs'])} != official {len(official)}"
            )
        return (
            np.asarray(coarse["sim"], dtype=np.float32),
            list(coarse["index_names"]),
            list(coarse["refs"]),
            list(coarse["tgts"]),
        )
    key_to_row = {fiq_paths.pair_key(r, t): i for i, (r, t) in enumerate(zip(coarse["refs"], coarse["tgts"]))}
    index_names = coarse["index_names"]
    name_arr = np.array(index_names)
    sim = coarse["sim"]
    aligned = []
    refs, tgts = [], []
    n_fallback = 0
    for row in official:
        ref = str(row["candidate"])
        tgt = str(row["target"])
        refs.append(ref)
        tgts.append(tgt)
        key = fiq_paths.pair_key(ref, tgt)
        if key in key_to_row:
            aligned.append(sim[key_to_row[key]])
        else:
            n_fallback += 1
            order = np.argsort(name_arr)
            scores = np.full(len(index_names), -1e6, dtype=np.float32)
            scores[order] = np.linspace(0.0, -1.0, num=len(index_names), dtype=np.float32)
            if ref in {n: i for i, n in enumerate(index_names)}:
                scores[index_names.index(ref)] = -1e9
            aligned.append(scores)
    if n_fallback:
        logger.warning("%s official queries without BFSAR row: %d (alpha fallback)", category, n_fallback)
    return np.stack(aligned, axis=0), index_names, refs, tgts


def build_topk_payload(
    sim: np.ndarray,
    index_names: List[str],
    refs: List[str],
    tgts: List[str],
    category: str,
    k_save: int,
    split: str = "val",
) -> Dict:
    sorted_names = rank_by_similarity(sim, index_names)
    n_q, n_g = sorted_names.shape
    no_ref = []
    labels = []
    for i in range(n_q):
        ranking = [str(n) for n in sorted_names[i] if str(n) != str(refs[i])]
        if len(ranking) < k_save:
            raise ValueError(f"{category} query {i}: ranking {len(ranking)} < k_save {k_save}")
        no_ref.append(ranking[:k_save])
        labels.append([n == str(tgts[i]) for n in ranking[:k_save]])
    return {
        "sorted_index_names": np.array(no_ref),
        "labels": torch.tensor(np.asarray(labels, dtype=bool)),
        "target_names": list(tgts),
        "index_names": list(index_names),
        "dress_types": category,
        "split": split,
        "source": "coarse",
        "reference_names": list(refs),
    }


def extract_blip_index_cpu(model_stage2, names: List[str], preprocess, batch_size: int, num_workers: int, device):
    ds = FIQClassicImages(names, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    chunks = []
    out_names = []
    model_stage2.eval()
    with torch.no_grad():
        for batch_names, images in tqdm(loader, desc="BLIP gallery embed"):
            images = images.to(device, non_blocking=True)
            feats = model_stage2.img_embed(images)
            chunks.append(feats.detach().cpu())
            out_names.extend(list(batch_names))
    return torch.cat(chunks, dim=0), out_names


def rerank_category(
    payload: Dict,
    model_stage1,
    model_stage2,
    preprocess,
    category: str,
    k_rerank: int,
    device,
    num_workers: int,
    gallery_batch: int,
    cand_chunk: int = 50,
    split: str = "val",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    split = str(payload.get("split") or split)
    with open(fiq_paths.caption_json(category, split), "r", encoding="utf-8") as f:
        official = json.load(f)
    refs = [str(x) for x in payload["reference_names"]]
    tgts = [str(x) for x in payload["target_names"]]
    # Val: map by (ref, tgt). Test: no target + duplicate candidates → row order.
    if official and "target" in official[0]:
        cap_by_pair = {
            (str(row["candidate"]), str(row["target"])): cr_joined_caption(row.get("captions") or [])
            for row in official
        }
        captions = []
        missing_cap = 0
        for r, t in zip(refs, tgts):
            cap = cap_by_pair.get((r, t))
            if cap is None:
                missing_cap += 1
                captions.append(" ")
            else:
                captions.append(cap)
        if missing_cap:
            raise RuntimeError(
                f"{category}: {missing_cap}/{len(refs)} payload pairs missing from official captions"
            )
    else:
        if len(official) != len(refs):
            raise RuntimeError(
                f"{category}/{split}: official {len(official)} != payload {len(refs)}"
            )
        for i, row in enumerate(official):
            if str(row["candidate"]) != refs[i]:
                raise RuntimeError(
                    f"{category}/{split}: candidate order mismatch at {i}: "
                    f"{row['candidate']} vs {refs[i]}"
                )
        captions = [cr_joined_caption(row.get("captions") or []) for row in official]

    index_names = list(payload["index_names"])
    topk = np.array(payload["sorted_index_names"], dtype=object)[:, :k_rerank]
    labels = payload["labels"].numpy()[:, :k_rerank] if torch.is_tensor(payload["labels"]) else np.asarray(payload["labels"])[:, :k_rerank]

    gallery_feats, feat_names = extract_blip_index_cpu(
        model_stage2, index_names, preprocess, gallery_batch, num_workers, device,
    )
    name_to_i = {n: i for i, n in enumerate(feat_names)}
    name_to_feat = {}
    for n, i in name_to_i.items():
        name_to_feat[n] = gallery_feats[i]

    # Test has no labels: always score. Val keeps previous in-pool gate.
    force_score = split == "test" or not labels.any()

    pred = []
    model_stage1.eval()
    model_stage2.eval()
    for i in tqdm(range(len(refs)), desc=f"CR rerank {category}"):
        in_pool = force_score or bool(labels[i].any())
        cand_names = [str(x) for x in list(topk[i])]
        if in_pool:
            ref_feat = name_to_feat[refs[i]].unsqueeze(0).to(device)
            z_t = model_stage1.img_txt_fusion(
                ref_feat, None, [captions[i]], train=False, return_raw=True
            )
            parts = []
            chunk = max(int(cand_chunk), 1)
            for s in range(0, len(cand_names), chunk):
                e = min(s + chunk, len(cand_names))
                cand = torch.stack([name_to_feat[n] for n in cand_names[s:e]]).to(device)
                part = model_stage2.img_txt_fusion_val(z_t, cand, [captions[i]]).detach().cpu().numpy()
                parts.append(np.asarray(part).reshape(-1))
            logits = np.concatenate(parts, axis=0).astype(np.float32)
        else:
            logits = np.full((k_rerank,), -1e5, dtype=np.float32)
        pred.append(logits)

    pred = np.stack(pred, axis=0)
    rerank_idx = np.argsort(-pred, axis=1)
    reranked = np.take_along_axis(topk, rerank_idx, axis=1)
    pool_hits = np.take_along_axis(labels, rerank_idx, axis=1)

    def r_at(arr, k):
        k = min(k, arr.shape[1])
        return float(arr[:, :k].sum() / len(arr) * 100.0)

    pool_metrics = {
        "coverage": float(labels.any(axis=1).mean() * 100.0),
        "pool_recall_at10": r_at(pool_hits, 10),
        "pool_recall_at50": r_at(pool_hits, 50),
        "pool_recall_at100": r_at(pool_hits, 100),
        "pool_recall_at300": r_at(pool_hits, 300),
    }
    return reranked, pred, pool_metrics


