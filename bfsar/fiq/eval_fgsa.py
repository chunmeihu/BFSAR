"""Fashion-IQ val evaluation: per-category gallery, mean R@10/50/100/200."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from bfsar.cirr.eval_fgsa import _similarity_matrix, encode_gallery
from bfsar.cirr.metrics import rank_by_similarity

logger = logging.getLogger("fiq_hbi")


@torch.no_grad()
def encode_fiq_queries(model, query_loader, device):
    if hasattr(model, "module"):
        model = model.module
    model.eval()

    text_feats, cls_list, text_masks = [], [], []
    reference_names, target_names, categories = [], [], []

    for batch in tqdm(query_loader, desc="encode queries"):
        text_ids, text_mask, ref_names, tgt_names, cats, _ = batch
        text_ids = text_ids.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        tf, cls = model.get_text_feat(text_ids, text_mask)
        text_feats.append(tf.cpu())
        cls_list.append(cls.cpu())
        text_masks.append(text_mask.cpu())
        reference_names.extend(list(ref_names))
        target_names.extend(list(tgt_names))
        categories.extend(list(cats))
    return (
        torch.cat(text_feats, dim=0),
        torch.cat(cls_list, dim=0),
        torch.cat(text_masks, dim=0),
        reference_names,
        target_names,
        categories,
    )


def compute_fiq_metrics(
    sorted_index_names: np.ndarray,
    reference_names: Sequence[str],
    target_names: Sequence[str],
) -> Dict[str, float]:
    """CIRR-style: drop the reference (candidate) image, then R@10/50/100/200."""
    n_queries = len(target_names)
    ranks = np.empty(n_queries, dtype=np.int64)
    n_missing_target = 0
    n_ref_in_gallery = 0
    for i in range(n_queries):
        ranking = [name for name in sorted_index_names[i] if name != reference_names[i]]
        n_ref_in_gallery += int(len(ranking) < sorted_index_names.shape[1])
        try:
            ranks[i] = ranking.index(target_names[i])
        except ValueError:
            ranks[i] = len(ranking)
            n_missing_target += 1

    def recall_at(k: int) -> float:
        return float((ranks < k).mean() * 100.0)

    return {
        "recall_at10": recall_at(10),
        "recall_at50": recall_at(50),
        "recall_at100": recall_at(100),
        "recall_at200": recall_at(200),
        "recall_at300": recall_at(300),
        "recall_at400": recall_at(400),
        "recall_at500": recall_at(500),
        "n_queries": float(n_queries),
        "n_missing_target": float(n_missing_target),
        "n_ref_in_gallery": float(n_ref_in_gallery),
    }


@torch.no_grad()
def evaluate_fiq_category(
    model,
    query_loader,
    gallery_loader,
    device,
    category: str,
    query_batch: int = 32,
    gallery_batch: int = 64,
) -> Dict[str, float]:
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    text_feats, cls_feats, text_masks, ref_names, tgt_names, _ = encode_fiq_queries(
        model, query_loader, device
    )
    sim = _similarity_matrix(
        model,
        text_feats,
        cls_feats,
        text_masks,
        gallery_feats,
        gallery_masks,
        device,
        query_batch=query_batch,
        gallery_batch=gallery_batch,
    )
    sorted_names = rank_by_similarity(sim, index_names)
    metrics = compute_fiq_metrics(sorted_names, ref_names, tgt_names)
    logger.info(
        "FIQ val %s: R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f queries=%d gallery=%d missing_tgt=%d",
        category,
        metrics["recall_at10"],
        metrics["recall_at50"],
        metrics["recall_at100"],
        metrics["recall_at200"],
        int(metrics["n_queries"]),
        len(index_names),
        int(metrics["n_missing_target"]),
    )
    return metrics


@torch.no_grad()
def evaluate_fiq_val(
    model,
    category_loaders: Dict[str, Tuple],
    device,
    query_batch: int = 32,
    gallery_batch: int = 64,
) -> Dict[str, float]:
    """Official FIQ: per-category R@K, then unweighted mean over categories."""
    logger.info("Starting FIQ val over %d categories", len(category_loaders))
    per_cat: Dict[str, Dict[str, float]] = {}
    for category, (query_loader, gallery_loader) in category_loaders.items():
        logger.info("Encoding FIQ val category=%s", category)
        per_cat[category] = evaluate_fiq_category(
            model,
            query_loader,
            gallery_loader,
            device,
            category,
            query_batch=query_batch,
            gallery_batch=gallery_batch,
        )

    cats = list(per_cat.keys())
    mean_r10 = float(np.mean([per_cat[c]["recall_at10"] for c in cats]))
    mean_r50 = float(np.mean([per_cat[c]["recall_at50"] for c in cats]))
    mean_r100 = float(np.mean([per_cat[c]["recall_at100"] for c in cats]))
    mean_r200 = float(np.mean([per_cat[c]["recall_at200"] for c in cats]))
    metrics: Dict[str, float] = {
        "recall_at10": mean_r10,
        "recall_at50": mean_r50,
        "recall_at100": mean_r100,
        "recall_at200": mean_r200,
    }
    for cat, cat_m in per_cat.items():
        metrics[f"{cat}_recall_at10"] = cat_m["recall_at10"]
        metrics[f"{cat}_recall_at50"] = cat_m["recall_at50"]
        metrics[f"{cat}_recall_at100"] = cat_m["recall_at100"]
        metrics[f"{cat}_recall_at200"] = cat_m["recall_at200"]
        metrics[f"{cat}_n_queries"] = cat_m["n_queries"]

    logger.info(
        "FIQ val mean: R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f | %s",
        mean_r10,
        mean_r50,
        mean_r100,
        mean_r200,
        " ".join(
            f"{c}:R@10={per_cat[c]['recall_at10']:.2f}/R@50={per_cat[c]['recall_at50']:.2f}"
            f"/R@100={per_cat[c]['recall_at100']:.2f}/R@200={per_cat[c]['recall_at200']:.2f}"
            for c in cats
        ),
    )
    return metrics
