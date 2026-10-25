"""Full CIRR val evaluation: text query vs image gallery."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from HBI.cirr.metrics_cirr_hbi import compute_cirr_metrics, rank_by_similarity

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_gallery(model, gallery_loader, device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    if hasattr(model, "module"):
        model = model.module
    model.eval()

    feats, masks, names = [], [], []
    for crops, crop_mask, name, _ in tqdm(gallery_loader, desc="encode gallery"):
        crops = crops.to(device, non_blocking=True)
        crop_mask = crop_mask.to(device, non_blocking=True)
        b = crops.size(0)
        flat = crops.view(b * crops.size(1), *crops.shape[2:])
        flat_mask = crop_mask.view(b * crop_mask.size(1), -1) if crop_mask.dim() > 2 else crop_mask
        # crops: B x K x 3 x H x W
        vf = model.get_video_feat(flat.view(b, crops.size(1), *crops.shape[2:]), crop_mask)
        feats.append(vf.cpu())
        masks.append(crop_mask.cpu())
        names.extend(list(name))
    return torch.cat(feats, dim=0), torch.cat(masks, dim=0), names


@torch.no_grad()
def encode_queries(model, query_loader, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list, list, list]:
    if hasattr(model, "module"):
        model = model.module
    model.eval()

    text_feats, cls_list, text_masks = [], [], []
    reference_names, target_names, group_members = [], [], []

    for batch in tqdm(query_loader, desc="encode queries"):
        text_ids, text_mask, ref_names, tgt_names, groups, _ = batch
        text_ids = text_ids.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        tf, cls = model.get_text_feat(text_ids, text_mask)
        text_feats.append(tf.cpu())
        cls_list.append(cls.cpu())
        text_masks.append(text_mask.cpu())
        reference_names.extend(list(ref_names))
        target_names.extend(list(tgt_names))
        for members in groups:
            group_members.append(list(members))
    return (
        torch.cat(text_feats, dim=0),
        torch.cat(cls_list, dim=0),
        torch.cat(text_masks, dim=0),
        reference_names,
        target_names,
        group_members,
    )


def _similarity_matrix(
    model,
    text_feats,
    cls_feats,
    text_masks,
    gallery_feats,
    gallery_masks,
    device,
    query_batch: int = 32,
    gallery_batch: int = 64,
) -> np.ndarray:
    if hasattr(model, "module"):
        model = model.module
    n_q = text_feats.size(0)
    n_g = gallery_feats.size(0)
    sim = np.zeros((n_q, n_g), dtype=np.float32)

    for qs in range(0, n_q, query_batch):
        qe = min(qs + query_batch, n_q)
        t_feat = text_feats[qs:qe].to(device)
        cls = cls_feats[qs:qe].to(device)
        t_mask = text_masks[qs:qe].to(device)
        row = []
        for gs in range(0, n_g, gallery_batch):
            ge = min(gs + gallery_batch, n_g)
            v_feat = gallery_feats[gs:ge].to(device)
            v_mask = gallery_masks[gs:ge].to(device)
            logits = model.get_similarity_logits(t_feat, cls, v_feat, t_mask, v_mask, shaped=True)
            row.append(logits.cpu().numpy())
        sim[qs:qe] = np.concatenate(row, axis=1)
    return sim


@torch.no_grad()
def evaluate_cirr_val(
    model,
    query_loader,
    gallery_loader,
    device,
    query_batch: int = 32,
    gallery_batch: int = 64,
) -> Dict[str, float]:
    gallery_feats, gallery_masks, index_names = encode_gallery(model, gallery_loader, device)
    text_feats, cls_feats, text_masks, ref_names, tgt_names, group_members = encode_queries(
        model, query_loader, device)

    sim = _similarity_matrix(
        model, text_feats, cls_feats, text_masks,
        gallery_feats, gallery_masks, device,
        query_batch=query_batch, gallery_batch=gallery_batch,
    )
    sorted_names = rank_by_similarity(sim, index_names)
    metrics = compute_cirr_metrics(sorted_names, ref_names, tgt_names, group_members)
    logger.info(
        "CIRR val: R@1=%.2f R@5=%.2f R@10=%.2f R@50=%.2f | "
        "G-R@1=%.2f G-R@2=%.2f G-R@3=%.2f",
        metrics["recall_at1"], metrics["recall_at5"], metrics["recall_at10"], metrics["recall_at50"],
        metrics["group_recall_at1"], metrics["group_recall_at2"], metrics["group_recall_at3"],
    )
    return metrics
