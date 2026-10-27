"""CIRR validation metrics (R@K and Group R@K)."""

from __future__ import absolute_import, division, print_function, unicode_literals

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


def compute_cirr_metrics(
    sorted_index_names: np.ndarray,
    reference_names: Sequence[str],
    target_names: Sequence[str],
    group_members: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """
    Args:
        sorted_index_names: [num_queries, gallery_size] ranked image names (reference NOT removed yet)
    """
    n_queries = len(target_names)
    n_gallery = sorted_index_names.shape[1]

    ref_expanded = np.repeat(np.array(reference_names), n_gallery).reshape(n_queries, n_gallery)
    ref_mask = sorted_index_names != ref_expanded
    sorted_no_ref = sorted_index_names[ref_mask].reshape(n_queries, n_gallery - 1)

    target_expanded = np.repeat(np.array(target_names), n_gallery - 1).reshape(n_queries, -1)
    labels = sorted_no_ref == target_expanded

    group_members_arr = np.array(group_members, dtype=object)
    if group_members_arr.ndim != 2:
        raise ValueError(
            f"group_members must be [num_queries, group_size], got shape {group_members_arr.shape}"
        )
    group_mask = (
        (sorted_no_ref[..., None] == group_members_arr[:, None, :]).sum(-1).astype(bool)
    )
    group_labels = labels[group_mask].reshape(n_queries, -1)

    def recall_at(labels_arr, k):
        return float(torch.sum(labels_arr[:, :k]).item() / len(labels_arr) * 100)

    return {
        "recall_at1": recall_at(torch.tensor(labels), 1),
        "recall_at5": recall_at(torch.tensor(labels), 5),
        "recall_at10": recall_at(torch.tensor(labels), 10),
        "recall_at50": recall_at(torch.tensor(labels), 50),
        "group_recall_at1": recall_at(torch.tensor(group_labels), 1),
        "group_recall_at2": recall_at(torch.tensor(group_labels), 2),
        "group_recall_at3": recall_at(torch.tensor(group_labels), 3),
    }


def compute_cirr_metrics_no_ref(
    sorted_no_ref: np.ndarray,
    reference_names: Sequence[str],
    target_names: Sequence[str],
    group_members: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Metrics when rankings already exclude the reference image per query."""
    n_queries = len(target_names)
    n_gallery = sorted_no_ref.shape[1]

    target_expanded = np.repeat(np.array(target_names), n_gallery).reshape(n_queries, -1)
    labels = sorted_no_ref == target_expanded

    group_members_arr = np.array(group_members, dtype=object)
    group_mask = (
        (sorted_no_ref[..., None] == group_members_arr[:, None, :]).sum(-1).astype(bool)
    )
    group_labels = labels[group_mask].reshape(n_queries, -1)

    def recall_at(labels_arr, k):
        return float(torch.sum(labels_arr[:, :k]).item() / len(labels_arr) * 100)

    return {
        "recall_at1": recall_at(torch.tensor(labels), 1),
        "recall_at5": recall_at(torch.tensor(labels), 5),
        "recall_at10": recall_at(torch.tensor(labels), 10),
        "recall_at50": recall_at(torch.tensor(labels), 50),
        "group_recall_at1": recall_at(torch.tensor(group_labels), 1),
        "group_recall_at2": recall_at(torch.tensor(group_labels), 2),
        "group_recall_at3": recall_at(torch.tensor(group_labels), 3),
    }


def rank_by_similarity(
    sim_matrix: np.ndarray,
    index_names: List[str],
) -> np.ndarray:
    """sim_matrix: [num_queries, num_gallery] higher is better."""
    order = np.argsort(-sim_matrix, axis=1)
    return np.array(index_names)[order]
