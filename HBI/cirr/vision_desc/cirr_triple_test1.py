"""CIRR test1 queries for HBI triple−Removed coarse retrieval (no target labels)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from HBI.cirr import cirr_paths
from HBI.cirr.vision_desc.eval_hbi_triple_vision import CIRRTripleVisionDataset
from HBI.cirr.vision_desc.pos_neg_utils import split_legacy_constraints


def load_test1_triplets():
    with open(cirr_paths.caption_json("test1"), "r", encoding="utf-8") as f:
        return json.load(f)


class CIRRTripleTest1Dataset(CIRRTripleVisionDataset):
    """Same triple text fusion as val, keyed by pairid+reference for test1 LLM json."""

    def __init__(
        self,
        dual_json: Optional[str] = None,
        tokenizer=None,
        max_words: int = 77,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_words = max_words
        triplets = load_test1_triplets()
        with open(cirr_paths.llm_json("test1"), "r", encoding="utf-8") as f:
            llm = json.load(f)
        dual = {}
        if dual_json and Path(dual_json).is_file():
            with open(dual_json, "r", encoding="utf-8") as f:
                dual = json.load(f)

        self.queries = []
        for idx, t in enumerate(triplets):
            if max_samples is not None and idx >= max_samples:
                break
            ref = t["reference"]
            pairid = t["pairid"]
            llm_key = cirr_paths.test1_llm_key(pairid, ref)
            modified = str(llm.get(llm_key, {}).get("ModifiedImageCaption") or "").strip()
            relative = str(t.get("caption") or "").strip()
            if modified and not relative:
                relative = modified
            elif relative and not modified:
                modified = relative
            if not relative:
                continue
            if not modified:
                modified = relative
            dual_item = dual.get(llm_key) or {}
            vision = str(dual_item.get("vision_side") or "").strip()
            if dual_json and Path(dual_json).is_file() and not vision:
                raise ValueError(
                    f"Missing vision_side for test1 key {llm_key!r} in {dual_json}; "
                    "run: bash HBI-main/scripts/run_cirr_test1_vision_generate.sh"
                )
            if not vision:
                vision = relative
            self.queries.append(
                {
                    "idx": idx,
                    "pairid": pairid,
                    "vision": vision,
                    "modified": modified,
                    "relative": relative,
                    "reference_name": ref,
                    "target_name": ref,
                    "group_members": t["img_set"]["members"],
                }
            )
        print(f"[CIRRTripleTest1Dataset] queries={len(self.queries)}")


def load_neg_items_test1(refs: List[str], pairids: List[int]) -> tuple:
    with open(cirr_paths.llm_json("test1"), "r", encoding="utf-8") as f:
        llm = json.load(f)
    neg_items_all = []
    for ref, pid in zip(refs, pairids):
        key = cirr_paths.test1_llm_key(pid, ref)
        neg_items_all.append(split_legacy_constraints(llm.get(key, {}).get("Removed") or ""))
    has_neg = np.array([1 if items else 0 for items in neg_items_all], dtype=np.float32)
    return neg_items_all, has_neg
