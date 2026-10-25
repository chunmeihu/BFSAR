"""Parse pos/neg constraint JSON from LLM output."""

from __future__ import annotations

from typing import Any, List, Sequence


def format_object_entry(entry: Any) -> str:
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        obj = str(entry.get("object") or "").strip()
        attrs = entry.get("attributes") or []
        if isinstance(attrs, str):
            attrs = [attrs]
        attr_parts = [str(a).strip() for a in attrs if str(a).strip()]
        if obj and attr_parts:
            return f"{obj} ({', '.join(attr_parts)})"
        return obj or ", ".join(attr_parts)
    return str(entry).strip()


def items_to_pos_concat(existent: Sequence[Any]) -> str:
    parts = [format_object_entry(x) for x in (existent or [])]
    parts = [p for p in parts if p]
    return ", ".join(parts)


def items_to_neg_list(nonexistent: Sequence[Any]) -> List[str]:
    out = [format_object_entry(x) for x in (nonexistent or [])]
    return [p for p in out if p]


def split_legacy_constraints(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    return [p.strip() for p in str(text).split(",") if p.strip()]


def parse_pos_neg_obj(obj: dict) -> tuple[str, List[str]]:
    existent = obj.get("existent")
    if existent is None and isinstance(obj.get("target_image"), dict):
        existent = obj["target_image"].get("existent")
    nonexistent = obj.get("nonexistent")
    if nonexistent is None and isinstance(obj.get("target_image"), dict):
        nonexistent = obj["target_image"].get("nonexistent")
    pos_concat = items_to_pos_concat(existent or [])
    neg_items = items_to_neg_list(nonexistent or [])
    return pos_concat, neg_items
