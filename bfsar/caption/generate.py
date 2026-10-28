"""Regenerate target captions Tc and vision-side captions Tv.

The released JSON files are already under data/. This script writes the same
paths from the official annotations and the prompts in data/prompts/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bfsar.caption.qwen_vllm import Qwen3VLEngine

# Target captions follow MCoT-MVS. Vision-side captions use the model from this project.
TARGET_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
VISION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
from bfsar.cirr import cirr_paths
from bfsar.fiq import fiq_paths

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "data" / "prompts"


def _read_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _parse_json_obj(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    blob = re.search(r"\{[\s\S]*\}", text)
    if not blob:
        return None
    try:
        obj = json.loads(blob.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _as_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _parse_target(raw: str, fallback: str) -> dict:
    obj = _parse_json_obj(raw) or {}
    caption = _as_text(obj.get("ModifiedImageCaption") or obj.get("target_query"))
    return {
        "ModifiedImageCaption": caption or fallback or " ",
        "Retained": _as_text(obj.get("Retained") or obj.get("RetainedContent")),
        "Removed": _as_text(obj.get("Removed") or obj.get("DeletedContent")),
    }


def _parse_vision(raw: str, fallback: str) -> str:
    obj = _parse_json_obj(raw) or {}
    for key in ("target_query", "query", "caption", "target_caption"):
        text = _as_text(obj.get(key))
        if text:
            return text
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    return lines[-1] if lines else fallback


def _cirr_file_split(split: str) -> str:
    if split in ("test", "test1"):
        return "test1"
    if split not in ("train", "val"):
        raise ValueError("CIRR split must be train, val, or test")
    return split


def _load_cirr_rows(split: str) -> List[dict]:
    file_split = _cirr_file_split(split)
    with open(cirr_paths.caption_json(file_split), "r", encoding="utf-8") as f:
        triplets = json.load(f)
    with open(cirr_paths.split_json(file_split), "r", encoding="utf-8") as f:
        name_to_rel = json.load(f)
    rows = []
    for item in triplets:
        reference = str(item["reference"])
        if item.get("target_hard"):
            target = str(item["target_hard"])
            key = cirr_paths.pair_key(reference, target)
        else:
            target = ""
            key = cirr_paths.test1_llm_key(item.get("pairid"), reference)
        rows.append(
            {
                "key": key,
                "reference": reference,
                "target": target,
                "pairid": item.get("pairid"),
                "instruction": str(item.get("caption") or "").strip(),
                "image": cirr_paths.image_path(name_to_rel, reference),
            }
        )
    return rows


def _load_fiq_rows(split: str, categories: Sequence[str]) -> List[dict]:
    if split not in ("train", "val", "test"):
        raise ValueError("FashionIQ split must be train, val, or test")
    rows = []
    for category in categories:
        with open(fiq_paths.caption_json(category, split), "r", encoding="utf-8") as f:
            official = json.load(f)
        for index, item in enumerate(official):
            reference = str(item["candidate"])
            if item.get("target") is not None and str(item["target"]).strip():
                target = str(item["target"])
            else:
                target = f"__q{index}__"
            captions = item.get("captions") or []
            instruction = " ".join(str(text).strip() for text in captions if str(text).strip())
            image = fiq_paths.image_path(reference)
            if image is None:
                raise FileNotFoundError(reference)
            rows.append(
                {
                    "key": fiq_paths.pair_key(reference, target),
                    "category": category,
                    "index": index,
                    "reference": reference,
                    "target": target,
                    "captions": captions,
                    "instruction": instruction,
                    "image": image,
                    "official": item,
                }
            )
    return rows


def _generate_batches(engine: Qwen3VLEngine, rows: Sequence[dict], template: str, max_tokens: int, batch_size: int):
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start : start + batch_size])
        prompts = [template.replace("[[MODIFICATION]]", row["instruction"] or " ") for row in chunk]
        raws = engine.generate_batch(
            prompts,
            image_paths=[row["image"] for row in chunk],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        print(f"[generate] {min(start + batch_size, len(rows))}/{len(rows)}")
        yield chunk, raws


def generate_cirr(engine, split: str, kind: str, batch_size: int, force: bool) -> None:
    rows = _load_cirr_rows(split)
    file_split = _cirr_file_split(split)
    if kind in ("target", "both"):
        path = Path(cirr_paths.llm_json(file_split))
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not force else {}
        pending = [
            row for row in rows
            if force or not _as_text(current.get(row["key"], {}).get("ModifiedImageCaption"))
        ]
        print(f"[cirr {file_split} target] todo={len(pending)} / {len(rows)}")
        template = _read_prompt("target_caption.txt")
        for chunk, raws in _generate_batches(engine, pending, template, 512, batch_size):
            for row, raw in zip(chunk, raws):
                record = dict(current.get(row["key"], {}))
                record.update(_parse_target(raw, row["instruction"]))
                current[row["key"]] = record
            _atomic_write_json(path, current)
    if kind in ("vision", "both"):
        path = Path(cirr_paths.vision_dual_json(file_split))
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not force else {}
        pending = [
            row for row in rows
            if force or not _as_text(current.get(row["key"], {}).get("vision_side"))
        ]
        print(f"[cirr {file_split} vision] todo={len(pending)} / {len(rows)}")
        template = _read_prompt("vision_caption.txt")
        for chunk, raws in _generate_batches(engine, pending, template, 256, batch_size):
            for row, raw in zip(chunk, raws):
                record = dict(current.get(row["key"], {}))
                record.update(
                    {
                        "reference": row["reference"],
                        "instruction": row["instruction"],
                        "vision_side": _parse_vision(raw, row["instruction"]),
                        "vision_raw": raw,
                    }
                )
                if row["target"]:
                    record["target_hard"] = row["target"]
                if row["pairid"] is not None:
                    record["pairid"] = row["pairid"]
                current[row["key"]] = record
            _atomic_write_json(path, current)


def generate_fiq(engine, split: str, kind: str, categories: Sequence[str], batch_size: int, force: bool) -> None:
    rows = _load_fiq_rows(split, categories)
    if kind in ("target", "both"):
        by_cat: Dict[str, List[dict]] = {category: [] for category in categories}
        for row in rows:
            by_cat[row["category"]].append(row)
        for category, cat_rows in by_cat.items():
            path = Path(fiq_paths.llm_json(category, split))
            existing = {}
            if path.is_file() and not force:
                for item in json.loads(path.read_text(encoding="utf-8")):
                    target = str(item.get("target") or "")
                    existing[fiq_paths.pair_key(str(item["candidate"]), target)] = item
            pending = [
                row for row in cat_rows
                if force or not _as_text(existing.get(row["key"], {}).get("ModifiedImageCaption"))
            ]
            print(f"[fashioniq {split} {category} target] todo={len(pending)} / {len(cat_rows)}")
            template = _read_prompt("target_caption.txt")
            parsed = {}

            def _write_fiq_target() -> None:
                written = []
                for row in cat_rows:
                    record = {
                        "candidate": row["reference"],
                        "target": row["target"],
                        "captions": row["captions"],
                    }
                    previous = existing.get(row["key"], {})
                    record["ModifiedImageCaption"] = previous.get("ModifiedImageCaption", "")
                    record["Retained"] = previous.get("Retained", "")
                    record["Removed"] = previous.get("Removed", "")
                    if row["key"] in parsed:
                        record.update(parsed[row["key"]])
                    written.append(record)
                _atomic_write_json(path, written)

            for chunk, raws in _generate_batches(engine, pending, template, 512, batch_size):
                for row, raw in zip(chunk, raws):
                    parsed[row["key"]] = _parse_target(raw, row["instruction"])
                _write_fiq_target()
    if kind in ("vision", "both"):
        path = Path(fiq_paths.vision_dual_json(split))
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not force else {}
        pending = [
            row for row in rows
            if force or not _as_text(current.get(row["key"], {}).get("vision_side"))
        ]
        print(f"[fashioniq {split} vision] todo={len(pending)} / {len(rows)}")
        template = _read_prompt("vision_caption.txt")
        for chunk, raws in _generate_batches(engine, pending, template, 256, batch_size):
            for row, raw in zip(chunk, raws):
                current[row["key"]] = {
                    "category": row["category"],
                    "reference": row["reference"],
                    "target_hard": row["target"],
                    "instruction": row["instruction"],
                    "vision_side": _parse_vision(raw, row["instruction"]),
                    "vision_raw": raw,
                }
            _atomic_write_json(path, current)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate target captions and vision-side captions")
    parser.add_argument("--dataset", required=True, choices=["cirr", "fashioniq"])
    parser.add_argument("--split", choices=["train", "val", "test"], help="default: train, val, and test")
    parser.add_argument("--kind", default="both", choices=["target", "vision", "both"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--categories", nargs="+", default=list(fiq_paths.CATEGORIES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    return parser.parse_args()


def _run(engine, args, split: str, kind: str) -> None:
    if args.dataset == "cirr":
        generate_cirr(engine, split, kind, args.batch_size, args.force)
    else:
        generate_fiq(engine, split, kind, args.categories, args.batch_size, args.force)


def _release(engine) -> None:
    del engine
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    splits = [args.split] if args.split else ["train", "val", "test"]
    if args.kind in ("target", "both"):
        engine = Qwen3VLEngine(TARGET_MODEL, gpu_memory_utilization=args.gpu_memory_utilization)
        for split in splits:
            _run(engine, args, split, "target")
        _release(engine)
    if args.kind in ("vision", "both"):
        engine = Qwen3VLEngine(VISION_MODEL, gpu_memory_utilization=args.gpu_memory_utilization)
        for split in splits:
            _run(engine, args, split, "vision")
        _release(engine)


if __name__ == "__main__":
    main()
