#!/usr/bin/env python3
"""Train composed query matching on FashionIQ with in-batch contrastive loss."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.mvr.cqm_model import (
    ARCH_TAG,
    DEFAULT_ITM_ID,
    build_blip2_qtr,
    load_blip2_qtr_ckpt,
)
from bfsar.mvr.cqm_data_fiq import (
    FIQQTRRelativeTrain,
    fiq_qtr_collate,
    fiq_targetpad_preprocess,
    generate_randomized_fiq_caption,
)
from bfsar.mvr.cqm_rerank_fiq import rerank_all_categories

logger = logging.getLogger("train_fiq_blip2_qtr")
DEFAULT_OUT = str(ROOT / "output/cqm/fashioniq")


def _abort_cirr(*paths: str) -> None:
    for p in paths:
        if p and "cirr" in os.path.abspath(str(p)).lower():
            raise ValueError(f"FIQ QTR must not use a cirr path: {p}")


def parse_args():
    p = argparse.ArgumentParser(description="BLIP-2 QTR Stage-A FIQ cqm")
    p.add_argument("--stage", type=str, default="a", choices=["a"], help="contrastive composed-query training")
    p.add_argument("--pretrained_id", type=str, default=DEFAULT_ITM_ID)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--blip_max_epoch", type=int, default=40)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16, help="Per-GPU")
    p.add_argument("--blip_bs", type=int, default=8)
    p.add_argument("--blip_img_tune", action="store_true")
    p.add_argument("--grad_accumulation_step", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--k_val", type=int, default=100)
    p.add_argument("--validate_every", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--early_stop_patience", type=int, default=6)
    p.add_argument("--fiq_categories", type=str, default="dress,shirt,toptee")
    p.add_argument("--topk_dir", type=str, default=str(ROOT / "output/fiq_vision_desc"))
    p.add_argument("--val_max_queries", type=int, default=None)
    p.add_argument("--gallery_batch", type=int, default=8)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--reset_optimizer", action="store_true")
    # Val model selection: maximize w10*R@10 + w50*R@50 (default favors R@50).
    p.add_argument("--val_score_w10", type=float, default=0.4)
    p.add_argument("--val_score_w50", type=float, default=0.6)
    p.add_argument(
        "--global_contrastive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import datetime

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _cosine_lr(optimizer, epoch: int, max_epoch: int, init_lr: float, min_lr: float = 0.0) -> float:
    import math

    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max(max_epoch, 1))) + min_lr
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def _embed_images(model, images, blip_bs: int, grad: bool):
    raw = _unwrap(model)
    param_dtype = next(raw.parameters()).dtype
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        chunks = torch.split(images, blip_bs)
        return torch.vstack([raw.img_embed(mini).to(dtype=param_dtype) for mini in chunks])


def _gather_contrastive_batch(ref_feats, tgt_feats, captions, *, world_size: int, rank: int):
    if world_size == 1:
        return ref_feats, tgt_feats, list(captions), 0
    import torch.distributed.nn.functional as dist_nn

    ref_global = torch.cat(dist_nn.all_gather(ref_feats), dim=0)
    tgt_global = torch.cat(dist_nn.all_gather(tgt_feats), dim=0)
    cap_parts: list = [None] * world_size
    dist.all_gather_object(cap_parts, list(captions))
    caps_global = [c for part in cap_parts for c in part]
    return ref_global, tgt_global, caps_global, rank * ref_feats.size(0)


def _rebuild_optimizer(model, lr: float, weight_decay: float):
    raw = _unwrap(model)
    params = [p for p in raw.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _save_ckpt(path: Path, model, optim, scaler, epoch: int, args, best_r10: float) -> None:
    raw = _unwrap(model)
    payload = {
        "BLIP2_QTR": raw.state_dict(),
        "arch_tag": ARCH_TAG,
        "stage": args.stage,
        "epoch": epoch,
        "train": f"fiq_qtr_stage_{args.stage}",
        "dataset": "fashionIQ",
        "pretrained_id": args.pretrained_id,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else {},
        "best_pool_r10": best_r10,
        "grad_accumulation_step": args.grad_accumulation_step,
        "batch_size": args.batch_size,
        "global_contrastive": args.global_contrastive,
        "temperature": args.temperature,
        "exp_name": args.exp_name,
        "k_val": args.k_val,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _dump_exp_config(out_dir: Path, args, world_size: int) -> None:
    global_b = args.batch_size * world_size if args.global_contrastive else args.batch_size
    cfg = {
        "arch_tag": ARCH_TAG,
        "dataset": "fashionIQ",
        "description": (
            f"FIQ QTR Stage-A: Q-Former_C cqm ITC on cats={args.fiq_categories}; "
            "official dual captions with CR 4-way randomization; freeze ViT-G; "
            "val = BFSAR Top-K pool mean R@10 (over listed cats)"
        ),
        "stage": args.stage,
        "pretrained_id": args.pretrained_id,
        "batch_size_per_gpu": args.batch_size,
        "world_size": world_size,
        "global_ce_B": global_b,
        "grad_accumulation_step": args.grad_accumulation_step,
        "lr": args.lr,
        "temperature": args.temperature,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "blip_max_epoch": args.blip_max_epoch,
        "k_val": args.k_val,
        "validate_every": args.validate_every,
        "early_stop_patience": args.early_stop_patience,
        "fiq_categories": args.fiq_categories,
        "topk_dir": args.topk_dir,
        "caption": "official FIQ dual + generate_randomized_fiq_caption",
        "val_metric": (
            f"maximize {args.val_score_w10}*pool_R@10 + {args.val_score_w50}*pool_R@50 "
            f"(mean over cats; Top-{args.k_val})"
        ),
        "val_score_w10": args.val_score_w10,
        "val_score_w50": args.val_score_w50,
        "cirr_frozen": False,
        "loads_cirr_weights": False,
        "hyperparam_vs_cirr": {
            "kept": ["tau=0.07", "freeze_ViT", "cosine", "wd=0.05", "score_mode=itc"],
            "changed": [
                "k_val 50→100 (FIQ R@10, pool coverage)",
                "val metric pool R@1 → mean pool R@10",
                "caption single → dual randomized",
                "early_stop_patience 0→6",
                "mixed 3 categories, no CIRR group negs",
            ],
        },
    }
    with open(out_dir / "exp_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def _validate_pool(args, model: nn.Module) -> dict:
    torch.cuda.empty_cache()
    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    score_mode = "itc" if args.stage == "a" else "itm"
    out = rerank_all_categories(
        _unwrap(model),
        categories=cats,
        k_rerank=args.k_val,
        score_mode=score_mode,
        topk_dir=Path(args.topk_dir),
        max_queries=args.val_max_queries if not args.smoke else 20,
        gallery_batch=args.gallery_batch,
        num_workers=0 if args.smoke else 4,
    )
    return out


def train(args, device: torch.device, rank: int, world_size: int, is_distributed: bool) -> None:
    is_main = rank == 0
    _abort_cirr(
        args.output_dir,
        args.topk_dir or "",
        args.resume or "",
    )
    if args.validate_every <= 0 and not args.smoke:
        raise ValueError("--validate_every must be > 0")
    if not args.smoke and not args.exp_name and not args.resume:
        raise ValueError("Pass --exp_name so FIQ QTR runs stay isolated.")
    if args.stage != "a":
        raise NotImplementedError(
            "FashionIQ CQM uses the contrastive composed-query loss."
        )

    cats = [x.strip() for x in args.fiq_categories.split(",") if x.strip()]
    model = build_blip2_qtr(
        device,
        pretrained_id=args.pretrained_id,
        local_files_only=args.local_files_only,
        freeze_vision=not args.blip_img_tune,
        stage=args.stage,
    )

    if is_distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )

    if is_main:
        n_trainable = sum(p.numel() for p in _unwrap(model).parameters() if p.requires_grad) / 1e6
        global_b = args.batch_size * world_size if (is_distributed and args.global_contrastive) else args.batch_size
        logger.info(
            "FIQ arch=%s stage=%s trainable=%.2fM | world=%d local_bs=%d global_ce_B=%d cats=%s",
            ARCH_TAG, args.stage, n_trainable, world_size, args.batch_size, global_b, cats,
        )

    preprocess = fiq_targetpad_preprocess()
    train_ds = FIQQTRRelativeTrain(preprocess, categories=cats, split="train")
    collate_fn = fiq_qtr_collate
    sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    optim = _rebuild_optimizer(model, args.lr, args.weight_decay)
    ce_a = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(args.output_dir) / (args.exp_name or f"stage{args.stage}_legacy")
    if args.resume and not args.exp_name:
        out_dir = Path(args.resume).resolve().parent
    _abort_cirr(str(out_dir))
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        _dump_exp_config(out_dir, args, world_size)
        logger.info("Experiment dir: %s", out_dir)

    start_epoch = 0
    best_r10 = -1.0
    best_r50 = -1.0
    best_score = -1.0
    bad_vals = 0
    w10 = float(args.val_score_w10)
    w50 = float(args.val_score_w50)
    if w10 < 0 or w50 < 0 or (w10 + w50) <= 0:
        raise ValueError("--val_score_w10/--val_score_w50 must be non-negative and sum > 0")
    if is_main:
        logger.info("Val selection score = %.2f*R@10 + %.2f*R@50", w10, w50)
    if args.resume:
        resume_path = Path(args.resume)
        _abort_cirr(str(resume_path))
        if resume_path.is_file():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            load_blip2_qtr_ckpt(_unwrap(model), str(resume_path), device)
            _unwrap(model).set_stage(args.stage)
            optim = _rebuild_optimizer(model, args.lr, args.weight_decay)
            start_epoch = int(state.get("epoch", -1)) + 1
            best_r10 = float(state.get("best_pool_r10", -1.0))
            if not args.reset_optimizer:
                if "optimizer" in state:
                    try:
                        optim.load_state_dict(state["optimizer"])
                    except Exception as e:  # noqa: BLE001
                        if is_main:
                            logger.warning("Optimizer state mismatch (%s); fresh AdamW", e)
                        optim = _rebuild_optimizer(model, args.lr, args.weight_decay)
                if scaler.is_enabled() and state.get("scaler"):
                    scaler.load_state_dict(state["scaler"])
            else:
                # Fresh finetune: drop old best (metric / pool size may differ).
                best_r10 = -1.0
                best_r50 = -1.0
                best_score = -1.0
            if is_main:
                logger.info(
                    "Resumed %s epoch=%d best_score=%.2f reset_optimizer=%s k_val=%d",
                    resume_path, start_epoch, best_score, args.reset_optimizer, args.k_val,
                )

    max_epochs = 1 if args.smoke else args.epochs
    ckpt_last = out_dir / f"qtr_stage{args.stage}_last.pt"
    ckpt_best = out_dir / f"qtr_stage{args.stage}_best_pool_r10.pt"
    stop_training = False

    for epoch in range(start_epoch, max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        _cosine_lr(optim, epoch, args.blip_max_epoch, args.lr, 0.0)
        model.train()
        if not args.blip_img_tune:
            _unwrap(model).vision_model.eval()
        if args.stage == "b":
            raw_b = _unwrap(model)
            raw_b.qformer_c.eval()
            raw_b.embeddings.eval()

        total_loss = 0.0
        n_steps = 0
        optim.zero_grad(set_to_none=True)
        it = tqdm(loader, desc=f"fiq-qtr-{args.stage} ep{epoch}", ncols=88) if is_main else loader
        raw = _unwrap(model)

        for idx, batch in enumerate(it):
            if batch is None:
                continue
            if args.smoke and idx >= 8:
                break
            ref_imgs, tgt_imgs, cap_a, cap_b = batch
            captions = generate_randomized_fiq_caption(cap_a, cap_b)
            ref_imgs = ref_imgs.to(device, non_blocking=True)
            tgt_imgs = tgt_imgs.to(device, non_blocking=True)
            b = ref_imgs.size(0)

            ref_feats = _embed_images(model, ref_imgs, args.blip_bs, args.blip_img_tune)
            tgt_feats = _embed_images(model, tgt_imgs, args.blip_bs, args.blip_img_tune)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                use_global = is_distributed and args.global_contrastive
                if use_global:
                    ref_g, tgt_g, caps_g, off = _gather_contrastive_batch(
                        ref_feats, tgt_feats, captions, world_size=world_size, rank=rank
                    )
                    logits_full = raw.stage_a_logits_scaled(
                        ref_g, tgt_g, caps_g, temperature=args.temperature
                    )
                    logits_rows = logits_full[off : off + b]
                    gt = torch.arange(off, off + b, device=device, dtype=torch.long)
                    loss_raw = ce_a(logits_rows, gt)
                else:
                    logits = raw.stage_a_logits_scaled(
                        ref_feats, tgt_feats, captions, temperature=args.temperature
                    )
                    gt = torch.arange(b, device=device, dtype=torch.long)
                    loss_raw = ce_a(logits, gt)
                loss = loss_raw / args.grad_accumulation_step

            scaler.scale(loss).backward()
            if ((idx + 1) % args.grad_accumulation_step == 0) or (idx + 1 == len(loader)):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            total_loss += float(loss_raw.item())
            n_steps += 1

        if is_distributed:
            dist.barrier()

        if is_main:
            logger.info("epoch %d train CE %.4f", epoch, total_loss / max(n_steps, 1))
            _save_ckpt(ckpt_last, model, optim, scaler, epoch, args, best_r10)

            if args.validate_every > 0 and (epoch % args.validate_every == 0 or args.smoke):
                was_training = model.training
                model.eval()
                val = _validate_pool(args, model)
                if was_training:
                    model.train()
                    if not args.blip_img_tune:
                        _unwrap(model).vision_model.eval()
                r10 = float(val["mean"]["pool_recall_at10"])
                r50 = float(val["mean"]["pool_recall_at50"])
                score = w10 * r10 + w50 * r50
                improved = score > best_score + 1e-9
                if improved:
                    best_score = score
                    best_r10 = r10
                    best_r50 = r50
                    bad_vals = 0
                    _save_ckpt(ckpt_best, model, optim, scaler, epoch, args, best_r10)
                    with open(out_dir / "best_val.json", "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "epoch": epoch,
                                "val_score": score,
                                "val_score_w10": w10,
                                "val_score_w50": w50,
                                "mean": val["mean"],
                                "per_category": val["per_category"],
                            },
                            f,
                            indent=2,
                        )
                    logger.info(
                        "New best val score=%.2f (%.2f*R10+%.2f*R50) | R@10=%.2f R@50=%.2f",
                        score, w10, w50, r10, r50,
                    )
                else:
                    bad_vals += 1
                    logger.info(
                        "No improvement (best score=%.2f R10=%.2f R50=%.2f, bad_vals=%d/%d)",
                        best_score, best_r10, best_r50, bad_vals, args.early_stop_patience,
                    )
                    if args.early_stop_patience > 0 and bad_vals >= args.early_stop_patience and not args.smoke:
                        logger.info(
                            "Early stop at epoch %d best_score=%.2f R10=%.2f R50=%.2f",
                            epoch, best_score, best_r10, best_r50,
                        )
                        stop_training = True

        if is_distributed:
            flag = torch.tensor([1 if (is_main and stop_training) else 0], device=device, dtype=torch.int)
            dist.broadcast(flag, src=0)
            stop_training = bool(flag.item())
            dist.barrier()
        if stop_training:
            break

    if is_main:
        logger.info(
            "Done. out_dir=%s best_score=%.2f R10=%.2f R50=%.2f stage=%s",
            out_dir, best_score, best_r10, best_r50, args.stage,
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    rank, world_size, local_rank, is_distributed = _init_distributed()
    if is_distributed and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    try:
        train(args, device, rank, world_size, is_distributed)
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
