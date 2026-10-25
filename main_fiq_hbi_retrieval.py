#!/usr/bin/env python
"""Stage 2: Fashion-IQ text-to-image retrieval with HBICIRR (ViT-L/14, multi-crop CLS).

Protocol mirrors CIRR coarse retrieval (main_cirr_hbi_retrieval.py): same model, losses,
optimizer, and DDP recipe, with FIQ 6-crop CLS. Fresh HBI + FIQ Stage-1 estimator only —
does not load CIRR retrieval weights. Val is official FIQ: per-category gallery,
mean R@10/50/100/200; best checkpoint is selected by mean R@100 (R@200 tie-break).
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random
import time
from datetime import timedelta
from os.path import exists, join

import numpy as np
import torch

from HBI.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from HBI.cirr.modeling_hbi_cirr import HBICIRR
from HBI.cirr.optimization_bertadam import BertAdam
from HBI.fiq.fiq_dataloader_builders import (
    build_fiq_train_dataloader,
    build_fiq_query_dataloader,
    build_fiq_gallery_dataloader,
)
from HBI.fiq.eval_fiq_hbi import evaluate_fiq_val
from HBI.fiq import fiq_paths
from HBI.utils.comm import is_main_process, synchronize
from HBI.utils.logger import setup_logger
from HBI.utils.metric_logger import MetricLogger

global logger


def get_args():
    parser = argparse.ArgumentParser(
        description="Fashion-IQ HBI retrieval (ModifiedImageCaption -> category gallery)")
    parser.add_argument("--do_train", type=int, default=0)
    parser.add_argument("--do_eval", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--coef_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-GPU batch; contrastive batch = batch_size * num_gpus")
    parser.add_argument("--batch_size_val", type=int, default=16)
    parser.add_argument("--max_words", type=int, default=77)
    parser.add_argument("--max_frames", type=int, default=6, help="Number of image crops; FIQ uses 6")
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--distributed", type=int, default=0,
                        help="Set 1 for DDP; auto-enabled when launched via torchrun")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation (does not increase contrastive negatives)")
    parser.add_argument("--gradient_checkpointing", type=int, default=1,
                        help="Checkpoint ViT visual blocks to save memory (recommended)")
    parser.add_argument("--image_encode_microbatch", type=int, default=6,
                        help="Encode crops in micro-batches of N images (0=disable)")
    parser.add_argument("--n_display", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500, help="Run FIQ val every N steps (0=epoch only)")
    parser.add_argument("--base_encoder", type=str, default="ViT-L/14")
    parser.add_argument("--clip_checkpoint", type=str, default=fiq_paths.CLIP_VITL14_CHECKPOINT)
    parser.add_argument("--agg_module", type=str, default="None", choices=["None", "seqLSTM", "seqTransf"])
    parser.add_argument("--interaction", type=str, default="wti")
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--estimator", type=str, default="None", help="Stage-1 banzhafteacher weights")
    parser.add_argument("--init_model", type=str, default=None, help="Optional HBICIRR checkpoint (not CIRR SOTA)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="1-based epoch to start/resume from; loads epoch{start_epoch-1} weights when >1")
    parser.add_argument("--resume_best_r10", type=float, default=0.0,
                        help="Best val mean R@10 from prior stage (logged; selection uses R@100)")
    parser.add_argument("--resume_best_r100", type=float, default=0.0,
                        help="Best val mean R@100 from prior stage (keeps best checkpoint tracking on resume)")
    parser.add_argument("--resume_best_r200", type=float, default=0.0,
                        help="Best val mean R@200 from prior stage (R@100 tie-break)")
    parser.add_argument("--kl", type=float, default=2.0)
    parser.add_argument("--skl", type=float, default=1.0)
    parser.add_argument("--v_rate0", type=float, default=0.5,
                        help="Visual CTM stage-0 keep rate; 0.5 on 6 crops → ~3 clusters")
    parser.add_argument("--v_rate1", type=float, default=1.0,
                        help="Visual CTM stage-1 keep rate; 1.0 on 6 crops keeps all players")
    parser.add_argument("--t_rate0", type=float, default=0.25)
    parser.add_argument("--t_rate1", type=float, default=0.5)
    parser.add_argument("--rate", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument(
        "--wti_levels",
        type=int,
        default=3,
        choices=[1, 3],
        help="WTI hierarchy depth: 3=entity+action+event (default), 1=entity-only (ablation A3)",
    )
    parser.add_argument("--fiq_image_root", type=str, default=None)
    parser.add_argument(
        "--fiq_categories",
        type=str,
        default="dress,shirt,toptee",
        help="Comma-separated Fashion-IQ categories",
    )
    return parser.parse_args()


def _categories(args):
    raw = getattr(args, "fiq_categories", None) or "dress,shirt,toptee"
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def setup_device(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        if "RANK" in os.environ:
            args.distributed = 1
        if args.distributed and "RANK" in os.environ:
            # FIQ val encodes 3 galleries (~15k images). H-14 12-crop val is
            # ~90-130 min; rank 1 waits at the barrier the whole time.
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(hours=6)
            )
            args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
            args.world_size = int(os.environ.get("WORLD_SIZE", 1))
        else:
            args.device = torch.device("cuda", args.local_rank)
            args.world_size = 1
            args.distributed = 0
    else:
        args.device = torch.device("cpu")
        args.world_size = 1
        args.distributed = 0
    return args


def optimizer_steps_per_epoch(num_batches, grad_accum_steps):
    return (num_batches + grad_accum_steps - 1) // grad_accum_steps


def maybe_optimizer_step(model, optimizer, micro_step, grad_accum_steps, num_batches, epoch_micro_step):
    is_accum_boundary = (micro_step % grad_accum_steps == 0)
    is_epoch_tail = (epoch_micro_step == num_batches)
    if is_accum_boundary or is_epoch_tail:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return True
    return False


def build_model(args):
    model = HBICIRR(args)
    ckpt = resolve_resume_checkpoint(args)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint from %s", ckpt)
    return model.to(args.device)


def resolve_resume_checkpoint(args):
    if args.init_model and exists(args.init_model):
        return args.init_model
    start_epoch_idx = max(0, args.start_epoch - 1)
    if start_epoch_idx <= 0:
        return None
    prev_epoch = start_epoch_idx
    candidates = [
        join(args.output_dir, f"pytorch_model.bin.best.{prev_epoch}"),
        join(args.output_dir, f"pytorch_model.bin.{prev_epoch}"),
    ]
    for path in candidates:
        if exists(path):
            return path
    raise FileNotFoundError(
        f"No checkpoint for epoch {prev_epoch} under {args.output_dir}. "
        f"Tried: {candidates}"
    )


def prep_optimizer(args, model, t_total):
    inner = model.module if hasattr(model, "module") else model
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    named = list(inner.named_parameters())
    decay_clip = [(n, p) for n, p in named if "clip." in n and not any(nd in n for nd in no_decay)]
    decay_other = [(n, p) for n, p in named if "clip." not in n and not any(nd in n for nd in no_decay)]
    nodecay_clip = [(n, p) for n, p in named if "clip." in n and any(nd in n for nd in no_decay)]
    nodecay_other = [(n, p) for n, p in named if "clip." not in n and any(nd in n for nd in no_decay)]
    groups = [
        {"params": [p for _, p in decay_clip], "weight_decay": args.weight_decay, "lr": args.lr * args.coef_lr},
        {"params": [p for _, p in decay_other], "weight_decay": args.weight_decay},
        {"params": [p for _, p in nodecay_clip], "weight_decay": 0.0, "lr": args.lr * args.coef_lr},
        {"params": [p for _, p in nodecay_other], "weight_decay": 0.0},
    ]
    opt = BertAdam(groups, lr=args.lr, warmup=args.warmup_proportion, schedule="warmup_cosine",
                   b1=0.9, b2=0.98, e=1e-6, t_total=t_total, weight_decay=args.weight_decay, max_grad_norm=1.0)
    if args.distributed and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)
    return opt, model


def save_model(epoch, args, model, tag=""):
    inner = model.module if hasattr(model, "module") else model
    name = f"pytorch_model.bin.{tag + '.' if tag else ''}{epoch}"
    path = join(args.output_dir, name)
    torch.save(inner.state_dict(), path)
    logger.info("Saved model to %s", path)
    return path


def build_val_category_loaders(args, tokenizer):
    loaders = {}
    for cat in _categories(args):
        query_loader, n_q, _ = build_fiq_query_dataloader(args, tokenizer, split="val", category=cat)
        gallery_loader, n_g, _ = build_fiq_gallery_dataloader(args, split="val", category=cat)
        logger.info("FIQ val %s: queries=%d gallery=%d", cat, n_q, n_g)
        loaders[cat] = (query_loader, gallery_loader)
    return loaders


def run_fiq_eval(args, model, category_loaders=None):
    tokenizer = ClipTokenizer()
    if category_loaders is None:
        category_loaders = build_val_category_loaders(args, tokenizer)
    return evaluate_fiq_val(
        model,
        category_loaders,
        args.device,
        query_batch=args.batch_size_val,
        gallery_batch=args.batch_size_val,
    )


def _is_better_coverage(metrics, best_r100, best_r200):
    """Prefer mean R@100, then R@200. Coverage for later CR rerank matters more than R@10."""
    r100 = float(metrics["recall_at100"])
    r200 = float(metrics["recall_at200"])
    if r100 > best_r100 + 1e-8:
        return True
    if abs(r100 - best_r100) <= 1e-8 and r200 > best_r200 + 1e-8:
        return True
    return False


def train(args, model, train_loader, train_sampler, category_loaders):
    num_batches = len(train_loader)
    steps_per_epoch = optimizer_steps_per_epoch(num_batches, args.grad_accum_steps)
    start_epoch_idx = max(0, args.start_epoch - 1)
    remaining_epochs = args.epochs - start_epoch_idx
    if remaining_epochs <= 0:
        raise ValueError(
            f"start_epoch={args.start_epoch} must be < epochs={args.epochs}"
        )
    max_optim_steps = steps_per_epoch * remaining_epochs
    optimizer, model = prep_optimizer(args, model, max_optim_steps)
    meters = MetricLogger(delimiter="  ")
    best_r10 = float(args.resume_best_r10)
    best_r100 = float(args.resume_best_r100)
    best_r200 = float(args.resume_best_r200)
    micro_step = num_batches * start_epoch_idx
    optim_step = 0
    optimizer.zero_grad()

    contrastive_batch = args.batch_size * args.world_size
    update_batch = args.batch_size * args.grad_accum_steps * args.world_size
    logger.info(
        "Training epochs %d..%d (%d remaining); lr=%.2e coef_lr=%.2e warmup=%.2f",
        start_epoch_idx + 1, args.epochs, remaining_epochs,
        args.lr, args.coef_lr, args.warmup_proportion,
    )
    logger.info(
        "Optimizer steps: %d per epoch, %d total (resume best mean R@100=%.2f R@200=%.2f R@10=%.2f)",
        steps_per_epoch, max_optim_steps, best_r100, best_r200, best_r10,
    )
    logger.info(
        "Contrastive in-batch batch=%d (per-GPU %d x world %d); "
        "optimizer update batch=%d (accum=%d)",
        contrastive_batch, args.batch_size, args.world_size,
        update_batch, args.grad_accum_steps,
    )

    for epoch in range(start_epoch_idx, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        end = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            micro_step += 1
            text_ids, text_mask, crops, crop_mask, idx = batch
            text_ids = text_ids.to(args.device, non_blocking=True)
            text_mask = text_mask.to(args.device, non_blocking=True)
            crops = crops.to(args.device, non_blocking=True)
            crop_mask = crop_mask.to(args.device, non_blocking=True)
            idx = idx.to(args.device, non_blocking=True)

            loss, loss1, loss2 = model(text_ids, text_mask, crops, crop_mask, idx, micro_step)
            if args.world_size > 1:
                loss = loss.mean()
                loss1 = loss1.mean()
                loss2 = loss2.mean()

            (loss / args.grad_accum_steps).backward()

            stepped = maybe_optimizer_step(
                model, optimizer, micro_step, args.grad_accum_steps, num_batches, batch_idx)
            if stepped:
                optim_step += 1
                inner = model.module if hasattr(model, "module") else model
                torch.clamp_(inner.clip.logit_scale.data, max=np.log(100))
                meters.update(
                    loss=float(loss.item()),
                    E=float(loss1.item()),
                    M=float(loss2.item()),
                    time=time.time() - end,
                )
                end = time.time()

            if optim_step > 0 and optim_step % args.n_display == 0 and is_main_process():
                logger.info(
                    "epoch %d optim %d/%d  %s",
                    epoch + 1, optim_step, max_optim_steps, str(meters),
                )

            if args.eval_every > 0 and optim_step > 0 and optim_step % args.eval_every == 0:
                if is_main_process():
                    logger.info(
                        "FIQ val at optim %d (3 galleries; H-14 often 60-120 min; "
                        "tqdm is in train.stdout, not this log)",
                        optim_step,
                    )
                    metrics = evaluate_fiq_val(
                        model, category_loaders, args.device,
                        query_batch=args.batch_size_val, gallery_batch=args.batch_size_val,
                    )
                    if _is_better_coverage(metrics, best_r100, best_r200):
                        best_r100 = metrics["recall_at100"]
                        best_r200 = metrics["recall_at200"]
                        best_r10 = metrics["recall_at10"]
                        logger.info(
                            "New best by R@100: mean R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f",
                            metrics["recall_at10"], metrics["recall_at50"],
                            metrics["recall_at100"], metrics["recall_at200"],
                        )
                        save_model(epoch + 1, args, model, tag="best")
                    model.train()
                else:
                    logger.info("Rank waiting at barrier during FIQ val (timeout=6h)")
                synchronize()

        if is_main_process():
            save_model(epoch + 1, args, model)
            logger.info(
                "FIQ val after epoch %d (3 galleries; H-14 often 60-120 min; "
                "tqdm is in train.stdout, not this log)",
                epoch + 1,
            )
            metrics = evaluate_fiq_val(
                model, category_loaders, args.device,
                query_batch=args.batch_size_val, gallery_batch=args.batch_size_val,
            )
            if _is_better_coverage(metrics, best_r100, best_r200):
                best_r100 = metrics["recall_at100"]
                best_r200 = metrics["recall_at200"]
                best_r10 = metrics["recall_at10"]
                logger.info(
                    "New best by R@100: mean R@10=%.2f R@50=%.2f R@100=%.2f R@200=%.2f",
                    metrics["recall_at10"], metrics["recall_at50"],
                    metrics["recall_at100"], metrics["recall_at200"],
                )
                save_model(epoch + 1, args, model, tag="best")
        else:
            logger.info("Rank waiting at barrier during FIQ val (timeout=6h)")
        synchronize()

    logger.info(
        "Best val mean R@100=%.2f R@200=%.2f (R@10=%.2f)",
        best_r100, best_r200, best_r10,
    )


def main():
    global logger
    args = get_args()
    if "cirr" in os.path.abspath(args.output_dir).lower():
        raise ValueError(f"FIQ run must not write under a cirr output path: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("fiq_hbi", args.output_dir, args.local_rank)
    args = setup_device(args)

    for k in sorted(vars(args)):
        logger.info("  %s: %s", k, getattr(args, k))

    tokenizer = ClipTokenizer()
    model = build_model(args)

    if args.do_eval and not args.do_train:
        metrics = run_fiq_eval(args, model)
        logger.info("Eval done: %s", metrics)
        return

    if args.do_train:
        train_loader, train_len, train_sampler = build_fiq_train_dataloader(args, tokenizer)
        category_loaders = build_val_category_loaders(args, tokenizer)
        logger.info("Train pairs: %d", train_len)
        train(args, model, train_loader, train_sampler, category_loaders)


if __name__ == "__main__":
    main()
