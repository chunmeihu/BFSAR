#!/usr/bin/env python
"""Stage 1: train Banzhaf interaction estimator on CIRR (ViT-L/14, multi-crop CLS)."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import os
import random
from os.path import join

import numpy as np
import torch

from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from bfsar.cirr.modeling_bm import BMEstimator
from bfsar.cirr.optimization_bertadam import BertAdam
from bfsar.cirr.cirr_dataloader_builders import build_cirr_train_dataloader
from bfsar.cirr import cirr_paths
from bfsar.utils.comm import is_main_process, synchronize
from bfsar.utils.logger import setup_logger
from bfsar.utils.metric_logger import MetricLogger

global logger


def get_args():
    parser = argparse.ArgumentParser(
        description="CIRR Banzhaf estimator (BFSAR Stage 1, no MSRVTT warm-start)")
    parser.add_argument("--do_train", type=int, default=1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Stage-1 default 1e-2 (same as BFSAR README)")
    parser.add_argument("--coef_lr", type=float, default=1e-3)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation; effective update batch = batch_size * grad_accum_steps")
    parser.add_argument("--max_words", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=12, help="Number of image crops (players)")
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--distributed", type=int, default=0,
                        help="Auto-enabled when launched via torchrun")
    parser.add_argument("--n_display", type=int, default=50)
    parser.add_argument("--base_encoder", type=str, default="ViT-L/14")
    parser.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    parser.add_argument("--agg_module", type=str, default="None", choices=["None", "seqLSTM", "seqTransf"])
    parser.add_argument("--interaction", type=str, default="fgsa")
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--banzhaf_mc_samples", type=int, default=50,
                        help="Monte Carlo samples per token-crop pair (BFSAR default 100)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="1-based epoch to start/resume from; loads epoch{start_epoch-1}.pt when >1")
    parser.add_argument("--cirr_image_root", type=str, default=None)
    return parser.parse_args()


def setup_device(args):
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        if "RANK" in os.environ:
            args.distributed = 1
        if args.distributed and "RANK" in os.environ:
            torch.distributed.init_process_group(backend="nccl")
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

    rank_seed = args.seed + args.local_rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
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


def will_optimizer_step(micro_step, grad_accum_steps, num_batches, epoch_micro_step):
    return (micro_step % grad_accum_steps == 0) or (epoch_micro_step == num_batches)


def backward_context(model, args, micro_step, grad_accum_steps, num_batches, epoch_micro_step):
    sync_grads = will_optimizer_step(micro_step, grad_accum_steps, num_batches, epoch_micro_step)
    if args.distributed and hasattr(model, "no_sync") and not sync_grads:
        return model.no_sync()
    return contextlib.nullcontext()


def prep_optimizer(args, model):
    if hasattr(model, "module"):
        inner = model.module
    else:
        inner = model
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    params = list(inner.named_parameters())
    groups = [
        {"params": [p for n, p in params if "banzhafteacher" in n and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in params if "banzhafteacher" in n and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    return BertAdam(groups, lr=args.lr, warmup=args.warmup_proportion, schedule="warmup_cosine",
                    b1=0.9, b2=0.98, e=1e-6, weight_decay=args.weight_decay, max_grad_norm=1.0)


def save_estimator(epoch, args, model):
    inner = model.module if hasattr(model, "module") else model
    path = join(args.output_dir, f"banzhaf_estimator.epoch{epoch}.pt")
    torch.save(inner.banzhafteacher.state_dict(), path)
    logger.info("Saved estimator to %s", path)
    return path


def load_estimator_checkpoint(args, model, epoch):
    inner = model.module if hasattr(model, "module") else model
    path = join(args.output_dir, f"banzhaf_estimator.epoch{epoch}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    state = torch.load(path, map_location=args.device)
    inner.banzhafteacher.load_state_dict(state)
    logger.info("Loaded estimator checkpoint from %s", path)
    return path


def main():
    global logger
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("cirr_estimator", args.output_dir, args.local_rank)
    args = setup_device(args)

    # 2-GPU DDP: accum=2 keeps global update batch ~128 (32×2×2); single-GPU uses accum=4
    if args.world_size > 1 and args.grad_accum_steps == 4:
        args.grad_accum_steps = max(1, 128 // (args.batch_size * args.world_size))
        if is_main_process():
            logger.info(
                "Dual-GPU: auto grad_accum_steps=%d (global update batch=%d)",
                args.grad_accum_steps,
                args.batch_size * args.grad_accum_steps * args.world_size,
            )

    for k in sorted(vars(args)):
        logger.info("  %s: %s", k, getattr(args, k))

    tokenizer = ClipTokenizer()
    train_loader, train_len, train_sampler = build_cirr_train_dataloader(args, tokenizer)
    logger.info("Train pairs: %d", train_len)

    model = BMEstimator(args).to(args.device)
    if args.distributed and torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True)

    optimizer = prep_optimizer(args, model)
    num_batches = len(train_loader)
    optim_steps_per_epoch = optimizer_steps_per_epoch(num_batches, args.grad_accum_steps)
    max_optim_steps = optim_steps_per_epoch * args.epochs
    for g in optimizer.param_groups:
        g["t_total"] = max_optim_steps

    eff_batch = args.batch_size * args.grad_accum_steps * args.world_size
    logger.info(
        "Effective batch: per-GPU micro=%d, accum=%d, world=%d -> global update batch=%d",
        args.batch_size, args.grad_accum_steps, args.world_size, eff_batch,
    )
    logger.info("Optimizer steps: %d per epoch, %d total", optim_steps_per_epoch, max_optim_steps)

    start_epoch_idx = max(0, args.start_epoch - 1)
    if start_epoch_idx > 0:
        load_estimator_checkpoint(args, model, start_epoch_idx)
        if is_main_process():
            logger.info("Resuming from epoch %d (loaded epoch%d.pt)", args.start_epoch, start_epoch_idx)

    meters = MetricLogger(delimiter="  ")
    micro_step = num_batches * start_epoch_idx
    optim_step = optim_steps_per_epoch * start_epoch_idx
    optimizer.zero_grad()
    for epoch in range(start_epoch_idx, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(train_loader, start=1):
            micro_step += 1
            text_ids, text_mask, crops, crop_mask, idx = batch
            text_ids = text_ids.to(args.device, non_blocking=True)
            text_mask = text_mask.to(args.device, non_blocking=True)
            crops = crops.to(args.device, non_blocking=True)
            crop_mask = crop_mask.to(args.device, non_blocking=True)
            idx = idx.to(args.device, non_blocking=True)

            loss = model(text_ids, text_mask, crops, crop_mask, idx, micro_step)
            if loss is None:
                continue
            if hasattr(loss, "mean"):
                loss = loss.mean()
            with backward_context(model, args, micro_step, args.grad_accum_steps, num_batches, batch_idx):
                (loss / args.grad_accum_steps).backward()

            stepped = maybe_optimizer_step(
                model, optimizer, micro_step, args.grad_accum_steps, num_batches, batch_idx)
            if stepped:
                optim_step += 1
                meters.update(loss=float(loss.item()))

            if optim_step > 0 and optim_step % args.n_display == 0 and is_main_process():
                logger.info("epoch %d optim %d/%d  %s", epoch + 1, optim_step, max_optim_steps, str(meters))

        if is_main_process():
            save_estimator(epoch + 1, args, model)
        synchronize()

    logger.info("Estimator training finished.")


if __name__ == "__main__":
    main()
