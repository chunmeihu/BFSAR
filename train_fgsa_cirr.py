#!/usr/bin/env python
"""Stage 2/3: CIRR text-to-image retrieval with FGSAModel (ViT-L/14, multi-crop CLS)."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random
import time
from os.path import exists, join

import numpy as np
import torch

from bfsar.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from bfsar.cirr.modeling_fgsa import FGSAModel
from bfsar.cirr.optimization_bertadam import BertAdam
from bfsar.cirr.cirr_dataloader_builders import (
    build_cirr_train_dataloader,
    build_cirr_query_dataloader,
    build_cirr_gallery_dataloader,
)
from bfsar.cirr.eval_fgsa import evaluate_cirr_val
from bfsar.cirr import cirr_paths
from bfsar.utils.comm import is_main_process, synchronize
from bfsar.utils.logger import setup_logger
from bfsar.utils.metric_logger import MetricLogger

global logger


def get_args():
    parser = argparse.ArgumentParser(
        description="CIRR BFSAR retrieval (text ModifiedImageCaption -> target image gallery)")
    parser.add_argument("--do_train", type=int, default=0)
    parser.add_argument("--do_eval", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--coef_lr", type=float, default=1e-3)
    parser.add_argument(
        "--freeze_clip",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=freeze OpenCLIP backbone (no CLIP grads); train heads/Banzhaf only",
    )
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-GPU batch; contrastive batch = batch_size * num_gpus")
    parser.add_argument("--batch_size_val", type=int, default=16)
    parser.add_argument("--max_words", type=int, default=32)
    parser.add_argument("--max_frames", type=int, default=12, help="Number of image crops")
    parser.add_argument("--image_resolution", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--distributed", type=int, default=0,
                        help="Set 1 for DDP; auto-enabled when launched via torchrun")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation (does not increase contrastive negatives)")
    parser.add_argument("--gradient_checkpointing", type=int, default=1,
                        help="Checkpoint ViT visual blocks to save memory (recommended)")
    parser.add_argument("--image_encode_microbatch", type=int, default=12,
                        help="Encode crops in micro-batches of N images (0=disable)")
    parser.add_argument("--n_display", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500, help="Run CIRR val every N steps (0=epoch only)")
    parser.add_argument("--base_encoder", type=str, default="ViT-L/14")
    parser.add_argument("--clip_checkpoint", type=str, default=cirr_paths.CLIP_VITL14_CHECKPOINT)
    parser.add_argument("--agg_module", type=str, default="None", choices=["None", "seqLSTM", "seqTransf"])
    parser.add_argument("--interaction", type=str, default="fgsa")
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--estimator", type=str, default="None", help="Stage-1 banzhafteacher weights")
    parser.add_argument("--init_model", type=str, default=None, help="Optional FGSAModel checkpoint (not MSRVTT)")
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="1-based epoch to start/resume from; loads epoch{start_epoch-1} weights when >1")
    parser.add_argument("--resume_best_r1", type=float, default=0.0,
                        help="Best val R@1 from prior stage (keeps best checkpoint tracking on resume)")
    parser.add_argument("--kl", type=float, default=2.0)
    parser.add_argument("--lambda_b", type=float, default=1.0)
    parser.add_argument("--v_rate0", type=float, default=0.25)
    parser.add_argument("--v_rate1", type=float, default=0.5)
    parser.add_argument("--t_rate0", type=float, default=0.25)
    parser.add_argument("--t_rate1", type=float, default=0.5)
    parser.add_argument("--rate", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument(
        "--fgsa_levels",
        type=int,
        default=1,
        choices=[1, 3],
        help="Matching depth. The released model uses a single level.",
    )
    parser.add_argument("--cirr_image_root", type=str, default=None)
    return parser.parse_args()


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
    model = FGSAModel(args)
    ckpt = resolve_resume_checkpoint(args)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        # Keep Stage-1 Banzhaf teacher from --estimator; init_model (e.g. A2e)
        # may carry a random/unused teacher that would overwrite it.
        est = getattr(args, "estimator", None)
        if est and str(est) not in ("None", "none", "") and exists(str(est)):
            drop = [k for k in state if k.startswith("banzhafteacher")]
            for k in drop:
                state.pop(k)
            if drop:
                logger.info(
                    "Kept estimator teacher (%s); dropped %d banzhafteacher keys from %s",
                    est, len(drop), ckpt,
                )
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint from %s", ckpt)
    if int(getattr(args, "freeze_clip", 0) or 0) == 1:
        n_frozen = 0
        for name, param in model.named_parameters():
            if name.startswith("clip."):
                param.requires_grad = False
                n_frozen += 1
        logger.info("freeze_clip=1: froze %d CLIP parameters", n_frozen)
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
    named = [(n, p) for n, p in inner.named_parameters() if p.requires_grad]
    freeze_clip = int(getattr(args, "freeze_clip", 0) or 0) == 1
    decay_clip = [(n, p) for n, p in named if "clip." in n and not any(nd in n for nd in no_decay)]
    decay_other = [(n, p) for n, p in named if "clip." not in n and not any(nd in n for nd in no_decay)]
    nodecay_clip = [(n, p) for n, p in named if "clip." in n and any(nd in n for nd in no_decay)]
    nodecay_other = [(n, p) for n, p in named if "clip." not in n and any(nd in n for nd in no_decay)]
    groups = []
    if not freeze_clip:
        groups.append(
            {"params": [p for _, p in decay_clip], "weight_decay": args.weight_decay, "lr": args.lr * args.coef_lr}
        )
    groups.append({"params": [p for _, p in decay_other], "weight_decay": args.weight_decay})
    if not freeze_clip:
        groups.append(
            {"params": [p for _, p in nodecay_clip], "weight_decay": 0.0, "lr": args.lr * args.coef_lr}
        )
    groups.append({"params": [p for _, p in nodecay_other], "weight_decay": 0.0})
    # Drop empty param groups (BertAdam can choke on them).
    groups = [g for g in groups if g["params"]]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    logger.info(
        "Optimizer trainable tensors=%d params=%.2fM freeze_clip=%d",
        sum(len(g["params"]) for g in groups), n_train / 1e6, int(freeze_clip),
    )
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


def run_cirr_eval(args, model):
    tokenizer = ClipTokenizer()
    query_loader, _, _ = build_cirr_query_dataloader(args, tokenizer, split="val")
    gallery_loader, _, _ = build_cirr_gallery_dataloader(args, split="val")
    return evaluate_cirr_val(model, query_loader, gallery_loader, args.device,
                             query_batch=args.batch_size_val, gallery_batch=args.batch_size_val)


def train(args, model, train_loader, train_sampler, query_loader, gallery_loader):
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
    best_r1 = float(args.resume_best_r1)
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
        "Optimizer steps: %d per epoch, %d total (resume best R@1=%.2f)",
        steps_per_epoch, max_optim_steps, best_r1,
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

            if args.eval_every > 0 and optim_step > 0 and optim_step % args.eval_every == 0 and is_main_process():
                metrics = evaluate_cirr_val(model, query_loader, gallery_loader, args.device)
                if metrics["recall_at1"] > best_r1:
                    best_r1 = metrics["recall_at1"]
                    save_model(epoch + 1, args, model, tag="best")
                model.train()

        if is_main_process():
            metrics = evaluate_cirr_val(model, query_loader, gallery_loader, args.device)
            save_model(epoch + 1, args, model)
            if metrics["recall_at1"] > best_r1:
                best_r1 = metrics["recall_at1"]
                save_model(epoch + 1, args, model, tag="best")
        synchronize()

    logger.info("Best val R@1: %.2f", best_r1)


def main():
    global logger
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("cirr_hbi", args.output_dir, args.local_rank)
    args = setup_device(args)

    for k in sorted(vars(args)):
        logger.info("  %s: %s", k, getattr(args, k))

    tokenizer = ClipTokenizer()
    model = build_model(args)

    if args.do_eval and not args.do_train:
        metrics = run_cirr_eval(args, model)
        logger.info("Eval done: %s", metrics)
        return

    if args.do_train:
        train_loader, train_len, train_sampler = build_cirr_train_dataloader(args, tokenizer)
        query_loader, _, _ = build_cirr_query_dataloader(args, tokenizer, split="val")
        gallery_loader, _, _ = build_cirr_gallery_dataloader(args, split="val")
        logger.info("Train pairs: %d", train_len)
        train(args, model, train_loader, train_sampler, query_loader, gallery_loader)


if __name__ == "__main__":
    main()
