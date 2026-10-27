#!/usr/bin/env python3
"""Train composed query matching on CIRR with in-batch contrastive loss."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfsar.mvr.cqm_rerank_cirr import DEFAULT_TOPK_VAL, run_cirr_blip2_qtr_rerank
from bfsar.mvr.cqm_data import get_cirr_preprocess, import_cirr_dataset
from bfsar.mvr.cvs_infer import CR_SRC, ASSET_ROOT, REPO_ROOT
from bfsar.mvr.cqm_model import (
    ARCH_TAG,
    DEFAULT_ITM_ID,
    build_blip2_qtr,
    load_blip2_qtr_ckpt,
)

logger = logging.getLogger("train_cirr_blip2_qtr")
DEFAULT_OUT = str(ROOT / "output/cqm/cirr")


class CIRRTrainGroupHardNegs(Dataset):
    """Wrap CIRR train-relative to also load group members as hard-neg images.

    CIRR img_set typically has 6 members: ref + target + 4 visually similar others.
    Those 4 are the extra ITM negatives (no train BFSAR top-k file exists).
    """

    def __init__(self, base_ds, n_hard: int = 4):
        import PIL.Image

        from bfsar.cirr import cirr_paths

        self.base = base_ds
        self.n_hard = int(n_hard)
        self._pil = PIL.Image
        # Same root CIRRDataset uses after hbi_blip_rerank remaps data_utils.base_path.
        self.img_root = Path(cirr_paths.CIRR_DATASET_ROOT)

    def __len__(self):
        return len(self.base)

    def _load_name(self, name: str):
        path = self.img_root / self.base.name_to_relpath[name]
        return self.base.preprocess(self._pil.open(path))

    def __getitem__(self, index):
        item = self.base[index]
        if item is None:
            return None
        ref, tgt, cap = item
        trip = self.base.triplets[index]
        ref_name = trip["reference"]
        tgt_name = trip["target_hard"]
        members = [m for m in trip["img_set"]["members"] if m not in (ref_name, tgt_name)]
        members = [m for m in members if m in self.base.name_to_relpath]
        if not members:
            others = [n for n in self.base.name_to_relpath if n not in (ref_name, tgt_name)]
            members = others[: self.n_hard]
        while members and len(members) < self.n_hard:
            members.append(members[len(members) % len(members)])
        negs = [self._load_name(m) for m in members[: self.n_hard]]
        if not negs:
            return None
        return ref, tgt, cap, torch.stack(negs, dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="CIRR composed query matching (contrastive)")
    p.add_argument("--stage", type=str, default="a", choices=["a"], help="contrastive composed-query training")
    p.add_argument("--pretrained_id", type=str, default=DEFAULT_ITM_ID)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment folder under --output_dir (required for non-smoke).",
    )
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--blip_max_epoch", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=8, help="Per-GPU")
    p.add_argument("--blip_bs", type=int, default=8)
    p.add_argument("--blip_img_tune", action="store_true")
    p.add_argument("--grad_accumulation_step", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--topk_pt_val", type=str, default=str(DEFAULT_TOPK_VAL))
    p.add_argument("--k_val", type=int, default=50)
    p.add_argument("--validate_every", type=int, default=2)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07, help="Stage-A ITC temperature")
    p.add_argument(
        "--itm_temperature",
        type=float,
        default=1.0,
        help="Stage-B: divide match logits by this before listwise CE",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Stage-B listwise CE label smoothing (recommended 0.1)",
    )
    p.add_argument(
        "--hard_margin",
        type=float,
        default=0.0,
        help="Stage-B: margin for hardest in-batch negative (0=off)",
    )
    p.add_argument(
        "--hard_margin_weight",
        type=float,
        default=0.0,
        help="Weight for hardest-neg margin loss",
    )
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint qformer_r (Stage-B, enables larger global B)",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after N vals without best improvement (0=disabled)",
    )
    p.add_argument(
        "--cqm_ckpt",
        type=str,
        default=None,
        help="Stage-B: load Stage-A best/last weights before freezing cqm",
    )
    p.add_argument(
        "--group_hard_negs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stage-B: append CIRR group members (excl. ref/target) as extra CE negatives",
    )
    p.add_argument(
        "--n_group_hard_negs",
        type=int,
        default=4,
        help="How many CIRR group members to use as hard negatives",
    )
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--reset_optimizer", action="store_true")
    p.add_argument(
        "--global_contrastive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DDP all_gather for global CE negatives",
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _resolve_out_dir(args) -> Path:
    base = Path(args.output_dir)
    if args.exp_name:
        return base / args.exp_name
    return base / f"stage{args.stage}_legacy"


def _dump_exp_config(out_dir: Path, args, world_size: int) -> None:
    import json

    global_b = args.batch_size * world_size if args.global_contrastive else args.batch_size
    cfg = {
        "arch_tag": ARCH_TAG,
        "description": (
            "Composed query matching: in-batch contrastive loss on the composed query and the target image."
        ),
        "stage": args.stage,
        "pretrained_id": args.pretrained_id,
        "batch_size_per_gpu": args.batch_size,
        "world_size": world_size,
        "global_ce_B": global_b,
        "grad_accumulation_step": args.grad_accumulation_step,
        "lr": args.lr,
        "temperature": args.temperature,
        "itm_temperature": args.itm_temperature,
        "label_smoothing": args.label_smoothing,
        "hard_margin": args.hard_margin,
        "hard_margin_weight": args.hard_margin_weight,
        "gradient_checkpointing": args.gradient_checkpointing,
        "early_stop_patience": args.early_stop_patience,
        "blip_max_epoch": args.blip_max_epoch,
        "epochs": args.epochs,
        "k_val": args.k_val,
        "validate_every": args.validate_every,
        "topk_pt_val": args.topk_pt_val,
        "cqm_ckpt": args.cqm_ckpt,
        "group_hard_negs": args.group_hard_negs,
        "n_group_hard_negs": args.n_group_hard_negs,
        "target_pool_r1": "beat CQM 49.96, then 53, squeeze toward 55",
        "stage_b_fixes": [
            "freeze embeddings (stop q* drift)",
            "CIRR group members as extra ITM hard-negs",
            "hardest-neg margin loss",
            "label smoothing",
            "native BLIP-2 ITM (not Twin-from-scratch)",
        ],
    }
    with open(out_dir / "exp_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _init_distributed() -> tuple[int, int, int, bool]:
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


def _stage_b_loss(
    logits: torch.Tensor,
    gt: torch.Tensor,
    *,
    temperature: float,
    label_smoothing: float,
    hard_margin: float,
    hard_margin_weight: float,
) -> tuple[torch.Tensor, float, float]:
    """Listwise CE + optional hardest-negative margin (no extra forward)."""
    import torch.nn.functional as F

    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    temp = max(float(temperature), 1e-6)
    loss_ce = ce(logits / temp, gt)
    loss_hard = logits.new_zeros(())
    if hard_margin_weight > 0 and hard_margin >= 0:
        b = logits.size(0)
        rows = torch.arange(b, device=logits.device)
        pos = logits[rows, gt]
        neg = logits.clone()
        neg[rows, gt] = -1e4
        hard_neg, _ = neg.max(dim=1)
        loss_hard = F.relu(hard_neg - pos + hard_margin).mean()
    loss = loss_ce + hard_margin_weight * loss_hard
    return loss, float(loss_ce.detach()), float(loss_hard.detach())


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


def _gather_contrastive_batch(
    ref_feats: torch.Tensor,
    tgt_feats: torch.Tensor,
    captions,
    *,
    world_size: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list, int]:
    if world_size == 1:
        return ref_feats, tgt_feats, list(captions), 0

    import torch.distributed.nn.functional as dist_nn

    ref_global = torch.cat(dist_nn.all_gather(ref_feats), dim=0)
    tgt_global = torch.cat(dist_nn.all_gather(tgt_feats), dim=0)
    cap_parts: list = [None] * world_size
    dist.all_gather_object(cap_parts, list(captions))
    caps_global = [c for part in cap_parts for c in part]
    offset = rank * ref_feats.size(0)
    return ref_global, tgt_global, caps_global, offset


def _rebuild_optimizer(model, lr: float, weight_decay: float):
    raw = _unwrap(model)
    params = [p for p in raw.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _save_ckpt(path: Path, model, optim, scaler, epoch: int, args, best_r1: float) -> None:
    raw = _unwrap(model)
    payload = {
        "BLIP2_QTR": raw.state_dict(),
        "arch_tag": ARCH_TAG,
        "stage": args.stage,
        "epoch": epoch,
        "train": f"qtr_stage_{args.stage}",
        "pretrained_id": args.pretrained_id,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else {},
        "best_pool_r1": best_r1,
        "grad_accumulation_step": args.grad_accumulation_step,
        "batch_size": args.batch_size,
        "global_contrastive": args.global_contrastive,
        "temperature": args.temperature,
        "exp_name": args.exp_name,
        "cqm_ckpt": args.cqm_ckpt,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def _validate_pool(args, device, model: nn.Module) -> dict:
    torch.cuda.empty_cache()
    score_mode = "itc" if args.stage == "a" else "itm"
    out = run_cirr_blip2_qtr_rerank(
        Path(args.topk_pt_val),
        split="val",
        k_rerank=args.k_val,
        model=_unwrap(model),
        device=device,
        pretrained_id=args.pretrained_id,
        local_files_only=args.local_files_only,
        max_queries=50 if args.smoke else None,
        num_workers=0,
        score_mode=score_mode,
    )
    pm = out.get("pool_metrics") or {}
    logger.info(
        "val[%s] pool R@1=%.2f R@5=%.2f R@50=%.2f (K=%s)",
        score_mode,
        pm.get("pool_recall_at1", float("nan")),
        pm.get("pool_recall_at5", float("nan")),
        pm.get("pool_recall_at50", float("nan")),
        args.k_val,
    )
    return pm


def train(args, device: torch.device, rank: int, world_size: int, is_distributed: bool) -> None:
    is_main = rank == 0
    if args.validate_every <= 0 and not args.smoke:
        raise ValueError("--validate_every must be > 0")
    if not args.smoke and not args.exp_name and not args.resume:
        raise ValueError("Pass --exp_name so Stage-A/B runs stay isolated.")

    if args.stage == "b" and not args.resume and not args.cqm_ckpt and not args.smoke:
        raise ValueError("Stage-B requires --cqm_ckpt (Stage-A best) or --resume")

    model = build_blip2_qtr(
        device,
        pretrained_id=args.pretrained_id,
        local_files_only=args.local_files_only,
        freeze_vision=not args.blip_img_tune,
        stage=args.stage,
    )

    # Stage-B: load cqm weights then re-freeze.
    if args.stage == "b" and args.cqm_ckpt:
        ck = Path(args.cqm_ckpt)
        if not ck.is_file():
            raise FileNotFoundError(f"cqm_ckpt not found: {ck}")
        missing, unexpected = load_blip2_qtr_ckpt(_unwrap(model), str(ck), device)
        if is_main:
            logger.info(
                "Loaded Stage-A cqm from %s (missing=%d unexpected=%d)",
                ck,
                len(missing),
                len(unexpected),
            )
        model.set_stage("b")  # freeze cqm again after load

    if args.gradient_checkpointing and args.stage == "b":
        _unwrap(model).enable_qformer_r_checkpointing()

    if is_distributed:
        # Stage-A freezes reranker; Stage-B freezes cqm → unused params.
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )

    if is_main:
        raw = _unwrap(model)
        n_trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad) / 1e6
        global_b = (
            args.batch_size * world_size
            if (is_distributed and args.global_contrastive)
            else args.batch_size
        )
        logger.info(
            "arch=%s stage=%s trainable=%.2fM | world=%d local_bs=%d global_ce_B=%d accum=%d",
            ARCH_TAG,
            args.stage,
            n_trainable,
            world_size,
            args.batch_size,
            global_b,
            args.grad_accumulation_step,
        )

    preprocess = get_cirr_preprocess()
    CIRRDataset, collate_fn = import_cirr_dataset()
    os.chdir(CR_SRC)
    try:
        train_ds = CIRRDataset("train", "relative", preprocess)
        if args.stage == "b" and args.group_hard_negs:
            train_ds = CIRRTrainGroupHardNegs(train_ds, n_hard=args.n_group_hard_negs)
            if is_main:
                logger.info(
                    "Stage-B group hard-negs ON (n=%d per query, excl. ref/target)",
                    args.n_group_hard_negs,
                )
    finally:
        os.chdir(REPO_ROOT)

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

    out_dir = _resolve_out_dir(args)
    if args.resume and not args.exp_name:
        out_dir = Path(args.resume).resolve().parent
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        _dump_exp_config(out_dir, args, world_size)
        logger.info("Experiment dir: %s", out_dir)

    start_epoch = 0
    best_r1 = -1.0
    bad_vals = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            load_blip2_qtr_ckpt(_unwrap(model), str(resume_path), device)
            _unwrap(model).set_stage(args.stage)
            # Rebuild optim after set_stage so param groups match trainable set.
            if args.reset_optimizer or True:
                # Always rebuild after stage lock; optionally restore states below.
                optim = _rebuild_optimizer(model, args.lr, args.weight_decay)
            start_epoch = int(state.get("epoch", -1)) + 1
            best_r1 = float(state.get("best_pool_r1", -1.0))
            if not args.reset_optimizer:
                if "optimizer" in state:
                    try:
                        optim.load_state_dict(state["optimizer"])
                    except Exception as e:  # noqa: BLE001
                        if is_main:
                            logger.warning("Optimizer state mismatch (%s); using fresh AdamW", e)
                        optim = _rebuild_optimizer(model, args.lr, args.weight_decay)
                if scaler.is_enabled() and state.get("scaler"):
                    scaler.load_state_dict(state["scaler"])
            if is_main:
                logger.info("Resumed %s epoch=%d best_r1=%.2f", resume_path, start_epoch, best_r1)
        elif is_main:
            logger.warning("Resume not found: %s", resume_path)

    max_epochs = 1 if args.smoke else args.epochs
    ckpt_last = out_dir / f"qtr_stage{args.stage}_last.pt"
    ckpt_best = out_dir / f"qtr_stage{args.stage}_best_pool_r1.pt"
    stop_training = False

    for epoch in range(start_epoch, max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        _cosine_lr(optim, epoch, args.blip_max_epoch, args.lr, 0.0)
        model.train()
        if not args.blip_img_tune:
            _unwrap(model).vision_model.eval()
        # Keep frozen cqm in eval for Stage-B BN/dropout stability.
        if args.stage == "b":
            raw_b = _unwrap(model)
            raw_b.qformer_c.eval()
            raw_b.embeddings.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_hm = 0.0
        n_steps = 0
        optim.zero_grad(set_to_none=True)
        it = tqdm(loader, desc=f"qtr-{args.stage} ep{epoch}", ncols=88) if is_main else loader
        raw = _unwrap(model)

        for idx, batch in enumerate(it):
            if args.smoke and idx >= 20:
                break

            hard_imgs = None
            if args.stage == "b" and args.group_hard_negs:
                ref_imgs, tgt_imgs, captions, hard_imgs = batch
            else:
                ref_imgs, tgt_imgs, captions = batch

            ref_imgs = ref_imgs.to(device, non_blocking=True)
            tgt_imgs = tgt_imgs.to(device, non_blocking=True)
            b = ref_imgs.size(0)

            ref_feats = _embed_images(model, ref_imgs, args.blip_bs, args.blip_img_tune)
            tgt_feats = _embed_images(model, tgt_imgs, args.blip_bs, args.blip_img_tune)
            hard_feats = None
            if hard_imgs is not None:
                hard_imgs = hard_imgs.to(device, non_blocking=True)
                g_hard = hard_imgs.size(1)
                hard_flat = hard_imgs.reshape(b * g_hard, *hard_imgs.shape[2:])
                hard_flat_feats = _embed_images(model, hard_flat, args.blip_bs, False)
                hard_feats = hard_flat_feats.view(b, g_hard, *hard_flat_feats.shape[1:])

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
                    ce_v, hm_v = float(loss_raw.detach()), 0.0
                else:
                    logits = raw.stage_a_logits_scaled(
                        ref_feats, tgt_feats, captions, temperature=args.temperature
                    )
                    gt = torch.arange(b, device=device, dtype=torch.long)
                    loss_raw = ce_a(logits, gt)
                    ce_v, hm_v = float(loss_raw.detach()), 0.0
                loss = loss_raw / args.grad_accumulation_step

            scaler.scale(loss).backward()
            if ((idx + 1) % args.grad_accumulation_step == 0) or (idx + 1 == len(loader)):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            total_loss += float(loss_raw.item())
            total_ce += ce_v
            total_hm += hm_v
            n_steps += 1

        if is_distributed:
            dist.barrier()

        if is_main:
            if args.stage == "b":
                logger.info(
                    "epoch %d train loss %.4f (ce %.4f hard_m %.4f)",
                    epoch,
                    total_loss / max(n_steps, 1),
                    total_ce / max(n_steps, 1),
                    total_hm / max(n_steps, 1),
                )
            else:
                logger.info("epoch %d train CE %.4f", epoch, total_loss / max(n_steps, 1))
            _save_ckpt(ckpt_last, model, optim, scaler, epoch, args, best_r1)

            if args.validate_every > 0 and (epoch % args.validate_every == 0 or args.smoke):
                was_training = model.training
                model.eval()
                pm = _validate_pool(args, device, model)
                if was_training:
                    model.train()
                    if not args.blip_img_tune:
                        _unwrap(model).vision_model.eval()
                    if args.stage == "b":
                        _unwrap(model).qformer_c.eval()
                        _unwrap(model).embeddings.eval()
                r1 = float(pm.get("pool_recall_at1", -1))
                if r1 > best_r1:
                    best_r1 = r1
                    bad_vals = 0
                    _save_ckpt(ckpt_best, model, optim, scaler, epoch, args, best_r1)
                    logger.info("New best val pool R@1=%.2f", r1)
                else:
                    bad_vals += 1
                    logger.info(
                        "No improvement (best=%.2f, bad_vals=%d/%d)",
                        best_r1,
                        bad_vals,
                        args.early_stop_patience,
                    )
                    if (
                        args.early_stop_patience > 0
                        and bad_vals >= args.early_stop_patience
                        and not args.smoke
                    ):
                        logger.info(
                            "Early stop at epoch %d (patience=%d, best=%.2f)",
                            epoch,
                            args.early_stop_patience,
                            best_r1,
                        )
                        stop_training = True

        if is_distributed:
            # Broadcast early-stop decision from rank0.
            flag = torch.tensor(
                [1 if (is_main and stop_training) else 0], device=device, dtype=torch.int
            )
            dist.broadcast(flag, src=0)
            stop_training = bool(flag.item())
            dist.barrier()

        if stop_training:
            break

    if is_main:
        logger.info("Done. out_dir=%s best_pool_r1=%.2f stage=%s", out_dir, best_r1, args.stage)


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
