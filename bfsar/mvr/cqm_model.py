"""Composed query matching (CQM).

A reference image and the modification text are encoded by a BLIP-2 Q-Former
into one composed query. Training is in-batch contrastive loss on the cosine
similarity between that query and the target image (temperature 0.07).
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from bfsar.mvr.cvs_infer import _prefer_offline_hf_bert
from bfsar.rerank.paths import blip2_pretrained_id

logger = logging.getLogger("blip2_qtr")

DEFAULT_ITM_ID = blip2_pretrained_id()
VISION_INPUT_SIZE = 224
ARCH_TAG = "qtr_v1c"  # v1c: freeze emb + CIRR group hard-negs on native ITM


def _get_cr_tokenizer():
    import os
    import sys

    from bfsar.mvr.cvs_infer import CR_SRC, REPO_ROOT

    if str(CR_SRC) not in sys.path:
        sys.path.insert(0, str(CR_SRC))
    os.chdir(CR_SRC)
    try:
        from blip import init_tokenizer  # noqa: WPS433

        return init_tokenizer()
    finally:
        os.chdir(REPO_ROOT)


def _resize_word_embeddings(embeddings: nn.Module, new_n: int) -> None:
    src = embeddings.word_embeddings
    old_n, dim = src.weight.shape
    if new_n == old_n:
        return
    pad_idx = getattr(src, "padding_idx", None)
    new_emb = nn.Embedding(new_n, dim, padding_idx=pad_idx)
    with torch.no_grad():
        n_copy = min(old_n, new_n)
        new_emb.weight[:n_copy].copy_(src.weight[:n_copy])
        if new_n > old_n:
            cls_id = 101 if old_n > 101 else 0
            for i in range(old_n, new_n):
                new_emb.weight[i].copy_(src.weight[cls_id])
    embeddings.word_embeddings = new_emb
    logger.info("Resized QTR word embeddings %d → %d", old_n, new_n)


class Blip2QTR(nn.Module):
    """BLIP-2 ITM-based CQM + Reranker for CIRR Stage-2."""

    def __init__(
        self,
        vision: nn.Module,
        embeddings: nn.Module,
        qformer_c: nn.Module,
        qformer_r: nn.Module,
        query_tokens_c: nn.Parameter,
        query_tokens_r: nn.Parameter,
        vision_projection: nn.Module,
        text_projection: nn.Module,
        itm_head: nn.Module,
        image_size: int = VISION_INPUT_SIZE,
    ) -> None:
        super().__init__()
        self.arch_tag = ARCH_TAG
        self.image_size = image_size
        self.num_query_tokens = query_tokens_c.shape[1]
        self.hidden_size = qformer_c.config.hidden_size

        self.vision_model = vision
        self.embeddings = embeddings
        self.qformer_c = qformer_c
        self.qformer_r = qformer_r
        self.query_tokens_c = query_tokens_c
        self.query_tokens_r = query_tokens_r
        self.vision_projection = vision_projection
        self.text_projection = text_projection
        self.itm_head = itm_head

        self.tokenizer = _get_cr_tokenizer()
        self.enc_token_id = self.tokenizer.enc_token_id
        _resize_word_embeddings(self.embeddings, len(self.tokenizer))

        self._cqm_frozen = False

    # ------------------------------------------------------------------ utils
    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] == self.image_size and images.shape[-2] == self.image_size:
            return images
        return F.interpolate(
            images, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
        )

    def img_embed(self, images: torch.Tensor, train: bool = True, atts: bool = False) -> torch.Tensor:
        """Raw ViT tokens [B, N, vision_width] for caching."""
        del train
        images = self._resize_images(images)
        out = self.vision_model(pixel_values=images, return_dict=True)
        tokens = out.last_hidden_state
        if atts:
            atts_t = torch.ones(tokens.size()[:-1], dtype=torch.long, device=tokens.device)
            return tokens, atts_t
        return tokens

    def _tokenize(self, text: Union[str, List[str]], device: torch.device):
        if isinstance(text, str):
            text = [text]
        enc = self.tokenizer(text, padding="longest", return_tensors="pt").to(device)
        enc.input_ids[:, 0] = self.enc_token_id
        return enc.input_ids, enc.attention_mask

    def freeze_cqm(self) -> None:
        for p in self.qformer_c.parameters():
            p.requires_grad = False
        self.query_tokens_c.requires_grad = False
        # embeddings shared; keep trainable for Stage-B ITM text path by default.
        self._cqm_frozen = True
        logger.info("QTR: cqm (qformer_c + query_tokens_c) frozen")

    def freeze_vision(self) -> None:
        for p in self.vision_model.parameters():
            p.requires_grad = False

    def set_stage(self, stage: str) -> None:
        """stage='a': train cqm (+projections); stage='b': freeze cqm, train reranker."""
        stage = stage.lower()
        if stage == "a":
            self._cqm_frozen = False
            for p in self.qformer_c.parameters():
                p.requires_grad = True
            self.query_tokens_c.requires_grad = True
            for p in self.qformer_r.parameters():
                p.requires_grad = False
            self.query_tokens_r.requires_grad = False
            for p in self.itm_head.parameters():
                p.requires_grad = False
            for p in self.vision_projection.parameters():
                p.requires_grad = True
            for p in self.text_projection.parameters():
                p.requires_grad = True
            for p in self.embeddings.parameters():
                p.requires_grad = True
            logger.info("QTR stage=A (cqm+proj trainable; reranker frozen)")
        elif stage == "b":
            self.freeze_cqm()
            for p in self.qformer_r.parameters():
                p.requires_grad = True
            self.query_tokens_r.requires_grad = True
            for p in self.itm_head.parameters():
                p.requires_grad = True
            for p in self.vision_projection.parameters():
                p.requires_grad = False
            for p in self.text_projection.parameters():
                p.requires_grad = False
            # CRITICAL: do NOT train embeddings in Stage-B.
            # embeddings are shared with frozen cqm; updating them drifts q*
            # away from Stage-A while qformer_c cannot compensate → val drop.
            for p in self.embeddings.parameters():
                p.requires_grad = False
            logger.info(
                "QTR stage=B (qformer_r+ITM+residual trainable; "
                "cqm+embeddings frozen)"
            )
        else:
            raise ValueError(f"stage must be 'a' or 'b', got {stage!r}")

    def enable_qformer_r_checkpointing(self) -> None:
        if hasattr(self.qformer_r, "gradient_checkpointing_enable"):
            self.qformer_r.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on qformer_r")
        else:
            logger.warning("qformer_r has no gradient_checkpointing_enable()")

    # ----------------------------------------------------------- Stage-A path
    def compose_queries(
        self,
        ref_vit: torch.Tensor,
        text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (q_star [B,Nq,D], input_ids, text_attn). Native ITM-style fusion on ref."""
        device = ref_vit.device
        input_ids, text_attn = self._tokenize(text, device)
        b = ref_vit.shape[0]
        query_tokens = self.query_tokens_c.expand(b, -1, -1)
        query_attn = torch.ones(b, self.num_query_tokens, device=device, dtype=torch.long)
        attention_mask = torch.cat([query_attn, text_attn], dim=1)

        query_embeds = self.embeddings(input_ids=input_ids, query_embeds=query_tokens)
        ref_mask = torch.ones(ref_vit.shape[:2], device=device, dtype=torch.long)
        out = self.qformer_c(
            query_embeds=query_embeds,
            query_length=self.num_query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=ref_vit,
            encoder_attention_mask=ref_mask,
            return_dict=True,
        )
        q_star = out.last_hidden_state[:, : self.num_query_tokens]
        return q_star, input_ids, text_attn

    def encode_image_queries(self, image_vit: torch.Tensor) -> torch.Tensor:
        """BLIP-2 image queries (no text) → [B, Nq, D]."""
        b = image_vit.shape[0]
        device = image_vit.device
        query_tokens = self.query_tokens_c.expand(b, -1, -1)
        img_mask = torch.ones(image_vit.shape[:2], device=device, dtype=torch.long)
        out = self.qformer_c(
            query_embeds=query_tokens,
            encoder_hidden_states=image_vit,
            encoder_attention_mask=img_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    def stage_a_contrastive_logits(
        self,
        ref_vit: torch.Tensor,
        tgt_vit: torch.Tensor,
        text: Union[str, List[str]],
    ) -> torch.Tensor:
        """
        CIR ITC: composed(ref,text) vs target image queries.
        Returns logits [B, B] (row = query, col = gallery in-batch).
        """
        q_star, _, _ = self.compose_queries(ref_vit, text)
        # Max-sim over query tokens after projection (BLIP-2 ITC style).
        q = F.normalize(self.vision_projection(q_star), dim=-1)  # [B,Nq,d]
        tgt_q = self.encode_image_queries(tgt_vit)
        t = F.normalize(self.vision_projection(tgt_q), dim=-1)  # [B,Nq,d]
        # logits[i,j] = max_{nq} q[i,nq] · t[j,nq']  via einsum then max
        # Use mean-pool for stability (simpler, strong baseline).
        q_pool = F.normalize(q.mean(dim=1), dim=-1)
        t_pool = F.normalize(t.mean(dim=1), dim=-1)
        return q_pool @ t_pool.t()

    # ----------------------------------------------------------- Stage-B path
    def score_pairs(
        self,
        q_star: torch.Tensor,
        text_ids: torch.Tensor,
        text_attn: torch.Tensor,
        cand_vit: torch.Tensor,
        *,
        return_logits2: bool = False,
    ) -> torch.Tensor:
        """
        ITM scores for paired (q_star, text) vs cand (shared batch dim N).

        Default: [N] match logits (class 1).
        return_logits2=True: [N, 2] full ITM logits.
        """
        n = cand_vit.shape[0]
        device = cand_vit.device
        # Composed queries + Stage-B residual (ITM topology, not CR twin).
        q_in = q_star + self.query_tokens_r.expand(n, -1, -1)
        query_attn = torch.ones(n, self.num_query_tokens, device=device, dtype=torch.long)
        attention_mask = torch.cat([query_attn, text_attn], dim=1)
        query_embeds = self.embeddings(input_ids=text_ids, query_embeds=q_in)
        cand_mask = torch.ones(cand_vit.shape[:2], device=device, dtype=torch.long)
        out = self.qformer_r(
            query_embeds=query_embeds,
            query_length=self.num_query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=cand_vit,
            encoder_attention_mask=cand_mask,
            return_dict=True,
        )
        q_out = out.last_hidden_state[:, : self.num_query_tokens]
        logits2 = self.itm_head(q_out).mean(dim=1)  # [N, 2]
        if return_logits2:
            return logits2
        return logits2[:, 1]  # match logit

    def _compose_for_rerank(
        self,
        ref_vit: torch.Tensor,
        text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = torch.no_grad() if self._cqm_frozen else torch.enable_grad()
        with ctx:
            q_star, input_ids, text_attn = self.compose_queries(ref_vit, text)
        if self._cqm_frozen:
            q_star = q_star.detach()
        return q_star, input_ids, text_attn

    def _score_query_vs_cands(
        self,
        q_i: torch.Tensor,
        ids_i: torch.Tensor,
        attn_i: torch.Tensor,
        cand_vit: torch.Tensor,
    ) -> torch.Tensor:
        n = cand_vit.shape[0]
        return self.score_pairs(
            q_i.expand(n, -1, -1),
            ids_i.expand(n, -1),
            attn_i.expand(n, -1),
            cand_vit,
        )

    def img_txt_fusion(
        self,
        ref_vit: torch.Tensor,
        tgt_vit: torch.Tensor,
        text: Union[str, List[str]],
        train: bool = True,
        hard_neg_vit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage-B B×B match logits; optional per-query group hard-negs → [B, B+G]."""
        del train
        b = ref_vit.shape[0]
        q_star, input_ids, text_attn = self._compose_for_rerank(ref_vit, text)

        rows: List[torch.Tensor] = []
        for i in range(b):
            row = self._score_query_vs_cands(
                q_star[i : i + 1], input_ids[i : i + 1], text_attn[i : i + 1], tgt_vit
            )
            if hard_neg_vit is not None:
                extra = self._score_query_vs_cands(
                    q_star[i : i + 1],
                    input_ids[i : i + 1],
                    text_attn[i : i + 1],
                    hard_neg_vit[i],
                )
                row = torch.cat([row, extra], dim=0)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def img_txt_fusion_contrastive_rows(
        self,
        ref_vit: torch.Tensor,
        tgt_vit: torch.Tensor,
        text: Union[str, List[str]],
        row_start: int,
        row_end: int,
        *,
        cand_chunk: int = 0,
        hard_neg_vit: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score local rows vs gathered batch (+ optional local group hard-negs)."""
        del cand_chunk  # always full-row (CR-style); kept for API compat
        q_star, input_ids, text_attn = self._compose_for_rerank(ref_vit, text)

        rows: List[torch.Tensor] = []
        for i in range(row_start, row_end):
            row = self._score_query_vs_cands(
                q_star[i : i + 1], input_ids[i : i + 1], text_attn[i : i + 1], tgt_vit
            )
            if hard_neg_vit is not None:
                extra = self._score_query_vs_cands(
                    q_star[i : i + 1],
                    input_ids[i : i + 1],
                    text_attn[i : i + 1],
                    hard_neg_vit[i - row_start],
                )
                row = torch.cat([row, extra], dim=0)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def img_txt_fusion_val(
        self,
        ref_vit: torch.Tensor,
        tgt_vit: torch.Tensor,
        text: Union[str, List[str]],
    ) -> torch.Tensor:
        """Val: one query, K candidates → [K] match logits (ITM)."""
        if ref_vit.dim() == 2:
            ref_vit = ref_vit.unsqueeze(0)
        k = tgt_vit.shape[0]
        with torch.no_grad():
            q_star, input_ids, text_attn = self.compose_queries(ref_vit, text)
        qi = q_star.expand(k, -1, -1)
        ids = input_ids.expand(k, -1)
        attn = text_attn.expand(k, -1)
        return self.score_pairs(qi, ids, attn, tgt_vit)

    def img_txt_fusion_val_itc(
        self,
        ref_vit: torch.Tensor,
        tgt_vit: torch.Tensor,
        text: Union[str, List[str]],
    ) -> torch.Tensor:
        """Val probe for Stage-A: cosine(composed, cand image-queries) → [K]."""
        if ref_vit.dim() == 2:
            ref_vit = ref_vit.unsqueeze(0)
        q_star, _, _ = self.compose_queries(ref_vit, text)
        q = F.normalize(self.vision_projection(q_star).mean(dim=1), dim=-1)  # [1,d]
        if tgt_vit.dim() == 2:
            tgt_vit = tgt_vit.unsqueeze(0)
        chunk = 50
        parts = []
        for i in range(0, tgt_vit.shape[0], chunk):
            tgt_q = self.encode_image_queries(tgt_vit[i : i + chunk])
            t = F.normalize(self.vision_projection(tgt_q).mean(dim=1), dim=-1)
            parts.append(q @ t.t())
        return torch.cat(parts, dim=-1).squeeze(0)

    def stage_a_logits_scaled(
        self,
        ref_vit: torch.Tensor,
        tgt_vit: torch.Tensor,
        text: Union[str, List[str]],
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Stage-A CIR ITC logits / τ for CE."""
        return self.stage_a_contrastive_logits(ref_vit, tgt_vit, text) / max(temperature, 1e-6)


def build_blip2_qtr(
    device: torch.device,
    *,
    pretrained_id: str = DEFAULT_ITM_ID,
    local_files_only: bool = True,
    freeze_vision: bool = True,
    stage: str = "a",
) -> Blip2QTR:
    _prefer_offline_hf_bert()
    from transformers import Blip2ForImageTextRetrieval
    from pathlib import Path

    pretrained_id = blip2_pretrained_id(pretrained_id)
    if Path(pretrained_id).is_dir():
        local_files_only = True

    # FP32 master weights for GradScaler.
    full = Blip2ForImageTextRetrieval.from_pretrained(
        pretrained_id, torch_dtype=torch.float32, local_files_only=local_files_only
    )
    vision = full.vision_model
    embeddings = full.embeddings
    qformer_c = full.qformer
    # Independent reranker Q-Former (same init) — key differentiator + freeze contract.
    qformer_r = copy.deepcopy(full.qformer)
    query_tokens_c = full.query_tokens
    # Learnable residual on composed queries for Stage-B (keeps DDP graph clean).
    query_tokens_r = nn.Parameter(torch.zeros_like(full.query_tokens))
    vision_projection = full.vision_projection
    text_projection = full.text_projection
    itm_head = full.itm_head
    # Drop unused refs on `full` so GC can free if any dangling.
    del full

    model = Blip2QTR(
        vision=vision,
        embeddings=embeddings,
        qformer_c=qformer_c,
        qformer_r=qformer_r,
        query_tokens_c=query_tokens_c,
        query_tokens_r=query_tokens_r,
        vision_projection=vision_projection,
        text_projection=text_projection,
        itm_head=itm_head,
    )
    if freeze_vision:
        model.freeze_vision()
    model.set_stage(stage)
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(
        "Blip2QTR ready (arch=%s stage=%s ckpt=%s trainable=%.2fM)",
        ARCH_TAG,
        stage,
        pretrained_id,
        n_train,
    )
    return model


def load_blip2_qtr_ckpt(
    model: Blip2QTR,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("BLIP2_QTR", state)
    msg = model.load_state_dict(sd, strict=False)
    model.to(device)
    return list(msg.missing_keys), list(msg.unexpected_keys)
