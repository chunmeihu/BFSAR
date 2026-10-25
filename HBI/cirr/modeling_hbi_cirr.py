"""
HBICIRR: HBI adapted for CIRR text-to-image retrieval with ViT-L/14 and multi-crop CLS.
Does not modify HBI/models/modeling.py; inherits forward / interaction methods.
"""

import os
from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from HBI.models.modeling import HBI as _HBI
from HBI.models.banzhaf import BanzhafModule
from HBI.models.cluster import CTM, TCBlock
from HBI.models.until_module import CrossEn, MSE, ArcCrossEn, KL
from HBI.cirr.clip_loader import build_clip_from_config


class HBICIRR(_HBI):
    """HBI with external CLIP checkpoint and CTM dims matching transformer_width (768 for ViT-L/14)."""

    def __init__(self, config):
        nn.Module.__init__(self)

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, "agg_module", "None")

        self.clip, meta = build_clip_from_config(config)
        state_dict = meta.state_dict
        transformer_width = meta.transformer_width
        transformer_heads = meta.transformer_heads
        context_length = meta.context_length

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": transformer_width,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": transformer_width,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        self.cross_config = cross_config

        if self.interaction == "wti":
            self.text_weight_fc = nn.Sequential(
                nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
                nn.Linear(2 * transformer_width, 1))
            self.video_weight_fc = nn.Sequential(
                nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
                nn.Linear(2 * transformer_width, 1))
            self.text_weight_fc0 = nn.Sequential(
                nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
                nn.Linear(2 * transformer_width, 1))
            self.video_weight_fc0 = nn.Sequential(
                nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
                nn.Linear(2 * transformer_width, 1))
            self.text_weight_fc1 = nn.Sequential(
                nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
                nn.Linear(2 * transformer_width, 1))
            self.video_weight_fc1 = nn.Sequential(
                nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
                nn.Linear(2 * transformer_width, 1))

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            from HBI.models.module_cross import Transformer as TransformerClip
            self.frame_position_embeddings = nn.Embedding(
                cross_config.max_position_embeddings, cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(
                    width=transformer_width,
                    layers=config.num_hidden_layers,
                    heads=transformer_heads,
                )
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(
                    input_size=cross_config.hidden_size,
                    hidden_size=cross_config.hidden_size,
                    batch_first=True,
                    bidirectional=False,
                    num_layers=1,
                )

        self.loss_fct = CrossEn(config)
        self.loss_arcfct = ArcCrossEn(margin=10)
        self.banzhafmodel = BanzhafModule(64)
        self.banzhafmodel0 = BanzhafModule(64)
        self.banzhafmodel1 = BanzhafModule(64)
        self.banzhafteacher = BanzhafModule(64)

        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)

        estimator_path = getattr(config, "estimator", None)
        if estimator_path and str(estimator_path) != "None" and os.path.isfile(estimator_path):
            teacher_state = torch.load(estimator_path, map_location="cpu")
            self.banzhafteacher.load_state_dict(teacher_state, strict=False)

        dim = transformer_width
        self.t_ctm0 = CTM(sample_ratio=config.t_rate0, embed_dim=dim, dim_out=dim, k=3)
        self.t_block0 = TCBlock(dim=dim, num_heads=8)
        self.t_ctm1 = CTM(sample_ratio=config.t_rate1, embed_dim=dim, dim_out=dim, k=3)
        self.t_block1 = TCBlock(dim=dim, num_heads=8)
        self.v_ctm0 = CTM(sample_ratio=config.v_rate0, embed_dim=dim, dim_out=dim, k=3)
        self.v_block0 = TCBlock(dim=dim, num_heads=8)
        self.v_ctm1 = CTM(sample_ratio=config.v_rate1, embed_dim=dim, dim_out=dim, k=3)
        self.v_block1 = TCBlock(dim=dim, num_heads=8)

        self.mse = MSE()
        self.kl = KL()

        if getattr(config, "gradient_checkpointing", 0):
            self._enable_visual_gradient_checkpointing()

        new_state_dict = OrderedDict()
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = any(
                "frame_position_embeddings" in k for k in state_dict.keys()
            )
            if not contain_frame_position:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                    elif self.agg_module == "seqTransf" and key.startswith("transformer.resblocks"):
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
        self.load_state_dict(new_state_dict, strict=False)

    def get_image_feat(self, image, image_mask, shaped=False):
        return self.get_video_feat(image, image_mask, shaped=shaped)

    def _enable_visual_gradient_checkpointing(self):
        """Trade compute for memory on ViT visual backbone (CIRR-only hook)."""
        from torch.utils.checkpoint import checkpoint

        visual = self.clip.visual
        resblocks = visual.transformer.resblocks

        def forward_with_checkpoint(x):
            for block in resblocks:
                x = checkpoint(block, x, use_reentrant=False)
            return x

        visual.transformer.forward = forward_with_checkpoint

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        micro = int(getattr(self.config, "image_encode_microbatch", 0) or 0)
        if micro > 0 and video.size(0) > micro:
            chunks = []
            for start in range(0, video.size(0), micro):
                chunk = video[start:start + micro]
                hidden = self.clip.encode_image(chunk, return_hidden=True)[0].float()
                chunks.append(hidden)
            video_feat = torch.cat(chunks, dim=0)
        else:
            video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.view(bs_pair, -1, video_feat.size(-1))
        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def _should_allgather(self):
        return (
            torch.cuda.is_available()
            and getattr(self.config, "distributed", 0)
            and torch.distributed.is_initialized()
        )

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        from HBI.models.until_module import AllGather
        allgather = AllGather.apply

        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, cls = self.get_text_video_feat(
            text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if self._should_allgather():
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()

            logit_scale = self.clip.logit_scale.exp()
            loss = 0.0

            t_idx_token = torch.arange(text_feat.size(1))[None, :].repeat(text_feat.size(0), 1)
            t_agg_weight = text_feat.new_ones(text_feat.size(0), text_feat.size(1), 1)
            t_token_dict = {
                "x": text_feat, "token_num": text_feat.size(1),
                "idx_token": t_idx_token, "agg_weight": t_agg_weight, "mask": text_mask.detach(),
            }
            v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
            v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
            v_token_dict = {
                "x": video_feat, "token_num": video_feat.size(1),
                "idx_token": v_idx_token, "agg_weight": v_agg_weight, "mask": video_mask.detach(),
            }

            M_t2v_logits, M_v2t_logits, logits = self.entity_level(
                text_feat, cls, video_feat, text_mask, video_mask)
            M_loss = (self.loss_fct(M_t2v_logits * logit_scale) + self.loss_fct(M_v2t_logits * logit_scale)) / 2
            logits = torch.diagonal(logits, dim1=0, dim2=1).permute(2, 0, 1)
            banzhaf = self.banzhafmodel(logits.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                teacher = self.banzhafteacher(logits.unsqueeze(1).clone().detach()).squeeze(1).detach()
                teacher = torch.einsum("btv,bt->btv", [teacher, text_mask])
                teacher = torch.einsum("btv,bv->btv", [teacher, video_mask])
            banzhaf = torch.einsum("btv,bt->btv", [banzhaf, text_mask])
            banzhaf = torch.einsum("btv,bv->btv", [banzhaf, video_mask])
            loss += M_loss + self.config.skl * (self.kl(banzhaf, teacher) + self.kl(banzhaf.T, teacher.T))

            # Ablation A3: train entity level only (skip action/event + hierarchy KL).
            wti_levels = int(getattr(self.config, "wti_levels", 3) or 3)
            if wti_levels <= 1:
                zero = loss.new_zeros(())
                return loss, zero, zero

            t_token_dict = self.t_block0(self.t_ctm0(t_token_dict))
            v_token_dict = self.v_block0(self.v_ctm0(v_token_dict))
            text_feat, video_feat = t_token_dict["x"], v_token_dict["x"]
            M_t2v_logits0, M_v2t_logits0, logits = self.action_level(
                text_feat, cls, video_feat, text_mask, video_mask, t_token_dict, v_token_dict)
            M_loss = (self.loss_fct(M_t2v_logits0 * logit_scale) + self.loss_fct(M_v2t_logits0 * logit_scale)) / 2
            logits = torch.diagonal(logits, dim1=0, dim2=1).permute(2, 0, 1)
            banzhaf = self.banzhafmodel0(logits.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                teacher = self.banzhafteacher(logits.unsqueeze(1).clone().detach()).squeeze(1).detach()
            s_loss = self.kl(banzhaf, teacher) + self.kl(banzhaf.T, teacher.T)
            loss += M_loss + self.config.skl * s_loss
            loss1 = M_loss + self.config.skl * s_loss

            t_token_dict = self.t_block1(self.t_ctm1(t_token_dict))
            v_token_dict = self.v_block1(self.v_ctm1(v_token_dict))
            text_feat, video_feat = t_token_dict["x"], v_token_dict["x"]
            M_t2v_logits1, M_v2t_logits1, logits = self.event_level(
                text_feat, cls, video_feat, text_mask, video_mask, t_token_dict, v_token_dict)
            M_loss = (self.loss_fct(M_t2v_logits1 * logit_scale) + self.loss_fct(M_v2t_logits1 * logit_scale)) / 2
            logits = torch.diagonal(logits, dim1=0, dim2=1).permute(2, 0, 1)
            banzhaf = self.banzhafmodel1(logits.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                teacher = self.banzhafteacher(logits.unsqueeze(1).clone().detach()).squeeze(1).detach()
            s_loss = self.kl(banzhaf, teacher) + self.kl(banzhaf.T, teacher.T)
            loss += M_loss + self.config.skl * s_loss
            loss2 = M_loss + self.config.skl * s_loss

            loss += self.config.kl * (
                self.kl(M_v2t_logits0, M_v2t_logits) + self.kl(M_v2t_logits1, M_v2t_logits)
                + self.kl(M_t2v_logits0, M_t2v_logits) + self.kl(M_t2v_logits1, M_t2v_logits))
            return loss, loss1, loss2
        return None
