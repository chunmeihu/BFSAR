"""Banzhaf interaction estimator for CIRR (Stage 1)."""

import os
from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from HBI.models.banzhaf import BanzhafModule, BanzhafInteraction
from HBI.models.until_module import CrossEn, MSE, ArcCrossEn, KL, AllGather
from HBI.cirr.clip_loader import build_clip_from_config

allgather = AllGather.apply


class HBICIRREstimator(nn.Module):
    def __init__(self, config):
        super(HBICIRREstimator, self).__init__()

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
        self.banzhafteacher = BanzhafModule(64)
        num_crops = getattr(config, "max_frames", 12)
        mc_samples = getattr(config, "banzhaf_mc_samples", 100)
        self.banzhafinteraction = BanzhafInteraction(config.max_words, num_crops, mc_samples)

        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)
        self.mse = MSE()

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

        for param in self.clip.parameters():
            param.requires_grad = False
        if self.agg_module == "seqTransf":
            for param in self.transformerClip.parameters():
                param.requires_grad = False
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            for param in self.frame_position_embeddings.parameters():
                param.requires_grad = False
        if self.interaction == "wti":
            for param in self.text_weight_fc.parameters():
                param.requires_grad = False
            for param in self.video_weight_fc.parameters():
                param.requires_grad = False

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)

        text_feat, video_feat, cls = self.get_text_video_feat(
            text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if (
                torch.cuda.is_available()
                and getattr(self.config, "distributed", 0)
                and torch.distributed.is_initialized()
            ):
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()

            logits, text_weight, video_weight = self.entity_level(
                text_feat, cls, video_feat, text_mask, video_mask)
            logits = torch.diagonal(logits, dim1=0, dim2=1).permute(2, 0, 1)
            y = self.banzhafinteraction(
                logits.clone().detach(), text_mask, video_mask, text_weight, video_weight
            ).detach()
            p = self.banzhafteacher(logits.unsqueeze(1)).squeeze(1)
            p = torch.einsum("btv,bt->btv", [p, text_mask])
            p = torch.einsum("btv,bv->btv", [p, video_mask])
            return self.mse(y, p)
        return None

    def entity_level(self, text_feat, cls, video_feat, text_mask, video_mask):
        if self.config.interaction == "wti":
            text_weight = self.text_weight_fc(text_feat).squeeze(2)
            text_weight.masked_fill_((1 - text_mask).to(torch.bool), float(-9e15))
            text_weight = torch.softmax(text_weight, dim=-1)
            video_weight = self.video_weight_fc(video_feat).squeeze(2)
            video_weight.masked_fill_((1 - video_mask).to(torch.bool), float(-9e15))
            video_weight = torch.softmax(video_weight, dim=-1)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum("atd,bvd->abtv", [text_feat, video_feat])
        retrieve_logits = torch.einsum("abtv,at->abtv", [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum("abtv,bv->abtv", [retrieve_logits, video_mask])
        return retrieve_logits, text_weight, video_weight

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if not shaped:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
        bs_pair = text_ids.size(0)
        cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        cls, text_feat = cls.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        cls = cls.view(bs_pair, -1, cls.size(-1)).squeeze(1)
        return text_feat, cls

    def get_video_feat(self, video, video_mask, shaped=False):
        if not shaped:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
        bs_pair, _ = video_mask.size()
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.view(bs_pair, -1, video_feat.size(-1))
        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if not shaped:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
        text_feat, cls = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)
        return text_feat, video_feat, cls

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            return video_feat
        if agg_module == "seqLSTM":
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(
                video_feat, torch.sum(video_mask, dim=-1).cpu(),
                batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training:
                self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            return video_feat + video_feat_original
        if agg_module == "seqTransf":
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)
            return video_feat + video_feat_original
        return video_feat

    def init_weights(self, module):
        from HBI.models.until_module import LayerNorm
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if hasattr(module, "beta") and hasattr(module, "gamma"):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
