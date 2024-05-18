# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

from arguments import args

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from transformers import RobertaModel, RobertaTokenizerFast
from transformers import CLIPTokenizer, CLIPTextModel

import utils.basic

from .util.misc import inverse_sigmoid
import functools
print = functools.partial(print, flush=True)

import ipdb
st = ipdb.set_trace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "silu":
        return swish
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Transformer(nn.Module):
    def __init__(self, 
                d_model=256, 
                nhead=8,
                num_encoder_layers=6, 
                num_decoder_layers=6, 
                dim_feedforward=1024, 
                dropout=0.1,
                activation="relu", 
                return_intermediate_dec=False,
                num_feature_levels=4, 
                dec_n_points=4,  
                enc_n_points=4,
                two_stage=False, 
                two_stage_num_proposals=300, 
                normalize_before=False,
                use_memory=False
                ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels
        self.use_memory = use_memory

        encoder_layer = TransformerEncoderLayer(
            d_model, 
            dim_feedforward,
            dropout, 
            activation,
            num_feature_levels, 
            nhead, 
            enc_n_points
            )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_norm = None
        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            num_feature_levels,
            dec_n_points,
        )

        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers,
            decoder_norm, return_intermediate_dec
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.reference_points = nn.Linear(d_model, 4)
        
        print(f'Training with {activation}.')

        self.frame_embed = nn.Parameter(torch.Tensor(2, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)
        normal_(self.frame_embed)

    def get_proposal_pos_embed(self, proposals, num_pos_feats=192):
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self, 
        srcs, 
        masks, 
        pos_embeds, 
        action_tgt,
        action_pos_embeds, 
        ):

        # image features at time step t self attend
        selfatn_out = self.self_attend_visual(
            srcs, masks, pos_embeds, 
        )
        memory, mask_flatten, lvl_pos_embed_flatten = selfatn_out

        dec_query_dict = {
            "mask": None,
            "feat_action": action_tgt,
            "pos_enc": action_pos_embeds,
            } # queries
        dec_vis_dict = {
            "feat": memory, 
            "mask": mask_flatten, 
            "pos_enc": lvl_pos_embed_flatten
            } # image
        
        output = self.decoder(
            dec_query_dict, # query
            dec_vis_dict, # visual
        )
        
        return output


    def self_attend_visual(
        self,
        srcs,
        masks,
        pos_embeds,
        ):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            src = src.flatten(3).transpose(2, 3)
            mask = mask.flatten(2)
            pos_embed = pos_embed.flatten(3).transpose(2, 3)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)
            lvl_pos_embed = lvl_pos_embed + self.frame_embed.view(1, 2, 1, -1) # 2 frames
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 2).flatten(1,2)
        mask_flatten = torch.cat(mask_flatten, 2).flatten(1,2)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2).flatten(1,2)
        
        memory = self.encoder(
            src=src_flatten, 
            pos=lvl_pos_embed_flatten, 
            padding_mask=mask_flatten,
            )

        return memory, mask_flatten, lvl_pos_embed_flatten

############# DECODER ###############
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, 
                dec_query_dict,
                dec_vis_dict,
                ):

        intermediate = []
        for lid, layer in enumerate(self.layers):

            output_a = layer(
                    dec_query_dict,
                    dec_vis_dict,
                )
            dec_query_dict["feat_action"] = output_a
            output = output_a

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 n_levels=4, n_points=4):
        super().__init__()

        # self attention queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # cross attention queries -> image
        self.multihead_attn_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout_v = nn.Dropout(dropout)
        self.norm_v = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self, 
        dec_query_dict,
        dec_vis_dict,
        pos_enc_label="pos_enc",
    ):
        '''
        pos_enc_label can be pos_enc or pos_enc_3d when history + image features have 3d pos encoding
        '''

        tgt = dec_query_dict["feat_action"] 

        # Cross attention queries -> images
        tgt2 = self.multihead_attn_v(query=self.with_pos_embed(tgt, dec_query_dict[pos_enc_label]).transpose(0, 1),
                                key=self.with_pos_embed(dec_vis_dict["feat"], dec_vis_dict["pos_enc"]).transpose(0, 1),
                                value=dec_vis_dict["feat"].transpose(0, 1), 
                                attn_mask=None,
                                key_padding_mask=dec_vis_dict["mask"])[0]
        tgt = tgt + self.dropout_v(tgt2.transpose(0, 1))
        tgt = self.norm_v(tgt) 

        # self attention queries 
        q = k = self.with_pos_embed(tgt, dec_query_dict[pos_enc_label])
        v = self.with_pos_embed(tgt, dec_query_dict[pos_enc_label])
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=v.transpose(0, 1), attn_mask=None,
                            key_padding_mask=dec_query_dict["mask"])[0]
        tgt = tgt + self.dropout1(tgt2.transpose(0, 1))
        tgt = self.norm1(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


############# DEFORMABLE ENCODER ###############
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self, 
        src, 
        pos=None, 
        padding_mask=None,
        ):
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(
                output, 
                pos, 
                padding_mask,
                )

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention queries - not deformable
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention from vision to lang
        self.multihead_attn_l = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.dropout_l = nn.Dropout(dropout)
        self.norm_l = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, 
        src, 
        pos, 
        padding_mask=None,
        ):

        # self attention - not deformable
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=src.transpose(0, 1), attn_mask=None,
                            key_padding_mask=padding_mask)[0]
        src = src + self.dropout1(src2.transpose(0, 1))
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def swish(x):
    return x * torch.sigmoid(x)


def build_transformer():
    return Transformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        )

######### TEXT MODULES #########
class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output