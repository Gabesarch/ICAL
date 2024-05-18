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

from SOLQ.util.misc import inverse_sigmoid
if args.do_deformable_atn_decoder or args.do_deformable_atn_encoder:
    print("Using deformable attention!")
    from SOLQ.models.ops.modules import MSDeformAttn
import functools
print = functools.partial(print, flush=True)

import ipdb
st = ipdb.set_trace

# from .rpe_attention import RPEMultiheadAttention, irpe

args.image_features_self_attend = True
args.rel_pos_encodings = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, normalize_before=False, text_encoder_type="roberta-base"):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels

        if args.init_obj_lang_query:
            # use object langauge features also as a query
            self.two_stage_num_proposals -= 1
            self.lang_query_embed = nn.Embedding(1, self.d_model) # pos encoding for language query

        self.single_atn_history_and_memory = args.single_atn_history_and_memory
        if args.within_demo and self.single_atn_history_and_memory:
            # indicate whether token belongs to history or memory (0=history, 1=mem)
            self.history_mem_embed = nn.Parameter(torch.Tensor(2, d_model))
            normal_(self.history_mem_embed)

        if args.image_features_self_attend:
            if args.do_deformable_atn_encoder:
                encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                                dropout, activation,
                                                                num_feature_levels, nhead, enc_n_points)
                encoder_norm = None #nn.LayerNorm(d_model) if normalize_before else None
                self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
            else:
                encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                        dropout, activation, normalize_before)
                encoder_norm = None #nn.LayerNorm(d_model) if normalize_before else None
                self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if args.do_deformable_atn_decoder:
            decoder_norm = None
            # assert False # TODO: deformable atn does not make sense for action query
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                            dropout, activation, normalize_before,
                                                            num_feature_levels, dec_n_points)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate_dec)
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.reference_points = nn.Linear(d_model, 4)
        
        print(f'Training with {activation}.')

        self._reset_parameters()

        if args.use_clip_text_encoder:
            self.text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-base-patch32")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32")
            if args.alfred_subgoals:
                self.tokenizer.add_tokens(['<<goal>>', '<<instr>>'])
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.max_text_length = 77 # clip max length is 77
            # config = self.text_encoder.config.text_config
        else:
            # Text Encoder
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                text_encoder_type)
            self.text_encoder = RobertaModel.from_pretrained(
                text_encoder_type)
            if args.alfred_subgoals:
                self.tokenizer.add_tokens(['<<goal>>', '<<instr>>'])
                self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.max_text_length = 512 # roberta max length is 512
            # config = self.text_encoder.config
        
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=0.1,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if args.do_deformable_atn_decoder or args.do_deformable_atn_encoder:
            for m in self.modules():
                if isinstance(m, MSDeformAttn):
                    m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals, num_pos_feats=192):
        # num_pos_feats = 128
        # num_pos_feats = 128
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
        query_embed,
        obj_srcs, 
        obj_masks, 
        obj_pos_embeds,
        text,
        B,
        S,
        action_pos_embeds,
        pos_embeds_3d, # img positional embed 3d 
        obj_poss_embeds_3d, # obj/history positional embed 3d
        ):
        # assert self.two_stage or query_embed is not None

        self.device = obj_srcs.device

        # Encode the text
        tokenized = self.tokenizer.batch_encode_plus(
            text, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt", 
            max_length=self.max_text_length
            ).to(self.device)
        encoded_text = self.text_encoder(**tokenized)
        text_memory = encoded_text.last_hidden_state #.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory = self.resizer(text_memory)  # B x seq X hidden_size
        dec_text_dict = {"feat":text_memory, "pos_enc":None, "mask":text_attention_mask} # text

        # image features at time step t self attend
        selfatn_out = self.self_attend_image(
                                            srcs, 
                                            masks, 
                                            pos_embeds, 
                                            dec_text_dict=dec_text_dict,
                                            pos_embeds_3d=pos_embeds_3d,
                                            )
        memory, mask_flatten, spatial_shapes, src_flatten, lvl_pos_embed_flatten, lvl_pos_embed_flatten_3d = selfatn_out

        # object features - flatten views + num objects to sequence dim
        src_flatten_obj = obj_srcs.flatten(1,2) # B x seq x hidden_size
        mask_flatten_obj = obj_masks.flatten(1,2) # B x seq
        lvl_pos_embed_flatten_obj = obj_pos_embeds.flatten(1,2) # B x seq x hidden_size
        if obj_poss_embeds_3d is not None:
            obj_poss_embeds_3d = obj_poss_embeds_3d.flatten(1,2) # B x seq x hidden_size

        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # prepare input for decoder
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        # hack implementation for two-stage Deformable DETR
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        
        if args.do_deformable_atn_decoder:
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
        else:
            reference_points = None

        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed_, tgt_ = torch.split(pos_trans_out, c, dim=2)
        
        if args.use_action_ghost_nodes:
            # query_embed = query_embed.clone() # need to clone to avoid in place 
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1).clone()
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1).clone()
            # tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
            query_embed += action_pos_embeds # positional encoding is 3D cos/sin translation/rotation positional encoding

        if args.do_decoder_2d3d:
            # keep action and object queries seperate
            query_embed_action = query_embed
            tgt_action = tgt
            query_embed = query_embed_
            tgt = tgt_

        init_reference_out = reference_points


        dec_query_dict = {
            "feat":tgt, 
            "pos_enc":query_embed, 
            "mask":None
            } # queries
        if args.do_decoder_2d3d:
            dec_query_dict["feat_action"] = tgt_action
            dec_query_dict["pos_enc_3d"] = query_embed_action
        dec_vis_dict = {
            "feat":memory, 
            "pos_enc":lvl_pos_embed_flatten, 
            "mask":mask_flatten, 
            "reference_points":reference_points, 
            "spatial_shapes":spatial_shapes,
            "level_start_index":level_start_index,
            "valid_ratios":valid_ratios,
            "pos_enc_3d":lvl_pos_embed_flatten_3d, # only used for do_decoder_2d3d
            } # image
        dec_obj_dict = {
            "feat":src_flatten_obj, 
            "pos_enc":lvl_pos_embed_flatten_obj, 
            "mask":mask_flatten_obj,
            "pos_enc_3d":obj_poss_embeds_3d # only used for do_decoder_2d3d
            } # object 
        

        if args.do_deformable_atn_decoder:
            hs, inter_references = self.decoder(
                dec_query_dict, # query
                dec_vis_dict, # visual
                dec_obj_dict, # history
                dec_text_dict, # text
                )
            inter_references_out = inter_references

        else:
            hs = self.decoder(
                dec_query_dict,
                dec_vis_dict,
                dec_obj_dict,
                dec_text_dict,
                )
            init_reference_out = None
            inter_references_out = None
        
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact #, seg_memory, seg_mask
        return hs, init_reference_out, inter_references_out, None, None #, seg_memory, seg_mask

    def self_attend_image(
        self,
        srcs,
        masks,
        pos_embeds,
        dec_text_dict=None,
        pos_embeds_3d=[],
        ):

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        lvl_pos_embed_flatten_3d = []
        spatial_shapes = []
        if False: #args.rel_pos_encodings:
            abs_pos = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            if len(pos_embeds_3d)>0:
                pos_embed_3d = pos_embeds_3d[lvl].flatten(2).transpose(1, 2)
                lvl_pos_embed_3d = pos_embed_3d + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten_3d.append(lvl_pos_embed_3d)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        if len(lvl_pos_embed_flatten_3d)>0:
            lvl_pos_embed_flatten_3d = torch.cat(lvl_pos_embed_flatten_3d, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        abs_pos = None

        if args.image_features_self_attend:
            
            # (1) encoder - self attention with image features
            if args.do_deformable_atn_encoder:
                memory = self.encoder(
                    src=src_flatten, 
                    spatial_shapes=spatial_shapes, 
                    level_start_index=level_start_index, 
                    valid_ratios=valid_ratios, 
                    pos=lvl_pos_embed_flatten, 
                    padding_mask=mask_flatten,
                    dec_text_dict=dec_text_dict
                    )
            else:
                memory = self.encoder(
                    src_flatten, 
                    src_key_padding_mask=mask_flatten, 
                    pos=lvl_pos_embed_flatten
                    )
        else:
            memory = src_flatten

        return memory, mask_flatten, spatial_shapes, src_flatten, lvl_pos_embed_flatten, lvl_pos_embed_flatten_3d

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=src.transpose(0, 1), attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2.transpose(0, 1))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=src2.transpose(0, 1), attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2.transpose(0, 1))
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

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
                dec_obj_dict,
                dec_text_dict,
                ):

        output = dec_query_dict["feat"]

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if args.do_deformable_atn_decoder:
                reference_points = dec_vis_dict["reference_points"]
                src_valid_ratios = dec_vis_dict["valid_ratios"]
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                dec_vis_dict["reference_points_input"] = reference_points_input

            output = layer(
                dec_query_dict,
                dec_vis_dict,
                dec_obj_dict,
                dec_text_dict,
            )
            dec_query_dict["feat"] = output
            
            if args.do_deformable_atn_decoder:
                # hack implementation for iterative bounding box refinement
                if self.bbox_embed is not None:
                    tmp = self.bbox_embed[lid](output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp
                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()
                    dec_vis_dict["reference_points"] = reference_points

            if args.do_decoder_2d3d:
                output_a = layer(
                        dec_query_dict,
                        dec_vis_dict,
                        dec_obj_dict,
                        dec_text_dict,
                        pos_enc_label="pos_enc_3d",
                    )
                dec_query_dict["feat_action"] = output_a
                output = torch.cat([output, output_a], dim=1)

            if self.return_intermediate:
                intermediate.append(output)
                if args.do_deformable_atn_decoder:
                    intermediate_reference_points.append(reference_points)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if args.do_deformable_atn_decoder:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points)
            else:
                return torch.stack(intermediate)

        if args.do_deformable_atn_decoder:
            return output, reference_points
        else:
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
        if args.do_deformable_atn_decoder:
            self.multihead_attn_v = MSDeformAttn(d_model, n_levels, nhead, n_points)
            self.dropout_v = nn.Dropout(dropout)
            self.norm_v = nn.LayerNorm(d_model)
            if args.do_decoder_2d3d:
                # for cross atn to image for action queries
                self.multihead_attn_a = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
                self.dropout_a = nn.Dropout(dropout)
                self.norm_a = nn.LayerNorm(d_model)
        else:
            self.multihead_attn_v = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.dropout_v = nn.Dropout(dropout)
            self.norm_v = nn.LayerNorm(d_model)

        # Cross attention queries -> language
        self.multihead_attn_l = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self.dropout_l = nn.Dropout(dropout)
        self.norm_l = nn.LayerNorm(d_model)

        # Cross attention queries -> object history
        self.multihead_attn_o = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )
        self.dropout_o = nn.Dropout(dropout)
        self.norm_o = nn.LayerNorm(d_model)

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
        dec_obj_dict,
        dec_text_dict,
        pos_enc_label="pos_enc",
        ):
        '''
        pos_enc_label can be pos_enc or pos_enc_3d when history + image features have 3d pos encoding
        '''
        if pos_enc_label=="pos_enc":
            # object queries
            tgt = dec_query_dict["feat"]
        elif pos_enc_label=="pos_enc_3d":
            # action queries
            tgt = dec_query_dict["feat_action"]
        else:
            assert(False)

        # self attention queries 
        q = k = self.with_pos_embed(tgt, dec_query_dict[pos_enc_label])
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=tgt.transpose(0, 1), attn_mask=None,
                            key_padding_mask=dec_query_dict["mask"])[0]
        tgt = tgt + self.dropout1(tgt2.transpose(0, 1))
        tgt = self.norm1(tgt)
        
        # Cross attention queries -> language
        tgt2 = self.multihead_attn_l(query=self.with_pos_embed(tgt, dec_query_dict[pos_enc_label]).transpose(0, 1),
                                key=self.with_pos_embed(dec_text_dict["feat"], dec_text_dict["pos_enc"]).transpose(0, 1),
                                value=dec_text_dict["feat"].transpose(0, 1), attn_mask=None,
                                key_padding_mask=dec_text_dict["mask"])[0]
        tgt = tgt + self.dropout_l(tgt2.transpose(0, 1))
        tgt = self.norm_l(tgt)

        # Cross attention queries -> object history
        tgt2 = self.multihead_attn_o(query=self.with_pos_embed(tgt, dec_query_dict[pos_enc_label]).transpose(0, 1),
                                key=self.with_pos_embed(dec_obj_dict["feat"], dec_obj_dict[pos_enc_label]).transpose(0, 1),
                                value=dec_obj_dict["feat"].transpose(0, 1), attn_mask=None,
                                key_padding_mask=dec_obj_dict["mask"])[0]
        tgt = tgt + self.dropout_o(tgt2.transpose(0, 1))
        tgt = self.norm_o(tgt)

        # cross attention queries -> image
        if args.do_deformable_atn_decoder:
            if pos_enc_label=="pos_enc":
                tgt2 = self.multihead_attn_v(self.with_pos_embed(tgt, dec_query_dict["pos_enc"]),
                                dec_vis_dict["reference_points_input"],
                                dec_vis_dict["feat"], 
                                dec_vis_dict["spatial_shapes"], 
                                dec_vis_dict["level_start_index"], 
                                dec_vis_dict["mask"]
                                )
                tgt = tgt + self.dropout_v(tgt2)
                tgt = self.norm_v(tgt)
            elif pos_enc_label=="pos_enc_3d": # action queries use regular attention + 3d encodings for do_decoder_2d3d
                tgt2 = self.multihead_attn_a(
                                    query=self.with_pos_embed(tgt, dec_query_dict["pos_enc_3d"]).transpose(0, 1),
                                    key=self.with_pos_embed(dec_vis_dict["feat"], dec_vis_dict["pos_enc_3d"]).transpose(0, 1),
                                    value=dec_vis_dict["feat"].transpose(0, 1), 
                                    attn_mask=None,
                                    key_padding_mask=dec_vis_dict["mask"])[0]
                tgt = tgt + self.dropout_a(tgt2.transpose(0, 1))
                tgt = self.norm_a(tgt)
            else:
                assert(False) # wrong pos encoding label

        else:
            tgt2 = self.multihead_attn_v(query=self.with_pos_embed(tgt, dec_query_dict["pos_enc"]).transpose(0, 1),
                                    key=self.with_pos_embed(dec_vis_dict["feat"], dec_vis_dict["pos_enc"]).transpose(0, 1),
                                    value=dec_vis_dict["feat"].transpose(0, 1), attn_mask=None,
                                    key_padding_mask=dec_vis_dict["mask"])[0]
            tgt = tgt + self.dropout_v(tgt2.transpose(0, 1))
            tgt = self.norm_v(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


############# DEFORMABLE ENCODER ###############
class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self, 
        src, 
        spatial_shapes, 
        level_start_index, 
        valid_ratios, 
        pos=None, 
        padding_mask=None,
        dec_text_dict=None,
        ):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(
                output, 
                pos, 
                reference_points, 
                spatial_shapes, 
                level_start_index, 
                padding_mask,
                dec_text_dict
                )

        return output

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        if args.vision_lang_ca_encoder:
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
        reference_points, 
        spatial_shapes, 
        level_start_index, 
        padding_mask=None,
        dec_text_dict=None
        ):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # cross vision --> lang
        # important for two-stage to get info about intermediate objects
        if args.vision_lang_ca_encoder and dec_text_dict is not None:
            src2 = self.multihead_attn_l(query=self.with_pos_embed(src, pos).transpose(0, 1),
                                    key=self.with_pos_embed(dec_text_dict["feat"], dec_text_dict["pos_enc"]).transpose(0, 1),
                                    value=dec_text_dict["feat"].transpose(0, 1), attn_mask=None,
                                    key_padding_mask=dec_text_dict["mask"])[0]
            src = src + self.dropout_l(src2.transpose(0, 1))
            src = self.norm_l(src)

        # ffn
        src = self.forward_ffn(src)

        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def swish(x):
    return x * torch.sigmoid(x)


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


def build_deforamble_transformer():
    return DeformableTransformer(
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
        text_encoder_type=args.text_encoder_type
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
