# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

# from .util import box_ops
# from .util.plot_utils import plot_3d, plot_object_tracks
from .util.misc import NestedTensor, nested_tensor_from_tensor_list, interpolate, inverse_sigmoid
from .position_encoding import PositionalEncoding3D, PositionEmbeddingLearned3D
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import numpy as np
import cv2
from .backbone import build_backbone
# from .matcher import build_matcher
# from .segmentation import PostProcessPanoptic
from .transformer import build_transformer
# from .dct import ProcessorDCT
# from detectron2.layers import paste_masks_in_image
# from detectron2.utils.memory import retry_if_cuda_oom
import copy
import functools
import time
from torchvision.ops import roi_pool, roi_align
import utils.geom
# from .pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

import ipdb
st = ipdb.set_trace
from arguments import args
from .losses import SetCriterion
import matplotlib.pyplot as plt
print = functools.partial(print, flush=True)

from torch.autograd import Variable

old_actions = True

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check_zeros(tensor, value=0):
    nonzero = ~torch.all(tensor.reshape(tensor.shape[0], -1)==value, dim=1)
    return nonzero

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class REPLAY_NETWORK(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, 
                backbone, 
                transformer, 
                num_classes, 
                num_queries, 
                num_feature_levels,
                num_actions, 
                aux_loss=True, 
                with_box_refine=False, 
                two_stage=False, 
                with_vector=False, 
                processor_dct=None, 
                vector_hidden_dim=256, 
                actions2idx=None, 
                ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_classes = num_classes
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.transformer.num_actions = num_actions
        self.class_embed = nn.Linear(hidden_dim, 128)
        self.actions2idx = actions2idx
        self.device = device

        # policy specific params
        self.args = args
        
        
        self.num_feature_levels = num_feature_levels
        self.num_movement_actions = sum(("Move" in k or "Rotate" in k or "Look" in k) for k in list(self.actions2idx.keys())) 
        
        if args.query_for_each_object:
            self.action_embed = nn.Linear(hidden_dim, 1)
            self.label_embed = nn.Linear(hidden_dim, 1)
            self.query_embed = nn.Embedding(num_actions + num_classes, hidden_dim*2)
        else:
            self.action_embed = nn.Linear(hidden_dim, 1)
            self.label_embed = nn.Linear(hidden_dim, num_classes)
            self.query_embed = nn.Embedding(num_actions + 1, hidden_dim*2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            input_proj_list = input_proj_list[::-1] # reverse because we want end first
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            if args.reduce_final_layer:
                in_channels = backbone.num_channels[-1]
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )])
            else:
                self.input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.visualize = False

        # custom weight initialization
        self.init()
        

    def init(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (self.transformer.decoder.num_layers + 1) if self.two_stage else self.transformer.decoder.num_layers
        self.action_embed = _get_clones(self.action_embed, num_pred)
        self.label_embed = _get_clones(self.label_embed, num_pred)

    def forward(
        self, 
        samples, 
        ):

        """ The forward expects a NestedTensor, which consists of:
               - samples: batched images, of shape [batch_size x nviews x 3 x H x W]
               - instance_masks: object masks in list batch, nviews, HxW

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        self.device = samples.device
        B, S, C, H, W = samples.shape
        self.B = B
        self.pack_dims = lambda x: utils.basic.pack_seqdim(x, B) # combines batch and sequence 
        self.unpack_dims = lambda x: utils.basic.unpack_seqdim(x, B) # seperates batch and sequence

        history_frame_inds = np.arange(samples.shape[1])

        # get image and object features from the backbone
        srcs_t, masks_t, poss_t = self.get_visual_features(
                samples
            )

        # get query embeds
        query_embeds = self.query_embed.weight
        action_tgt, action_pos_enc = self.get_ghost_nodes_query_embed(query_embeds)

        # encoder, decoder
        hs = self.transformer(
            srcs_t, masks_t, poss_t,
            action_tgt, action_pos_enc,
        )
        if args.query_for_each_object:
            hs_actions = hs[:,:,:self.num_actions]
            hs_labels = hs[:,:,self.num_actions:]
        else:
            hs_actions = hs[:,:,:-1]
            hs_labels = hs[:,:,-1:]

        out = self.extract_outputs(hs_actions, hs_labels)

        return out
    
    def get_ghost_nodes_query_embed(self, query_embeds):

        # extract value embedding + add learned position encoding to 3d positional encoding 
        query_embeds, tgt = torch.split(query_embeds, self.hidden_dim, dim=1)
        query_embeds = query_embeds.unsqueeze(0).expand(self.B, -1, -1)
        tgt = tgt.unsqueeze(0).expand(self.B, -1, -1)
        action_pos_enc = query_embeds 
        
        return tgt, action_pos_enc

    def get_visual_features(
            self, 
            samples, 
        ):

        B, S, C, H, W = samples.shape

        # remove images that are used for padding (placeholder images)
        samples = self.pack_dims(samples).contiguous().unbind(0)

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        features = features[::-1]
        pos = pos[::-1]

        num_feature_levs = len(features)

        srcs_t = []
        poss_t = []
        masks_t = []
        
        for l in range(self.num_feature_levels):

            #####%%%% Extract multiscale image features %%%####
            if (l > num_feature_levs - 1) or (num_feature_levs==1 and args.reduce_final_layer):
                if l == num_feature_levs or (num_feature_levs==1 and args.reduce_final_layer):
                    src = self.input_proj[l](features[0].tensors)
                W_l, H_l = src.shape[-2:]
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # mask = torch.ones((B*S,W_l,H_l), dtype=torch.bool).to(device)
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            else:
                feat = features[l]
                pos_l = pos[l]
                src, mask = feat.decompose()
                src = self.input_proj[l](src)
            _, C_l, W_l, H_l = src.shape

            ####%%% Extract object features %%%#####            
            _, C_pos_l, _, _ = pos_l.shape

            src = src.reshape(B,S,C_l,W_l,H_l)
            mask = mask.reshape(B,S,W_l,H_l)
            pos_l = pos_l.reshape(B,S,C_l,W_l,H_l)

            # 2D image features
            srcs_t.append(src)
            masks_t.append(mask)
            poss_t.append(pos_l)

        return srcs_t, masks_t, poss_t

    def extract_outputs(
        self, 
        hs_actions,
        hs_labels,
        ):
        outputs_actions = []
        outputs_labels = []
        for lvl in range(hs_actions.shape[0]):
            outputs_action = self.action_embed[lvl](hs_actions[lvl])
            outputs_actions.append(outputs_action)
            outputs_label = self.label_embed[lvl](hs_labels[lvl])
            outputs_labels.append(outputs_label)
        outputs_actions = torch.stack(outputs_actions)
        outputs_labels = torch.stack(outputs_labels)
        
        outputs_actions = outputs_actions.transpose(3,2)
        outputs_labels = outputs_labels.transpose(3,2)

        out = {
            'pred_actions': outputs_actions[-1],
            'pred_logits': outputs_labels[-1],
            }
        if self.aux_loss:
            out_aux = {
                'pred_actions': outputs_actions,
                'pred_logits': outputs_labels,
                }
            out['aux_outputs'] = self._set_aux_loss(out_aux)

        return out                

    @torch.jit.unused
    def _set_aux_loss(self, out_aux):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        out = []
        for l in range(out_aux[list(out_aux.keys())[0]].shape[0]-1):
            out_ = {}
            for k in out_aux.keys():
                out_[k] = out_aux[k][l]
            out.append(out_)
        return out

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, outputs, target_sizes, do_masks=True, return_features=False, features=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """        
        out_logits, out_bbox, out_vector, out_interms, out_actions = None, None, None, None, outputs['pred_actions']

        if args.remove_done_from_action_prediction:
            out_actions = out_actions[:,:,:-1]
        
        prob_actions = out_actions.squeeze(1).softmax(1)
        labels_action = torch.argmax(prob_actions, dim=1)

        if return_features and features is not None:
            features_keep = torch.gather(features, 1, topk_boxes.unsqueeze(-1).repeat(1,1,features.shape[-1]))

        results1 = [{'labels_action':a} for a in zip(labels_action)]

        results = {'pred1':results1}

        return results

class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5, processor_dct=None):
        super().__init__()
        self.threshold = threshold
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(num_classes, num_actions, actions2idx):
    device = torch.device(args.device)

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_transformer() if not args.checkpoint else build_cp_deforamble_transformer()
    model = REPLAY_NETWORK(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_actions=num_actions,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_vector=args.with_vector, 
        vector_hidden_dim=args.vector_hidden_dim,
        actions2idx=actions2idx,
    )

    weight_dict = {
        'loss_class_ce': args.cls_loss_coef, 
        'loss_action_ce': args.action_loss_coef,
        }
    print(f"Loss weights:\n: {weight_dict}")
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["action", "labels"]
    
    criterion = SetCriterion(
        num_classes, 
        weight_dict, 
        losses, 
        focal_alpha=args.focal_alpha, 
        with_vector=args.with_vector, 
        vector_loss_coef=args.vector_loss_coef,
        no_vector_loss_norm=args.no_vector_loss_norm,
        vector_start_stage=args.vector_start_stage)
    criterion.to(device)
    postprocessors = {}

    return model, criterion, postprocessors