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

from SOLQ.util import box_ops
from SOLQ.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .position_encoding import PositionalEncoding3D, PositionEmbeddingLearned3D
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import numpy as np
import cv2
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_stp import build_deforamble_transformer
from .dct import ProcessorDCT
from detectron2.structures import BitMasks
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
import copy
import functools
import time
# from torchvision.ops import RoIPool
from torchvision.ops import roi_pool, roi_align
import utils.geom

import ipdb
st = ipdb.set_trace
from arguments import args
print = functools.partial(print, flush=True)

from torch.autograd import Variable

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check_zeros(tensor):
    nonzero = ~torch.all(tensor.reshape(tensor.shape[0], -1)==0, dim=1)
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

class SOLQ(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 num_actions, aux_loss=True, with_box_refine=False, two_stage=False, with_vector=False, 
                 processor_dct=None, vector_hidden_dim=256, actions2idx=None):
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
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.transformer.num_actions = num_actions
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.actions2idx = actions2idx

        # policy specific params
        self.args = args

        if self.args.use_3d_img_pos_encodings:
            self.pix_T_camX = self.get_pix_T_camX()
            
        # extra losses
        self.intermediate_embed = nn.Linear(hidden_dim, 2)
        if self.args.query_per_action:
            self.action_embed = nn.Linear(hidden_dim, 1)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)
        self.num_feature_levels = num_feature_levels

        self.multiscale_query_proj = nn.Linear(hidden_dim*num_feature_levels, hidden_dim)

        if self.args.use_3d_pos_enc:
            self.step_size = args.STEP_SIZE
            self.dt = np.radians(args.DT)
            self.horizon_dt = np.radians(args.HORIZON_DT)
            self.pitch_range = np.radians(args.pitch_range)
            self.pos_enc_3d = PositionEmbeddingLearned3D(5, hidden_dim)
        
        if not self.args.use_3d_pos_enc or self.args.do_decoder_2d3d:
            self.frame_embed = positionalencoding1d(args.max_steps_per_subgoal+2, hidden_dim).transpose(1,0).to(device)

        num_queries += 1 # add an extra query for action prediction
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)

        if self.args.use_action_ghost_nodes:
            if not self.args.use_3d_pos_enc:
                assert(False) # not using 3d pos encodings here does not make sense
            print("Using action ghost nodes")
            self.num_movement_actions = sum(("Move" in k or "Rotate" in k or "Look" in k) for k in list(self.actions2idx.keys())) # all movement embeddings get the same node with different positional encodings
            
            self.query_embed = nn.Embedding(num_actions - self.num_movement_actions + 1, hidden_dim*2) # action prediction
            if self.args.learned_3d_pos_enc:
                self.action_distance_mapping = self.get_action_distance_mapping()

        elif self.args.query_per_action:
            print("Using query per action")
            self.query_embed = nn.Embedding(num_actions, hidden_dim*2) # action prediction

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
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

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        self.class_embed = _get_clones(self.class_embed, num_pred)
        self.intermediate_embed = _get_clones(self.intermediate_embed, num_pred)
        self.action_embed = _get_clones(self.action_embed, num_pred)
        self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        # hack implementation for iterative bounding box refinement
        self.transformer.decoder.bbox_embed = self.bbox_embed


        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])

        self.transformer.decoder.class_embed = self.class_embed
        for box_embed in self.bbox_embed:
            nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if args.do_self_atn_for_queries:
            self.transformer.multiscale_query_proj = self.multiscale_query_proj

    def forward(
        self, 
        samples, 
        boxes, 
        text, 
        positions=None,
        obj_centroids=None,
        depth=None,
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

        # get image and object features from the backbone
        srcs_t, masks_t, poss_t, poss_t_3d, obj_feats, obj_masks, obj_poss, \
            obj_poss_3d = self.get_visual_features(samples, boxes, obj_centroids, positions, depth)
        
        # let's concat the object features and learn a linear projection
        obj_feats = obj_feats.transpose(2,3).flatten(3,4)
        
        # project multiscale to correct size for query input
        obj_feats = self.multiscale_query_proj(obj_feats) # BxSxNxE

        # get query embeds
        query_embeds = self.query_embed.weight
        action_pos_enc = None
        if self.args.use_action_ghost_nodes:
            query_embeds, action_pos_enc = self.get_ghost_nodes_query_embed(query_embeds, positions)

        # encoder, decoder
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(
            srcs_t, masks_t, poss_t, query_embeds, 
            obj_srcs=obj_feats, obj_masks=obj_masks, obj_pos_embeds=obj_poss,
            text=text, B=B, S=S, action_pos_embeds=action_pos_enc, 
            pos_embeds_3d=poss_t_3d, obj_poss_embeds_3d=obj_poss_3d, 
        )

        if not self.args.do_decoder_2d3d: 
            # remove reference points for action queries
            if self.args.query_per_action:
                inter_references = inter_references[:,:,:-self.num_actions]
                init_reference = init_reference[:,:-self.num_actions]

        out = self.extract_outputs(
            hs, init_reference, inter_references,
            enc_outputs_coord_unact, enc_outputs_class)

        return out
    
    def get_ghost_nodes_query_embed(self, query_embeds, positions):
        # first is for positional encoding of the translation movements
        movement_pos = query_embeds[0:1].expand(self.num_movement_actions, query_embeds.shape[1])
        query_embeds = torch.cat([movement_pos, query_embeds[1:]])
        
        # pitch is not relative so we need to adjust pitch to be absolute
        lookdown = torch.clamp(positions[:,-1,-1] + self.action_distance_mapping["LookDown"][-1], min=min(self.pitch_range), max=max(self.pitch_range))
        lookup = torch.clamp(positions[:,-1,-1] + self.action_distance_mapping["LookUp"][-1], min=min(self.pitch_range), max=max(self.pitch_range))

        if self.args.learned_3d_pos_enc:
            action_pos = torch.stack(list(self.action_distance_mapping.values()), dim=0).unsqueeze(0).expand(self.B, len(self.action_distance_mapping), 5).clone()
            action_pos[:,list(self.action_distance_mapping.keys()).index("LookDown"),-1] = lookdown
            action_pos[:,list(self.action_distance_mapping.keys()).index("LookUp"),-1] = lookup

            pos_l_objs_3d = self.pos_enc_3d(action_pos.float())
            action_pos_enc = pos_l_objs_3d.transpose(1,2)
        return query_embeds, action_pos_enc

    def get_visual_features(
            self, samples, boxes, obj_centroids,
            positions, depth
        ):
        history_frame_inds = np.arange(samples.shape[1])
        frame_inds = history_frame_inds.copy()

        B, S, C, H, W = samples.shape
        M = boxes.shape[2]

        samples = samples.contiguous().view(B*S, C, H, W).unbind(0)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples) 
        num_feature_levs = len(features)

        srcs_t = []
        masks_t = []
        poss_t = []
        poss_t_3d = []

        # objects to track
        obj_feats = torch.ones(
            [B ,S, self.num_feature_levels, self.args.max_objs, self.hidden_dim]).to(self.device).to(features[-1].tensors.dtype)

        
        for l in range(self.num_feature_levels):

            #####%%%% Extract multiscale image features %%%####
            if l > num_feature_levs - 1:
                if l == num_feature_levs:
                    src = self.input_proj[l](features[-1].tensors)
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            else:
                feat = features[l]
                pos_l = pos[l]
                src, mask = feat.decompose()
                src = self.input_proj[l](src)
            _, C_l, H_l, W_l = src.shape

            ####%%% Extract object features %%%#####            
            _, C_pos_l, _, _ = pos_l.shape

            src = src.reshape(B,S,C_l,H_l,W_l)
            mask = mask.reshape(B,S,H_l,W_l)
            pos_l = pos_l.reshape(B,S,C_l,H_l,W_l)

            if self.args.roi_pool or self.args.roi_align:

                ########%%%%%%%% HISTORY %%%%%%%%############
                boxes = boxes.reshape(-1, M, 4)
                if l==0: 
                    '''
                    only need to compute pos enc & boxes for first layer since 
                    we pool object features from all layers for history
                    '''

                    # history pos encodings
                    pos_l_objs = self.get_history_positional_encoding(
                                        B,S,M,
                                        obj_centroids=obj_centroids,
                                        positions=positions,
                                        boxes=boxes,
                                        W_l=W_l, H_l=H_l, C_l=C_l,
                                        pos_l=pos_l,
                                        frame_inds=frame_inds
                                    )
                    
                    # transformer mask
                    obj_masks_ = ~check_zeros(boxes.reshape(-1,4)).view(B,S,M)
                    
                    if self.args.do_decoder_2d3d:
                        obj_poss_3d = pos_l_objs[1] # second positional enc is 3d
                        pos_l_objs = pos_l_objs[0] # first positional enc is 2d
                    else:
                        obj_poss_3d = None
                        
                    obj_poss = pos_l_objs
                    obj_masks = obj_masks_

                # ROI pool
                masks_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
                masks_xyxy_flat = masks_xyxy.float().reshape(B*S*M, 4)
                box_batch_inds = torch.arange(B*S, device=masks_xyxy.device).repeat(M,1).transpose(1,0).flatten().unsqueeze(1) # first column needs to be batch index
                masks_xyxy_flat = torch.cat([box_batch_inds, masks_xyxy_flat], dim=1)
                output_size = (int(H_l/4), int(W_l/4))
                if self.args.roi_pool:
                    feature_crop = roi_pool(src.reshape(B*S,C_l,H_l,W_l), masks_xyxy_flat, output_size=output_size, spatial_scale=W_l)
                elif self.args.roi_align:
                    feature_crop = roi_align(src.reshape(B*S,C_l,H_l,W_l), masks_xyxy_flat, output_size=output_size, spatial_scale=W_l)
                pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1).view(B,S,M,self.hidden_dim)

                obj_feats[:,:,l] = pooled_obj_feat


                ########%%%%%%%% CURRENT IMAGE %%%%%%%%############
                pos_l_im = self.get_img_positional_encoding(
                                    B,S,H,W,H_l,W_l,
                                    depth=depth,
                                    pos_l=pos_l,
                                    pitch_adjustment=positions[:,-1,-1] if positions is not None else None  # adjust xyz by current pitch (since pitch is in absolute terms)
                                    )
                
                if self.args.do_decoder_2d3d:
                    poss_t_3d.append(pos_l_im[1]) # second positional enc is 3d
                    pos_l_im = pos_l_im[0] # first positional enc is 2d
                srcs_t.append(src[:,-1]) # last image is current image
                masks_t.append(mask[:,-1]) # last image is current image
                poss_t.append(pos_l_im)

        return srcs_t, masks_t, poss_t, poss_t_3d, \
             obj_feats, obj_masks, obj_poss, obj_poss_3d

    def extract_outputs(
        self, hs, init_reference, inter_references, 
        enc_outputs_coord_unact, enc_outputs_class
        ):
        outputs_classes = []
        outputs_coords = []
        outputs_interms = []
        outputs_actions = []
        for lvl in range(hs.shape[0]):
            if self.args.do_deformable_atn_decoder:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
            if self.args.query_per_action:
                hs_ = hs[lvl][:,:-self.num_actions] # last num_actions are action queries
                as_ = hs[lvl][:,-self.num_actions:] # last num_actions are action queries
            outputs_class = self.class_embed[lvl](hs_)
            outputs_interm = self.intermediate_embed[lvl](hs_)
            outputs_action = self.action_embed[lvl](as_)
            tmp = self.bbox_embed[lvl](hs_)
            if self.args.do_deformable_atn_decoder:
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_interms.append(outputs_interm)
            outputs_actions.append(outputs_action)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_interms = torch.stack(outputs_interms)
        outputs_actions = torch.stack(outputs_actions)
        
        if self.args.query_per_action:
            outputs_actions = outputs_actions.transpose(3,2)

        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        out = {
            'pred_logits': outputs_class[-1], 
            'pred_boxes': outputs_coord[-1],
            'pred_interms': outputs_interms[-1],
            'pred_actions': outputs_actions[-1],
            }
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})
        if self.aux_loss:
            out_aux = {
                'pred_logits': outputs_class, 
                'pred_boxes': outputs_coord,
                'pred_interms': outputs_interms,
                'pred_actions': outputs_actions,
                }
            if self.with_vector:
                out_aux.update({'pred_vectors': outputs_vector})

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 
            'pred_boxes': enc_outputs_coord
            }
        return out

    def get_action_distance_mapping(self):
        return  {
            "MoveAhead": torch.tensor([0., 0., self.step_size, 0., 0.], device=device), 
            "RotateRight": torch.tensor([0., 0., 0., self.dt, 0.], device=device),
            "RotateLeft": torch.tensor([0., 0., 0., -self.dt, 0.], device=device),
            "LookDown": torch.tensor([0., 0., 0., 0., self.horizon_dt], device=device), # to replace in forward()
            "LookUp": torch.tensor([0., 0., 0., 0., -self.horizon_dt], device=device), # to replace in forward()
            "PickupObject": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "PutObject": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "OpenObject": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "CloseObject": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "SliceObject": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "ToggleObjectOn": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "ToggleObjectOff": torch.tensor([0., 0., 0., 0., 0.], device=device),
            "Done": torch.tensor([0., 0., 0., 0., 0.], device=device),
        }

    def get_img_positional_encoding(
        self,
        B,S,H,W,H_l,W_l,
        depth=None,
        pos_l=None,
        pitch_adjustment=None,
    ):  
        '''
        Get 2D and/or 3D positional encodings for the image features
        pitch adjustment: pitch in degrees to correct point cloud
        '''

        if self.args.do_decoder_2d3d:
            pos_l_img_3d = self.extract_img_posenc_2D3D(
                                B,S,H,W,H_l,W_l,
                                depth=depth,
                                pos_l=pos_l,
                                pitch_adjustment=pitch_adjustment,
                                mode='3D'
                                ) 
            pos_l_img_2d = self.extract_img_posenc_2D3D(
                                B,S,H,W,H_l,W_l,
                                depth=depth,
                                pos_l=pos_l,
                                pitch_adjustment=pitch_adjustment,
                                mode='2D'
                                )  
            pos_l_img = (pos_l_img_2d, pos_l_img_3d)

        return pos_l_img

    def extract_img_posenc_2D3D(
        self,
        B,S,H,W,H_l,W_l,
        depth=None,
        pos_l=None,
        pitch_adjustment=None,
        mode='2D'
    ):
        '''
        Extract 2D or 3D positional encodings for the image features
        mode: '2D' or '3D' to get positional 3D from position + centroids or 2D encodings
        '''

        if mode=='3D':
            '''
            Use 3D positional encodings for the current image by unprojecting the depth map 
            and interpolating to get 3d position for each feature point
            '''
            with torch.no_grad():
                xyz = utils.geom.depth2pointcloud(depth.unsqueeze(1), self.pix_T_camX.expand(B,4,4))
                if pitch_adjustment is not None:
                    '''
                    Adjust for pitch (head tilt) so that it is relative to 0 pitch
                    '''
                    # rx = torch.deg2rad(-pitch_adjustment) # pitch - positive is down, negative is up in aithor
                    rx = -pitch_adjustment # pitch - positive is down, negative is up in aithor
                    ry = torch.zeros(B).to(self.device) # yaw
                    rz = torch.zeros(B).to(self.device) # roll  
                    rot = utils.geom.eul2rotm(rx, ry, rz)
                    rotm = utils.geom.eye_4x4(B).to(self.device).to(xyz.dtype)
                    rotm[:,0:3,0:3] = rot
                    xyz = utils.geom.apply_4x4(rotm, xyz)
                    xyz[:,:,1] = -xyz[:,:,1]

                xyz = xyz.reshape(B, H, W, 3).permute(0,3,1,2)
                xyz = F.interpolate(xyz, mode='bilinear', size=(H_l,W_l), align_corners=False)

                if self.args.learned_3d_pos_enc:
                    xyz = xyz[:,[0,1,2],:,:].permute(0,2,3,1)
                    # append zeros since rotations are zero for all objs
                    pos_enc_input = torch.cat([xyz, torch.zeros(B,H_l,W_l,2).to(self.device)], dim=-1) 
                    pos_l_objs_3d = self.pos_enc_3d(pos_enc_input.flatten(1,2).float()).reshape(B,self.hidden_dim,H_l,W_l)
                    pos_l_im = pos_l_objs_3d
                else:
                    xyz = xyz[:,[0,1], :, :].permute(0,2,3,1) # throw away height info (not currently in positional encoding)
                    xyz = torch.round(xyz*(1/self.step_size))*self.step_size # round to step size of 3d pos encodings
                    rotation_ind_img = ((torch.zeros((B,H_l,W_l)) + np.pi*2)/self.dt, (torch.zeros((B,H_l,W_l)) + np.pi*2)/self.horizon_dt) # set to all zero rotation
                    rotation_inds_img = self.rot2dto1d_indices[rotation_ind_img[0].flatten().long(), rotation_ind_img[1].flatten().long()].reshape(B,H_l,W_l)
                    pos_l_objs_3d = self.pos_enc_3d[xyz[:,:,:,0].flatten().long(), xyz[:,:,:,1].flatten().long(), rotation_inds_img.flatten()].to(self.device).reshape(B,H_l,W_l,self.hidden_dim)
                    pos_l_objs_3d = pos_l_objs_3d.permute(0,3,1,2)
                    pos_l_im = pos_l_objs_3d
        elif mode=='2D':
            pos_l_im = pos_l[:,-1] # positional encoding only for current image
        else:
            assert(False) # wrong mode
        
        return pos_l_im

    def get_history_positional_encoding(
        self,
        B,
        S,
        M,
        obj_centroids=None,
        positions=None,
        boxes=None,
        W_l=None,
        H_l=None,
        C_l=None,
        pos_l=None,
        frame_inds=None
    ):
        '''
        Get 2D and/or 3D positional encodings for the history features
        '''
        if self.args.do_decoder_2d3d:
            # 3d history positional encoding
            pos_l_objs_3d = self.extract_history_posenc_2D3D(
                                    B,S,M,
                                    obj_centroids=obj_centroids,
                                    positions=positions,
                                    boxes=boxes,
                                    W_l=W_l,H_l=H_l,C_l=C_l,
                                    pos_l=pos_l,
                                    frame_inds=frame_inds,
                                    mode='3D'
                                )
            # 2d history positional encoding
            pos_l_objs_2d = self.extract_history_posenc_2D3D(
                                    B,S,M,
                                    obj_centroids=obj_centroids,
                                    positions=positions,
                                    boxes=boxes,
                                    W_l=W_l,H_l=H_l,C_l=C_l,
                                    pos_l=pos_l,
                                    frame_inds=frame_inds,
                                    mode='2D'
                                )
            pos_l_objs = (pos_l_objs_2d, pos_l_objs_3d)

        return pos_l_objs

    def extract_history_posenc_2D3D(
        self,
        B,
        S,
        M,
        obj_centroids=None,
        positions=None,
        boxes=None,
        C_l=None,
        W_l=None,
        H_l=None,
        pos_l=None,
        frame_inds=None,
        mode='2D'
    ):
        '''
        mode: '2D' or '3D' to get positional 3D from position + centroids or 2D encodings
        '''
        
        if mode=='3D':
            with torch.no_grad():
                if self.args.use_3d_obj_centroids:
                    if self.args.learned_3d_pos_enc:
                        # append zeros since rotations are zero for all objs
                        pos_enc_input = torch.cat([obj_centroids, torch.zeros(B,S,M,2).to(self.device)], dim=-1) 
                        if self.args.add_whole_image_mask:
                            # append agent position for whole image features
                            pos_enc_input[:,:,0,:] = positions
                        pos_l_objs_3d = self.pos_enc_3d(pos_enc_input.flatten(1,2).float()).transpose(1,2).reshape(B,S,M,self.hidden_dim)

                else:
                    if self.args.learned_3d_pos_enc:
                        pos_l_objs_3d = self.pos_enc_3d(positions.float()).transpose(1,2).reshape(B,S,1,self.hidden_dim).expand(B,S,M,self.hidden_dim)

                pos_l_objs = pos_l_objs_3d # add 3d positional embedding

        elif mode=='2D':
            # 2d object positional encoding
            masks_cx, masks_cy = boxes[:,:,0]*W_l, boxes[:,:,1]*W_l # mask is actually a box (cx, cy, lx, ly)
            masks_cx2, masks_cy2 = masks_cx.long().unsqueeze(-1).expand(B*S, M, self.hidden_dim).transpose(2,1), masks_cy.long().unsqueeze(-1).expand(B*S, M, self.hidden_dim).transpose(2,1)
            masks_x2 = masks_cx2*masks_cy2
            pos_l_objs = torch.gather(pos_l.reshape(B*S, C_l, H_l*W_l), 2, masks_x2).reshape(B,S, C_l, M).transpose(3,2)
            pos_l_objs_frame = self.frame_embed[frame_inds].reshape(1,S,1,self.hidden_dim).expand(B,S,M,self.hidden_dim)
            pos_l_objs += pos_l_objs_frame # add frame positional embedding
        else:
            assert(False) # wrong mode
        
        return pos_l_objs

    def extract_action_history_tokens(
        self,
        action_history,
        frame_inds,
        action_pad_frame=None
    ):  
        B,N = action_history.shape
        action_history = action_history.reshape(B*N)

        action_mask = action_history==-1

        invalid_actions = torch.where(action_mask) # history can be less than max history frames, these frames are tagged with -1
        valid_actions = torch.where(~action_mask) # history can be less than max history frames, these frames are tagged with -1
        
        action_tokens_ = self.action_token(action_history[valid_actions])
        action_tokens = torch.zeros([B*N,self.hidden_dim]).to(self.device).to(action_tokens_.dtype)
        action_tokens[valid_actions] = action_tokens_

        action_tokens = action_tokens.view(B,N,1,self.hidden_dim)
        action_pos = self.frame_embed[frame_inds].view(1, len(frame_inds), 1, self.hidden_dim).expand(B, len(frame_inds), 1, self.hidden_dim)
        action_mask = action_mask.view(B,N,1).to(torch.bool)

        if action_pad_frame is not None:
            action_tokens = torch.cat([action_tokens, torch.zeros(B, action_pad_frame, 1, self.hidden_dim).to(self.device)], dim=1)
            if not action_pos.shape[1]==N+action_pad_frame: # frame inds may already have this padding
                action_pos = torch.cat([action_pos, torch.zeros(B, action_pad_frame, 1, self.hidden_dim).to(self.device)], dim=1)
            action_mask = torch.cat([action_mask, torch.ones(B, action_pad_frame, 1).to(self.device).to(torch.bool)], dim=1) # padding should have attention mask

        return action_tokens, action_mask, action_pos

    def get_pix_T_camX(self):
        hfov = float(self.args.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.args.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.args.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        pix_T_camX[0,2] = self.args.W/2.
        pix_T_camX[1,2] = self.args.H/2.
        pix_T_camX = torch.from_numpy(pix_T_camX).to(device).unsqueeze(0).float()
        return pix_T_camX

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

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
    #             for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, 
                with_vector=False, 
                processor_dct=None, 
                vector_loss_coef=0.7, 
                no_vector_loss_norm=False,
                vector_start_stage=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.vector_loss_coef = vector_loss_coef
        self.no_vector_loss_norm = no_vector_loss_norm
        self.vector_start_stage = vector_start_stage

        print(f'Training with {6-self.vector_start_stage} vector stages.')

        print(f"Training with vector_loss_coef {self.vector_loss_coef}.")

        if not self.no_vector_loss_norm:
            print('Training with vector_loss_norm.')

        if args.use_action_weights: 
            self.action_weights = torch.tensor(args.action_weights).to(device)
        else:
            self.action_weights = None

            
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # also allows for no objects in view
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J)>0:
                target_classes_o.append(t["labels"][J])
        if len(target_classes_o)>0:
            target_classes_o = torch.cat(target_classes_o)
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_class_ce': loss_ce}

        if log:
            if len(target_classes_o)>0:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_intermediate(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        This is a binary loss
        """
        assert 'pred_interms' in outputs
        src_logits = outputs['pred_interms']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], 2,
                                    dtype=torch.int64, device=src_logits.device)
        
        # also allows for no objects in view
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J)>0:
                labels = t["labels"][J]
                interms = t["obj_targets"]
                if len(interms.shape)>1:
                    interms = interms.squeeze(0)
                target_classes_o_ = torch.zeros(len(labels), dtype=torch.long, device=src_logits.device) # 0 is not intermediate index
                for m in range(interms.shape[0]):
                    where_interm = torch.where(J.to(interms.device)==interms[m])[0]
                    target_classes_o_[where_interm] = 1 # 1 is intermediate index
                target_classes_o.append(target_classes_o_)
        if len(target_classes_o)>0:
            target_classes_o = torch.cat(target_classes_o)
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_interm_ce': loss_ce}

        if log:
            if len(target_classes_o)>0:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['intermediate_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_action(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_actions' in outputs
        src_logits = outputs['pred_actions'].squeeze(1)
        target_classes = torch.cat([t["expert_action"] for t in targets]) #.unsqueeze(1)
        loss_ce = F.cross_entropy(src_logits, target_classes, weight=self.action_weights) #, self.empty_weight)
        losses = {'loss_action_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['action_error'] = 100 - accuracy(src_logits, target_classes)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]

        # also allows for no objects in view
        target_boxes = []
        for t, (_, i) in zip(targets, indices):
            if len(i)>0:
                target_boxes.append(t['boxes'][i])

        if len(target_boxes) == 0:
            losses = {
                "loss_bbox": src_boxes.sum() * 0,
                "loss_giou": src_boxes.sum() * 0,
            }
            return losses

        target_boxes = torch.cat(target_boxes, dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_vectors" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_vectors"]
        src_boxes = outputs['pred_boxes']

        src_vectors = src_masks[src_idx]
        
        # also allows for no objects in view
        target_boxes = []
        target_masks_ = []
        valid = []
        for t, (_, i) in zip(targets, indices):
            if len(i)>0:
                target_boxes.append(box_ops.box_cxcywh_to_xyxy(t['boxes'][i]))
            target_masks_.append(t["masks"])

        if len(target_boxes) == 0:
            losses = {
                "loss_vector": src_vectors.sum() * 0
            }
            return losses

        target_boxes = torch.cat(target_boxes, dim=0) #.to(device=src_masks.device)
        target_masks, valid = nested_tensor_from_tensor_list(target_masks_).decompose()
        target_masks = target_masks.to(src_masks) #.to(device=src_masks.device)
        src_boxes = src_boxes[src_idx]
        target_masks = target_masks[tgt_idx]
        
        # scale boxes to mask dimesnions
        N, mask_w, mask_h = target_masks.shape
        target_sizes = torch.as_tensor([mask_w, mask_h]).unsqueeze(0).repeat(N, 1).to(device=src_masks.device)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        target_boxes = target_boxes * scale_fct


        # crop gt_masks
        n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
        gt_masks = BitMasks(target_masks)
        gt_masks = gt_masks.crop_and_resize(target_boxes, gt_mask_len).float()
        target_masks = gt_masks

        # perform dct transform
        target_vectors = []
        for i in range(target_masks.shape[0]):
            gt_mask_i = ((target_masks[i,:,:] >= 0.5)* 1).to(dtype=torch.uint8) 
            gt_mask_i = gt_mask_i.cpu().numpy().astype(np.float32)
            coeffs = cv2.dct(gt_mask_i)
            coeffs = torch.from_numpy(coeffs).flatten()
            coeffs = coeffs[torch.tensor(self.processor_dct.zigzag_table)]
            gt_label = coeffs.unsqueeze(0)
            target_vectors.append(gt_label)

        target_vectors = torch.cat(target_vectors, dim=0).to(device=src_vectors.device)
        losses = {}
        
        if self.no_vector_loss_norm:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='none').sum() / num_boxes
        else:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='mean')
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx.long(), src_idx.long()
        # return indices[0]

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx.long(), tgt_idx.long()
        # return indices[1]

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'intermediate': self.loss_intermediate,
            'action': self.loss_action,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' and i < self.vector_start_stage:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss in ['labels', 'intermediate', 'action']:
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'intermediate', 'action']:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss in ['labels', 'intermediate', 'action']:
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


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
        out_logits, out_bbox, out_vector, out_interms, out_actions = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_vectors'], outputs['pred_interms'], outputs['pred_actions'] #, outputs['batch_inds']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 50, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        prob_interms = out_interms.sigmoid()
        topk_boxes_interms = topk_boxes.unsqueeze(2).repeat(1,1,out_interms.shape[2])
        scores_interms = torch.gather(prob_interms, 1, topk_boxes_interms)
        labels_interms = torch.argmax(scores_interms, dim=2)
        scores_interms = torch.max(scores_interms, dim=2).values
        
        prob_actions = out_actions.squeeze(1).softmax(1)
        labels_action = torch.argmax(prob_actions, dim=1)

        if self.processor_dct is not None:
            n_keep = self.processor_dct.n_keep
            vectors = torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if self.processor_dct is not None and do_masks:
            masks = []
            n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
            b, r, c = vectors.shape
            for bi in range(b):
                outputs_masks_per_image = []
                for ri in range(r):
                    # here visual for training
                    idct = np.zeros((gt_mask_len ** 2))
                    idct[:n_keep] = vectors[bi,ri].cpu().numpy()
                    idct = self.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
                    re_mask = cv2.idct(idct)
                    max_v = np.max(re_mask)
                    min_v = np.min(re_mask)
                    re_mask = np.where(re_mask>(max_v+min_v) / 2., 1, 0)
                    re_mask = torch.from_numpy(re_mask)[None].float()
                    outputs_masks_per_image.append(re_mask)
                outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(out_vector.device)
                # here padding local mask to global mask
                outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
                    outputs_masks_per_image,  # N, 1, M, M
                    boxes[bi],
                    (img_h[bi], img_w[bi]),
                    threshold=0.5,
                )
                outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()
                masks.append(outputs_masks_per_image)
            masks = torch.stack(masks)

        if return_features and features is not None:
            features_keep = torch.gather(features, 1, topk_boxes.unsqueeze(-1).repeat(1,1,features.shape[-1]))


        if self.processor_dct is None or not do_masks:
            results1 = [{'scores': s, 'labels': l, 'boxes': b, 'scores_interm':i, 'labels_interm':e, 'labels_action':a} for s, l, b, i, e, a in zip(scores, labels, boxes, scores_interms, labels_interms, labels_action)]
        else:
            results1 = [{'scores': s, 'labels': l, 'boxes': b, 'masks': m, 'scores_interm':i, 'labels_interm':e, 'labels_action':a} for s, l, b, m, i, e, a in zip(scores, labels, boxes, masks, scores_interms, labels_interms, labels_action)]

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

    transformer = build_deforamble_transformer() if not args.checkpoint else build_cp_deforamble_transformer()
    if args.with_vector:
        processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)
    model = SOLQ(
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
        processor_dct=processor_dct if args.with_vector else None,
        vector_hidden_dim=args.vector_hidden_dim,
        actions2idx=actions2idx
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_class_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef,
        'loss_interm_ce': args.interm_loss_coef,
        'loss_action_ce': args.action_loss_coef,
        }
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_vector"] = 1
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        print("Doing mask loss!")
        losses += ["masks"]
    if args.do_intermediate_loss:
        print("doing intermediate object loss!")
        losses += ["intermediate"]
    if args.do_action_loss:
        print("doing action loss!")
        losses += ["action"]
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, 
                                                                        with_vector=args.with_vector, 
                                                                        processor_dct=processor_dct if args.with_vector else None,
                                                                        vector_loss_coef=args.vector_loss_coef,
                                                                        no_vector_loss_norm=args.no_vector_loss_norm,
                                                                        vector_start_stage=args.vector_start_stage)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector) else None)}

    if args.masks:
        postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
