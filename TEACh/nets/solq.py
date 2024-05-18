import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from torchvision import models
import ipdb
st = ipdb.set_trace
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import SOLQ.util.misc as ddetr_utils
from SOLQ.util import box_ops
from SOLQ.models import build_model
import random
from dataclasses import dataclass

@dataclass
class ARGS_SOLQ:
    # default params
    masks = True
    with_box_refine = True
    meta_arch = 'solq'
    with_vector = True
    batch_size = 4
    vector_hidden_dim = 1024
    vector_loss_coef = 3

    position_embedding = 'sine'
    with_box_refine = True
    two_stage = True
    n_keep = 256
    gt_mask_len = 128
    # vector_loss_coef = 0.7
    # vector_hidden_dim = 1024 #256
    activation = 'relu'
    checkpoint = False
    vector_start_stage = 0
    dcn = False
    frozen_weights=None
    pretrained = None
    backbone = 'resnet50'
    position_embedding_scale = 2 * np.pi
    num_feature_levels = 4
    enc_layers = 6
    dec_layers = 6
    dim_feedforward = 1024
    hidden_dim = 384
    dropout = 0.1
    nheads = 8
    num_queries = 300
    dec_n_points = 4
    enc_n_points = 4
    set_cost_class= 2
    set_cost_bbox = 5
    set_cost_giou = 2

    seed = 42

    lr = 2e-4
    lr_backbone_names = ["backbone.0"]
    lr_backbone = 2e-5
    lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    lr_linear_proj_mult = 0.1
    # batch_size = 2
    weight_decay = 1e-4
    epochs = 50
    lr_drop = 40
    device = 'cuda'
    save_period = 10
    # meta_arch = 'solq'
    loss_type = 'l1'

    mask_loss_coef = 1
    dice_loss_coef = 1
    cls_loss_coef = 2
    bbox_loss_coef = 5
    giou_loss_coef = 2
    focal_alpha = 0.25
    dilation = False

    aux_loss = True

    dataset_file = 'coco'
    coco_path = './data/coco'
    coco_panoptic_path = ''
    remove_difficult = False
    output_dir = ''
    device = 'cuda'
    seed = 42
    resume = ''
    start_epoch = 0
    eval = False
    num_workers = 2
    cache_mode = False
    transformer_mode = None
    use_egomotion = True

    no_vector_loss_norm = False

from arguments import args
# args = ARGS_SOLQ()

# fix the seed for reproducibility
seed = args.seed # + ddetr_utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class DDETR(nn.Module):
	# def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, num_classes, load_pretrained=False, num_classes2=None):
        super(DDETR, self).__init__()

        model, criterion, postprocessors = build_model(args, num_classes, num_classes2)

        if load_pretrained:
            # load pretrained model
            print('loading BASE pretrained ddetr...')
            PATH = '/projects/katefgroup/viewpredseg/checkpoints/ddetr_pretrained/solq_r50_final.pth'
            checkpoint = torch.load(PATH)
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # learn from scratch embeddings with class size info
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "class_embed" not in k}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "query_embed" not in k}
            model_dict.update(pretrained_dict) 
            msg = model.load_state_dict(pretrained_dict, strict=False)
            print(f'LOADED WITH MESSAGE: {msg}')

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.nms_threshold = 0.4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, targets=None, do_loss=True, return_features=False):

        outputs = self.model(rgb, return_features=return_features)

        if do_loss:
            loss_dict = self.criterion(outputs, targets)

            weight_dict = self.criterion.weight_dict

            losses = []
            for k in loss_dict.keys():
                if k in weight_dict:
                    losses.append(loss_dict[k] * weight_dict[k])
                    # print(k)
                elif ("class_error" in k) or ("cardinality_error" in k):
                    pass
                else:
                    print("LOSS IS ", k)
                    assert(False) # not in weight dict
            losses = sum(losses)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = ddetr_utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()
        else:
            losses = None
            loss_dict_reduced = None
            loss_dict_reduced_unscaled = None
            loss_dict_reduced_scaled = None
            losses_reduced_scaled = None

        out_dict = {}
        out_dict['outputs'] = outputs
        out_dict['losses'] = losses
        out_dict['loss_dict_reduced'] = loss_dict_reduced
        out_dict['loss_dict_reduced_unscaled'] = loss_dict_reduced_unscaled
        out_dict['loss_dict_reduced_scaled'] = loss_dict_reduced_scaled
        out_dict['postprocessors'] = self.postprocessors

        return out_dict