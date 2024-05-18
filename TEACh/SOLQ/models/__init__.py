# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from arguments import args
from .solq_base import build as build_solq
# from .solq_stp import build as build_solq_stp

def build_model(
    args, 
    num_classes, 
    num_classes2=None,
    actions2idx=None,
    ):
    # if 'stp' in args.DDETR_mode:
    #     return build_solq_stp(num_classes, num_classes2, actions2idx)
    # else:
    return build_solq(num_classes, num_classes2)

