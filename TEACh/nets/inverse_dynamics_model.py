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
from .ID_Transformer import build_model
import random
from dataclasses import dataclass
from arguments import args

# fix the seed for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class IDM(nn.Module):
    def __init__(
        self, 
        num_classes, 
        load_pretrained=False, 
        num_actions=None,
        actions2idx=None,
        action_weights=None
        ):
        super(IDM, self).__init__()

        model, criterion, postprocessors = build_model(
            args, 
            num_classes, 
            num_actions,
            actions2idx,
            )

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.nms_threshold = 0.4
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, 
        images, 
        targets=None, 
        do_loss=True
        ):

        outputs = self.model(
            images, 
            )

        if do_loss:
            loss_dict = self.criterion(outputs, targets)

            weight_dict = self.criterion.weight_dict

            losses = []
            for k in loss_dict.keys():
                if k in weight_dict:
                    losses.append(loss_dict[k] * weight_dict[k])
                elif "error" in k:
                    # error is not loss
                    pass
                else:
                    print("LOSS IS ", k)
                    assert(False) # not in weight dict
            losses = sum(losses)

            if torch.isnan(losses):
                losses = None
                print("Loss is NaN!")
                st()
                assert(False) # loss is NaN

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
            loss_value = None

        out_dict = {}
        out_dict['outputs'] = outputs
        out_dict['losses'] = losses
        out_dict['loss_dict_reduced'] = loss_dict_reduced
        out_dict['loss_value'] = loss_value
        out_dict['loss_dict_reduced_unscaled'] = loss_dict_reduced_unscaled
        out_dict['loss_dict_reduced_scaled'] = loss_dict_reduced_scaled
        out_dict['postprocessors'] = self.postprocessors

        return out_dict

    def predict(self, images):
        outputs = self.model(
            images,
            )
        pred_action = int(torch.argmax(self.softmax(outputs['pred_actions'].squeeze(1))).cpu().numpy())
        pred_label = int(torch.argmax(self.softmax(outputs['pred_logits'].squeeze(1))).cpu().numpy())
        pred_label_forceobject = int(torch.argmax(self.softmax(outputs['pred_logits'].squeeze(1)[:,:-1])).cpu().numpy())
        outputs = {"pred_action": pred_action, "pred_label":pred_label, "pred_label_forceobject":pred_label_forceobject}
        return outputs