from ai2thor.controller import Controller
import os
import ipdb
st = ipdb.set_trace
import numpy as np
import random
import cv2
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import sys
# sys.path.append('./')
import numpy as np
import torch
import time

from utils.improc import *
import torch.nn.functional as F 

from arguments import args
import wandb

# sys.path.append('./SOLQ')
# from nets.solq import DDETR
# import SOLQ.util.misc as ddetr_utils
# from SOLQ.util import box_ops
from backend import saverloader
# import argparse
# from utils.parser import get_args_parser

# sys.path.append('./Object-Detection-Metrics')
# from Detection_Metrics.pascalvoc_nofiles import get_map, ValidateFormats, ValidateCoordinatesTypes, add_bounding_box
# import glob
# from Detection_Metrics.lib.BoundingBox import BoundingBox
# from Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
# from Detection_Metrics.lib.Evaluator import *
# from Detection_Metrics.lib.utils_pascal import BBFormat

import torch.nn as nn
from tqdm import tqdm
from models.teach_train_depth_base import Ai2Thor_Base
import pickle
import random

from nets.depthnet import DepthNet, ZoeDepthLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# fix the seed for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class Ai2Thor(Ai2Thor_Base):
    def __init__(self):   

        super(Ai2Thor, self).__init__()

        # initialize wandb
        if args.set_name=="test00":
            wandb.init(mode="disabled")
        else:
            wandb.init(project="embodied-llm-teach", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

        self.model = DepthNet()
        self.model.to(device)
        self.model.train()

        self.loss = ZoeDepthLoss(self.model.config)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        # lr set by arg_parser
        # params_to_optimize = self.model.parameters()
        params_to_optimize = self.model.model.get_lr_params(args.lr)
        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_drop = args.lr_drop # every X epochs, drop lr by 0.1
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_drop)

        # for name, param in self.model.named_parameters():
        #     print(name, "requires grad?", param.requires_grad)

        self.start_step = 1
        if args.load_model:
            path = args.load_model_path 

            if args.lr_scheduler_from_scratch:
                print("LR SCHEDULER FROM SCRATCH")
                lr_scheduler_load = None
            else:
                lr_scheduler_load = self.lr_scheduler

            if args.optimizer_from_scratch:
                print("OPTIMIZER FROM SCRATCH")
                optimizer_load = None
            else:
                optimizer_load = self.optimizer
            
            self.start_step = saverloader.load_from_path(path, self.model, optimizer_load, strict=(not args.load_strict_false), lr_scheduler=lr_scheduler_load)

        if args.start_one:
            self.start_step = 1

        self.max_iters = args.max_iters
        self.log_freq = args.log_freq

    def run_episodes(self):
        
        self.ep_idx = 0

        print(f"Iterations go from {self.start_step} to {self.max_iters}")
        
        for iter_ in range(self.start_step, self.max_iters):

            print("Begin iter", iter_)
            print("set name:", args.set_name)
            self.iter = iter_

            if iter_ % self.log_freq == 0:
                self.log_iter = True
            else:
                self.log_iter = False

            # self.summ_writer = utils.improc.Summ_writer(
            #     writer=self.writer,
            #     global_step=iter_,
            #     log_freq=self.log_freq,
            #     fps=8,
            #     just_gif=True)

            # if args.save_output:
            #     print("RUNNING SAVE OUTPUT")
            #     self.save_output()
            #     return 
            
            total_loss = self.run_train()

            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                if args.clip_max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                else:
                    grad_total_norm = ddetr_utils.get_total_grad_norm(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()

            print(f"loss for iter {iter_} is: {total_loss}")

            if args.run_val:
                if iter_ % args.val_freq == 0:
                    with torch.no_grad():
                        self.run_val()

            if self.lr_scheduler is not None:
                if iter_ % args.lr_scheduler_freq== 0:
                    self.lr_scheduler.step()

            if iter_ % args.save_freq == 0:
                saverloader.save_checkpoint(
                        self.model, 
                        self.checkpoint_path, 
                        iter_, 
                        iter_, 
                        self.optimizer, 
                        keep_latest=args.keep_latest, 
                        lr_scheduler=self.lr_scheduler
                        )

        self.controller.stop()
        time.sleep(1)

    def run_train(self):
        total_loss = torch.tensor(0.0).to(device)

        if self.random_select: # select random house set 
            rand = np.random.randint(len(self.mapnames_train))
            mapname = self.mapnames_train[rand]
        else:
            assert(False)

        print("MAPNAME=", mapname)

        load_dict = None

        if args.load_train_agent:
            n = np.random.randint(args.n_train)
            n_messup = None
            outputs = self.load_agent_output(n, mapname, args.train_load_dir, save=True)
        else:
            n_messup = None
            outputs = self.load_agent_output(None, mapname, None, run_agent=True)
        
        if outputs[0] is None:
            return None

        rgb = outputs[0]
        depth_gt = outputs[1]

        mask = torch.logical_and(depth_gt >= self.model.config.min_depth, depth_gt <= self.model.config.max_depth).squeeze().unsqueeze(1) # depth range 0-10

        depth_pred = self.model(rgb)

        loss, losses_dict, pred_interp = self.loss(depth_pred, depth_gt, mask=mask)
        total_loss += loss

        wandb.log({"train/total_loss": total_loss.cpu().item(), 'batch': self.iter})
        wandb.log({"train/silog_loss": losses_dict['silog_loss'].cpu().item(), 'batch': self.iter})
        wandb.log({"train/l_grad": losses_dict['l_grad'].cpu().item(), 'batch': self.iter})
        if self.log_iter:
            wandb.log({"train/image": wandb.Image(rgb[0]), 'batch': self.iter})
            wandb.log({"train/pred_depth": wandb.Image(pred_interp[0,0]), 'batch': self.iter})
            wandb.log({"train/gt_depth": wandb.Image(depth_gt[0]), 'batch': self.iter})
                
        return total_loss

    @torch.no_grad()
    def run_val(self):

        print("VAL MODE")
        self.model.eval()

        total_loss = torch.tensor(0.0).to(device)
        total_loss_silog = torch.tensor(0.0).to(device)
        total_loss_l_grad = torch.tensor(0.0).to(device)

        num_total_iters = len(self.mapnames_val)*args.n_val
        for n in range(args.n_val):
            for episode in range(len(self.mapnames_val)):
                print("STARTING EPISODE ", episode, "iteration", n)

                mapname = self.mapnames_val[episode]
                print("MAPNAME=", mapname)

                if args.load_val_agent:
                    outputs = self.load_agent_output(n, mapname, args.val_load_dir, save=True)
                else:
                    outputs = self.load_agent_output(None, mapname, None, run_agent=True)
                
                if outputs[0] is None:
                    continue
                    # return None

                rgb = outputs[0]
                depth_gt = outputs[1]

                mask = torch.logical_and(depth_gt >= self.model.config.min_depth, depth_gt <= self.model.config.max_depth).squeeze().unsqueeze(1) # depth range 0-10

                # try:
                depth_pred = self.model(rgb)
                # except Exception as e:
                #     print(e)
                #     continue

                loss, losses_dict, pred_interp = self.loss(depth_pred, depth_gt, mask=mask)
                total_loss += loss
                total_loss_silog += losses_dict['silog_loss']
                total_loss_l_grad += losses_dict['l_grad']

                if n==0:
                    wandb.log({"val/image": wandb.Image(rgb[0]), 'batch': self.iter})
                    wandb.log({"val/pred_depth": wandb.Image(pred_interp[0,0]), 'batch': self.iter})
                    wandb.log({"val/gt_depth": wandb.Image(depth_gt[0]), 'batch': self.iter})

        wandb.log({"val/total_loss": total_loss.cpu().item() / num_total_iters, 'batch': self.iter})
        wandb.log({"val/silog_loss": total_loss_silog.cpu().item() / num_total_iters, 'batch': self.iter})
        wandb.log({"val/l_grad": total_loss_l_grad.cpu().item() / num_total_iters, 'batch': self.iter})
        
        self.model.train()

    def load_agent_output(self, n, mapname, load_dir, pick_rand_n=False, always_load_all_samples=False, save=False, run_agent=False, override_existing=False):
        
        if not run_agent:
            root = os.path.join(load_dir, mapname) 
            if not os.path.exists(root):
                os.mkdir(root)
            
            if pick_rand_n or n is None:
                ns = os.listdir(root)
                ns = np.array([int(n_[0]) for n_ in ns])
                n = np.random.choice(ns)
        
            print(f"Chose n to be {n}")

            pickle_fname = f'{n}.p'
            fname_ = os.path.join(root, pickle_fname)
        else:
            fname_ = ""

        if not os.path.isfile(fname_) or override_existing or run_agent: # if doesn't exist generate it
            if override_existing:
                print("WARNING: OVERRIDING EXISTING FILE IF THERE IS ONE")
            else:
                print("file doesn not exist...generating it")
        
            self.controller.reset(scene=mapname)

            load_dict = None

            outputs = self.run_agent(load_dict=load_dict)

            if save:
                print("saving", fname_)
                with open(fname_, 'wb') as f:
                    pickle.dump(outputs, f, protocol=4)
                print("done.")
        
        else:
            print("-----------------")
            print("LOADING OUTPUTS")
            print("-----------------")
            with open(fname_, 'rb') as f:
                outputs = pickle.load(f)        

        rgb = outputs[0]
        depth = outputs[1]
        if len(outputs[0])<args.batch_size:
            print("Warning: requested batch size larger than saved batch size. Returning max batch size of data.")
        if len(outputs[0])>args.batch_size and not always_load_all_samples:
            # sample
            N = np.arange(len(rgb))
            idxs = np.random.choice(N, size=args.batch_size, replace=False)
            rgb = rgb[idxs]
            depth = depth[idxs]

        outputs[0] = outputs[0].to(device)
        outputs[1] = outputs[1].to(device)
        
        return outputs


    
    
    def save_output(self):

        for n in range(args.n_train):
            for episode in range(len(self.mapnames_train)):
                mapname = self.mapnames_train[episode]

                print("MAPNAME=", mapname, "n=", n)

                outputs = self.load_agent_output(n, mapname, args.multiview_load_dir)

                print("DONE.", "MAPNAME=", mapname, "n=", n)

        for n in range(args.n_val):
            for episode in range(len(self.mapnames_val)):
                mapname = self.mapnames_val[episode]

                print("MAPNAME=", mapname, "n=", n)

                outputs = self.load_agent_output(n, mapname, args.multiview_load_dir)

                print("DONE.", "MAPNAME=", mapname, "n=", n)

        for n in range(args.n_test):
            for episode in range(len(self.mapnames_test)):
                mapname = self.mapnames_test[episode]

                print("MAPNAME=", mapname, "n=", n)

                outputs = self.load_agent_output(n, mapname, args.multiview_load_dir)

                print("DONE.", "MAPNAME=", mapname, "n=", n)


if __name__ == '__main__':
    Ai2Thor()