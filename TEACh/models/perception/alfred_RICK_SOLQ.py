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
from nets.solq import DDETR
import SOLQ.util.misc as ddetr_utils
from SOLQ.util import box_ops
from backend import saverloader
# import argparse
# from utils.parser import get_args_parser

# sys.path.append('./Object-Detection-Metrics')
from Detection_Metrics.pascalvoc_nofiles import get_map, ValidateFormats, ValidateCoordinatesTypes, add_bounding_box
import glob
from Detection_Metrics.lib.BoundingBox import BoundingBox
from Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
from Detection_Metrics.lib.Evaluator import *
from Detection_Metrics.lib.utils_pascal import BBFormat

import torch.nn as nn
from tqdm import tqdm
from models.alfred_RICK_SOLQ_base import Ai2Thor_Base
import pickle

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

        num_classes = len(self.include_classes) 
        num_classes2 = None
        self.score_boxes_name = 'pred1' # only one prediction head so same for both
        self.score_labels_name = 'pred1'

        load_pretrained = args.load_base_solq
        self.model = DDETR(num_classes, load_pretrained, num_classes2=num_classes2)
        self.model.cuda()

        

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        param_dicts = [
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                # "lr": args.lr_backbone,
                "lr": args.lr * args.lr_backbone_mult,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            }
        ]

        # lr set by arg_parser
        params_to_optimize = self.model.parameters()
        self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_drop = args.lr_drop # every X epochs, drop lr by 0.1
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_drop)

        for name, param in self.model.named_parameters():
            print(name, "requires grad?", param.requires_grad)

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
        # self.run_episodes()

        # initialize wandb
        if args.set_name=="test00":
            wandb.init(mode="disabled")
        else:
            wandb.init(project="embodied_ai", name=args.set_name)

    def run_episodes(self):
        
        self.ep_idx = 0

        print(f"Iterations go from {self.start_step} to {self.max_iters}")
        
        for iter_ in range(self.start_step, self.max_iters):

            print("Begin iter", iter_)
            print("set name:", args.set_name)

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

            if args.save_output:
                print("RUNNING SAVE OUTPUT")
                self.save_output()
                return 
            
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
        total_loss = torch.tensor(0.0).cuda() #to(self.device0) #.cuda()

        # if self.log_iter:
        #     mapper = mAP(score_boxes_name=self.score_boxes_name, score_labels_name=self.score_labels_name)
        #     if args.two_pred_heads:
        #         mapper2 = mAP(score_boxes_name=self.score_boxes_name, score_labels_name=self.score_labels_name2)

        if self.random_select: # select random house set 
            rand = np.random.randint(len(self.mapnames_train))
            mapname = self.mapnames_train[rand]
        else:
            assert(False)

        print("MAPNAME=", mapname)

        load_dict = None

        if args.load_train_agent:
            n = np.random.randint(args.n_train)
            # if args.mess_up_from_loaded:
            #     n_messup = np.random.randint(args.n_train_messup)
            # else:
            n_messup = None
            outputs = self.load_agent_output(n, mapname, args.train_load_dir, n_messup=n_messup, save=True)
        else:
            # n = np.random.randint(args.max_load_n)
            # n_messup = np.random.randint(args.n_train_messup)
            n_messup = None
            outputs = self.load_agent_output(None, mapname, None, n_messup=n_messup, run_agent=True)
        
        # time.sleep(1)

        if outputs[0] is None:
            return None

        out_dict = self.model(*outputs)
        out = out_dict['outputs']
        loss = out_dict['losses']
        loss_dict_reduced = out_dict['loss_dict_reduced']
        loss_dict_reduced_unscaled = out_dict['loss_dict_reduced_unscaled'] 
        loss_dict_reduced_scaled = out_dict['loss_dict_reduced_scaled'] 
        postprocessors = out_dict['postprocessors'] 

        total_loss += loss

        wandb.log({"train/total_loss": total_loss.cpu().item()})
        wandb.log({"train/loss_giou": loss_dict_reduced['loss_giou'].cpu().item()})
        wandb.log({"train/ce_loss": loss_dict_reduced['loss_ce'].cpu().item()})
        wandb.log({"train/class_accuracy": 100 - loss_dict_reduced['class_error'].cpu().item()})
        wandb.log({"train/class_error": loss_dict_reduced['class_error'].cpu().item()})
        loss_dict_reduced_unscaled = {f"train/{k}": v for k, v in loss_dict_reduced_unscaled.items()}
        loss_dict_reduced_scaled = {f"train/{k}": v for k, v in loss_dict_reduced_scaled.items()}
        wandb.log(loss_dict_reduced_unscaled)
        wandb.log(loss_dict_reduced_scaled)

        # for key in list(loss_dict_reduced_unscaled.keys()):
        #     self.summ_writer.summ_scalar(f'train_unscaled/{key}', loss_dict_reduced_unscaled[key])
        # for key in list(loss_dict_reduced_scaled.keys()):
        #     self.summ_writer.summ_scalar(f'train_scaled/{key}', loss_dict_reduced_scaled[key])

        if self.log_iter:
            rgb_x = [outputs[0][i] for i in range(len(outputs[0]))]
            targets = [outputs[1][i] for i in range(len(outputs[0]))]
            img_torch = self.draw_ddetr_predictions(rgb_x, out, targets, 
                                                    self.id_to_name, postprocessors=postprocessors, 
                                                    pred_label_name='pred1', gt_label_name='labels', 
                                                    score_threshold=args.score_threshold, text_size=1, plot_masks=args.plot_masks, 
                                                    )
            rgb_x_vis = img_torch.unsqueeze(0) #- 0.5
            name = f'train/prediction'
            wandb.log({name: wandb.Image(rgb_x_vis)})
            # self.summ_writer.summ_rgb(name, rgb_x_vis)
            # if args.do_predict_oop:
            #     img_torch = self.draw_ddetr_predictions(rgb_x, out, targets, 
            #                         self.id_to_label, postprocessors=postprocessors, 
            #                         pred_label_name='pred2', gt_label_name='labels2', 
            #                         score_threshold=args.score_threshold, text_size=1, plot_masks=args.plot_masks, 
            #                         )
            #     rgb_x_vis = img_torch.unsqueeze(0) #- 0.5
            #     name = f'train_OOP/prediction'
            #     wandb.log({name: wandb.Image(rgb_x_vis)})
                # self.summ_writer.summ_rgb(name, rgb_x_vis)

        # if self.log_iter:
        #     targets = outputs[1]
        #     # targets = dict()
        #     # for key in list(targets.keys()):
        #     #     if key not in targets:
        #     #         targets[key] = torch.tensor([]).cuda()
        #     #     if key=='batch_lens':
        #     #         targets_[key] += torch.max(targets[key]) + 1
        #     #     targets[key] = torch.cat([targets[key], targets_[key]], dim=0)
        #     mapper.add_boxes(out, postprocessors, self.W, self.H, self.id_to_name, targets, predictions_label_name='labels', do_nms=True)
        #     map_stat = mapper.get_stats()
        #     print("mAP:", map_stat)
        #     self.summ_writer.summ_scalar('train/mAP', map_stat)
        #     if args.two_pred_heads:
        #         mapper2.add_boxes(out, postprocessors, self.W, self.H, self.id_to_label, targets, predictions_label_name='labels2', do_nms=True)
        #         map_stat2 = mapper2.get_stats()
        #         print("mAP2:", map_stat2)
        #         self.summ_writer.summ_scalar('train/mAP2', map_stat2)
                
        return total_loss

    def load_agent_output(self, n, mapname, load_dir, n_messup, pick_rand_n=False, always_load_all_samples=False, save=False, run_agent=False, override_existing=False):
        
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

            # load_dict = None

            # if args.mess_up_from_loaded:
            #     messup_fname = os.path.join(args.mess_up_dir, mapname, f'{n_messup}.p')
            #     with open(messup_fname, 'rb') as f:
            #         load_dict = pickle.load(f)
            # else:
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
        targets = outputs[1]
        if len(outputs[0])<args.batch_size:
            print("Warning: requested batch size larger than saved batch size. Returning max batch size of data.")
        if len(outputs[0])>args.batch_size and not always_load_all_samples:
            # sample
            N = np.arange(len(rgb))
            idxs = np.random.choice(N, size=args.batch_size, replace=False)
            rgb = rgb[idxs]
            targets = [targets[i] for i in list(idxs)]

        # put on cuda
        for t in targets:
            for k in t.keys():
                t[k] = t[k].cuda()
        rgb = rgb.cuda()

        outputs[0] = rgb
        outputs[1] = targets
        
        return outputs


    def run_val(self):

        print("VAL MODE")
        self.model.eval()

        total_loss = torch.tensor(0.0).cuda()
        total_loss_giou = torch.tensor(0.0).cuda()
        total_loss_ce = torch.tensor(0.0).cuda()
        total_class_error = torch.tensor(0.0).cuda()
        total_class_accuracy = torch.tensor(0.0).cuda()
        # if args.do_predict_oop:
        #     total_loss_ce2 = torch.tensor(0.0).cuda()
        #     total_class_error2 = torch.tensor(0.0).cuda()
        #     total_class_accuracy2 = torch.tensor(0.0).cuda()
        
        # mapper = mAP(score_boxes_name=self.score_boxes_name, score_labels_name=self.score_labels_name)
        # if args.two_pred_heads:
        #     mapper2 = mAP(score_boxes_name=self.score_boxes_name, score_labels_name=self.score_labels_name2)

        num_total_iters = len(self.mapnames_val)*args.n_val
        for n in range(args.n_val):
            for episode in range(len(self.mapnames_val)):
                print("STARTING EPISODE ", episode, "iteration", n)

                mapname = self.mapnames_val[episode]
                print("MAPNAME=", mapname)

                if args.load_val_agent:
                    outputs = self.load_agent_output(n, mapname, args.val_load_dir, n_messup=n, save=True)
                else:
                    outputs = self.load_agent_output(None, mapname, None, n_messup=n, run_agent=True)
                
                # time.sleep(1)

                if outputs[0] is None:
                    return None
                out_dict = self.model(*outputs)
                out = out_dict['outputs']
                loss = out_dict['losses']
                loss_dict_reduced = out_dict['loss_dict_reduced']
                loss_dict_reduced_unscaled = out_dict['loss_dict_reduced_unscaled'] 
                loss_dict_reduced_scaled = out_dict['loss_dict_reduced_scaled'] 
                postprocessors = out_dict['postprocessors'] 
                total_loss += loss / num_total_iters
                total_loss_giou += loss_dict_reduced['loss_giou'] / num_total_iters
                total_loss_ce += loss_dict_reduced['loss_ce'] / num_total_iters
                total_class_error += loss_dict_reduced['class_error'] / num_total_iters
                total_class_accuracy += (100 - loss_dict_reduced['class_error']) / num_total_iters
                # if args.do_predict_oop:
                #     total_class_error2 += loss_dict_reduced['class_error2'] / num_total_iters
                #     total_class_accuracy2 += (100 - loss_dict_reduced['class_error2']) / num_total_iters
                #     total_loss_ce2 += loss_dict_reduced['loss_ce2']

                if n==0 and episode==0:
                    loss_dict_unscaled = dict()
                    for key in list(loss_dict_reduced_unscaled.keys()):
                        loss_dict_unscaled[key] = loss_dict_reduced_unscaled[key]
                    loss_dict_scaled = dict()
                    for key in list(loss_dict_reduced_scaled.keys()):
                        loss_dict_scaled[key] = loss_dict_reduced_scaled[key]
                else:
                    for key in list(loss_dict_reduced_unscaled.keys()):
                        loss_dict_unscaled[key] += loss_dict_reduced_unscaled[key]
                    for key in list(loss_dict_reduced_scaled.keys()):
                        loss_dict_scaled[key] += loss_dict_reduced_scaled[key]

                if args.plot_boxes and n==0: # only plot first n to save time
                    rgb_x = [outputs[0][i] for i in range(len(outputs[0]))]
                    targets = [outputs[1][i] for i in range(len(outputs[0]))]
                    img_torch = self.draw_ddetr_predictions(rgb_x, out, targets, 
                                                            self.id_to_name, postprocessors=postprocessors, 
                                                            pred_label_name='pred1', gt_label_name='labels', 
                                                            score_threshold=args.score_threshold, text_size=1, plot_masks=args.plot_masks, 
                                                            )
                    rgb_x_vis = img_torch.unsqueeze(0) - 0.5
                    name = f'val/prediction'
                    # self.summ_writer.summ_rgb(name, rgb_x_vis)
                    wandb.log({name: wandb.Image(rgb_x_vis)})
                    # if args.do_predict_oop:
                    #     img_torch = self.draw_ddetr_predictions(rgb_x, out, targets, 
                    #                         self.id_to_label, postprocessors=postprocessors, 
                    #                         pred_label_name='pred2', gt_label_name='labels2', 
                    #                         score_threshold=args.score_threshold, text_size=1, plot_masks=args.plot_masks, 
                    #                         )
                    #     rgb_x_vis = img_torch.unsqueeze(0) - 0.5
                    #     name = f'val_OOP/prediction'
                    #     self.summ_writer.summ_rgb(name, rgb_x_vis)

        wandb.log({"val/total_loss": total_loss.cpu().item()})
        wandb.log({"val/loss_giou": total_loss_giou.cpu().item()})
        wandb.log({"val/ce_loss": total_loss_ce.cpu().item()})
        wandb.log({"val/class_accuracy": total_class_accuracy.cpu().item()})
        wandb.log({"val/class_error": total_class_error.cpu().item()})
        loss_dict_unscaled = {f"val/{k}": v / num_total_iters for k, v in loss_dict_unscaled.items()}
        loss_dict_scaled = {f"val/{k}": v / num_total_iters for k, v in loss_dict_scaled.items()}
        wandb.log(loss_dict_unscaled)
        wandb.log(loss_dict_scaled)

        # self.summ_writer.summ_scalar('val/total_loss', total_loss.cpu().item())
        # self.summ_writer.summ_scalar('val/loss_giou', total_loss_giou.cpu().item())
        # self.summ_writer.summ_scalar('val/ce_loss', total_loss_ce.cpu().item())
        # self.summ_writer.summ_scalar('val/class_error', total_class_error.cpu().item())
        # self.summ_writer.summ_scalar('val/class_accuracy', total_class_accuracy.cpu().item())
        # if args.do_predict_oop:
        #     self.summ_writer.summ_scalar('val/ce_loss2', total_loss_ce2.cpu().item())
        #     self.summ_writer.summ_scalar('val/class_error2', total_class_error2.cpu().item())
        #     self.summ_writer.summ_scalar('val/class_accuracy2', total_class_accuracy2.cpu().item())

        # for key in list(loss_dict_unscaled.keys()):
        #     self.summ_writer.summ_scalar(f'val_unscaled/{key}', loss_dict_unscaled[key])
        # for key in list(loss_dict_scaled.keys()):
        #     self.summ_writer.summ_scalar(f'val_scaled/{key}', loss_dict_scaled[key])

        # map_stat = mapper.get_stats()
        # print("mAP val:", map_stat)
        # self.summ_writer.summ_scalar('val/mAP', map_stat)
        # if args.two_pred_heads:
        #     map_stat2 = mapper2.get_stats()
        #     print("mAP2 val:", map_stat2)
        #     self.summ_writer.summ_scalar('val/mAP2', map_stat2)

        # for episode in range(len(self.mapnames_val)):
        #     img_torch = self.draw_ddetr_predictions(rgb_x[episode], out_all[episode], [targets[episode]], postprocessors=postprocessors_all[episode])
        #     img_torch = F.interpolate(img_torch.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
        #     rgb_x_vis = img_torch.unsqueeze(0) - 0.5
        #     mapname = self.mapnames_val[episode]
        #     name = f'val/detections_{mapname}'
        #     self.summ_writer.summ_rgb(name, rgb_x_vis)
        
        self.model.train()
    
    def save_output(self):

        # mapnames = self.mapnames_train + self.mapnames_val + self.mapnames_test

        # mapnames = mapnames[:40]
        # mapnames = mapnames[40:80]
        # mapnames = mapnames[80:]
        # mapnames = mapnames[100:]

        # print(mapnames)
        # print("LENGTH MAPNAMES", len(mapnames))

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

        

    def draw_ddetr_predictions(
        self, rgb_x, out, targets, 
        id_to_label, postprocessors=None, 
        pred_label_name='pred1', gt_label_name='labels', 
        score_threshold=0.2, text_size=1, indx=0,
        plot_score=False, plot_top_X=None,only_keep_center_detections=False, plot_masks=True, 
        ):

        rect_th=1; text_size=text_size; text_th=1
        color = np.array([0,0,255], dtype='uint8')

        rgb_x = rgb_x[indx] # take indexed rgb

        # rgb_x is 3,H,W, 0-1
        img_ = rgb_x.permute(1,2,0).cpu().numpy()*255.
        rgb_batch = rgb_x.cuda().unsqueeze(0)
        targets_batch = []

        predictions = postprocessors['bbox'](out, torch.as_tensor([self.W, self.H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())
        # if args.do_masks:
        #     predictions = postprocessors['segm'](predictions, out, torch.as_tensor([self.W, self.H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), torch.as_tensor([self.W, self.H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())  
        pred_boxes = predictions[pred_label_name][indx]['boxes']
        pred_labels = predictions[pred_label_name][indx]['labels']
        pred_scores = predictions[pred_label_name][indx]['scores']
        if args.do_masks:
            pred_masks = predictions[pred_label_name][indx]['masks']
        print(pred_scores)
        if pred_boxes.shape[0]>1:
            keep, count = box_ops.nms(pred_boxes, pred_scores, 0.2, top_k=100)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            if plot_top_X is not None and pred_boxes.shape[0]>= plot_top_X:
                keep = torch.topk(pred_scores, k=plot_top_X).indices
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep] 
            else:
                if score_threshold is not None:
                    keep = pred_scores>score_threshold
                    pred_boxes = pred_boxes[keep]
                    pred_labels = pred_labels[keep]
                    pred_scores = pred_scores[keep] 

            if only_keep_center_detections:
                range_x = pred_boxes[:,[0,2]].cpu().numpy()
                in_range_x = np.array([lower <= self.W/2 <= upper for (lower, upper) in range_x])
                range_y = pred_boxes[:,[1,3]].cpu().numpy()
                in_range_y = np.array([lower <= self.H/2 <= upper for (lower, upper) in range_y])
                keep = np.logical_and(in_range_x, in_range_y)
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]
                # pred_scores_labels = pred_scores_labels[keep]

        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        if args.do_masks:
            pred_masks = pred_masks.cpu().numpy()
        print(pred_scores)

        gt_boxes = targets[indx]['boxes']
        if gt_boxes.nelement()>0:
            gt_boxes = gt_boxes.cpu().numpy() * self.W
            gt_boxes = box_ops.box_cxcywh_to_xyxy(torch.from_numpy(gt_boxes)).cpu().numpy()
            gt_labels = targets[indx][gt_label_name].cpu().numpy()
        if args.do_masks:
            gt_masks = targets[indx]['masks'].cpu().numpy()

        # plot predicted boxes
        img = img_.copy()
        for i in range(len(pred_boxes)):
            pred_class_name = str(id_to_label[int(pred_labels[i])])

            pred_score = pred_scores[i]

            cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),(0, 255, 0), rect_th)
            cv2.putText(img,pred_class_name, (int(pred_boxes[i][0:1]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
            if plot_score:
                cv2.putText(img,str(pred_score), (int(pred_boxes[i][2:3]), int(pred_boxes[i][3:4])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
            # cv2.putText(img,str(pred_score), (int(pred_boxes[i][1:2]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
            if args.plot_masks and args.plot_masks:
                pred_mask = np.squeeze(pred_masks[i])
                masked_img = np.where(pred_mask[...,None], color, img)
                img = cv2.addWeighted(img, 0.8, np.float32(masked_img), 0.2,0)

        img2 = img_.copy()
        if targets[indx]['boxes'].nelement()>0:
            for i in range(len(gt_boxes)):
                gt_class_name = str(id_to_label[int(gt_labels[i])])
                cv2.rectangle(img2, (int(gt_boxes[i][0]), int(gt_boxes[i][1])), (int(gt_boxes[i][2]), int(gt_boxes[i][3])),color=(0, 255, 0), thickness=rect_th)
                cv2.putText(img2,gt_class_name, (int(gt_boxes[i][0:1]), int(gt_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
                if args.do_masks and args.plot_masks:
                    gt_mask = np.squeeze(gt_masks[i])
                    masked_img = np.where(gt_mask[...,None], color, img2)
                    img2 = cv2.addWeighted(img2, 0.8, np.float32(masked_img), 0.2,0)

        img_torch2 = torch.from_numpy(img2).cuda().permute(2,0,1)/ 255.
        
        img = torch.from_numpy(img).cuda()/255.
        img = img.permute(2,0,1).float()

        img_torch = torch.cat([img, img_torch2], dim=2)

        return img_torch


class mAP():
    '''
    Util for computing mAP in COCO format
    '''
    def __init__(self, score_boxes_name, score_labels_name): 
        '''
        This function tracks boxes and computes mAP
        '''

        # initialize for mAP
        errors = []

        self.gtFormat = ValidateFormats('xyxy', '-gtformat', errors)
        self.detFormat = ValidateFormats('xyxy', '-detformat', errors)
        self.allBoundingBoxes = BoundingBoxes()
        self.allClasses = []

        self.gtCoordType = ValidateCoordinatesTypes('abs', '-gtCoordinates', errors)
        self.detCoordType = ValidateCoordinatesTypes('abs', '-detCoordinates', errors)
        self.imgSize = (0, 0)
        if self.gtCoordType == CoordinatesType.Relative:  # Image size is required
            assert(False) #imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
        if self.detCoordType == CoordinatesType.Relative:  # Image size is required
            assert(False) #imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)

        self.count = 0

        self.score_boxes_name = score_boxes_name # which prediction head output to use for scoring boxes
        self.score_labels_name = score_labels_name # which prediction head to use for label scores

        self.mAP = {}
        self.classes = {}
        self.aps = {}
        self.ars = {}
        self.mAR = {}
        self.precision = {}
        self.recall = {}


    def add_boxes(
        self, out, postprocessors, W, H, id_to_label, 
        targets, predictions_label_name='labels', 
        do_nms=False, remove_not_openable=False, 
        only_keep_center_detections=False, 
        centroids_xy=None, score_threshold=None, 
        do_instance_map=True, instance_map_with_cat=False,
        ):
        '''
        do_instance_map: Do instance-wise mAP rather than category-wise
        instance_map_with_cat: Should instances also predict the correct class to count as true positive?
        '''
        print("With category?", instance_map_with_cat)
        if do_nms:
            assert(False) # for tracking why NMS??
        # out = outputs[0]
        # postprocessors = outputs[-1]
        for t in range(len(targets)):
            predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), do_masks=False)
            pred_boxes = predictions[self.score_labels_name][t]['boxes']
            pred_labels = predictions[self.score_labels_name][t]['labels']
            # pred_scores = predictions[t]['scores']
            pred_scores_boxes = predictions[self.score_boxes_name][t]['scores']
            pred_scores_labels = predictions[self.score_labels_name][t]['scores']

            if do_instance_map:
                filter_inds = list(targets[t][1]['filter_inds'].cpu().numpy())

            if do_nms and centroids_xy is not None:
                # assert(False)# for tracking why NMS??
                keep_nms, count = box_ops.nms(pred_boxes, pred_scores_boxes, 0.2, top_k=100)
                # pred_boxes = pred_boxes[keep]
                # pred_labels = pred_labels[keep]
                # pred_scores_boxes = pred_scores_boxes[keep]
                # pred_scores_labels = pred_scores_labels[keep]

            # # for openness to remove not openable
            # if remove_not_openable and predictions_label_name=='labels_openness':
            #     keep = pred_labels>0
            #     pred_boxes = pred_boxes[keep]
            #     pred_labels = pred_labels[keep]
            #     pred_scores_boxes = pred_scores_boxes[keep]
            #     pred_scores_labels = pred_scores_labels[keep]

            if only_keep_center_detections:
                assert(False)
                # range_x = pred_boxes[:,[0,2]].cpu().numpy()
                # in_range_x = np.array([lower <= W/2 <= upper for (lower, upper) in range_x])
                # range_y = pred_boxes[:,[1,3]].cpu().numpy()
                # in_range_y = np.array([lower <= H/2 <= upper for (lower, upper) in range_y])
                # keep = np.logical_and(in_range_x, in_range_y)
                # pred_boxes = pred_boxes[keep]
                # pred_labels = pred_labels[keep]
                # pred_scores_boxes = pred_scores_boxes[keep]
                # pred_scores_labels = pred_scores_labels[keep]

            if centroids_xy is not None:
                centroids = centroids_xy[t]
                centroids_keep = []
                for centroid in centroids:
                    range_x = pred_boxes[:,[0,2]].cpu().numpy()
                    in_range_x = np.array([lower <= centroid[0] <= upper for (lower, upper) in range_x])
                    range_y = pred_boxes[:,[1,3]].cpu().numpy()
                    in_range_y = np.array([lower <= centroid[1] <= upper for (lower, upper) in range_y])
                    in_range = np.where(np.logical_and(in_range_x, in_range_y))[0]
                    if len(in_range)==0:
                        centroids_keep.append(-99)
                        continue
                    scores = pred_scores_labels.cpu().numpy()
                    argmax_centroid = in_range[np.argmax(scores[in_range])]
                    centroids_keep.append(argmax_centroid)
                if len(centroids_keep)>0:
                    keep = np.array(centroids_keep)
                else:
                    keep = np.zeros(len(pred_boxes)).astype(bool)
                
                
                # pred_boxes = pred_boxes[keep]
                # pred_labels = pred_labels[keep]
                # pred_scores_boxes = pred_scores_boxes[keep]
                # pred_scores_labels = pred_scores_labels[keep]
            # else:
            #     keep = pred_scores_labels > 0.1 # this filters out the not visible class and extra queries 
            #     pred_boxes = pred_boxes[keep]
            #     pred_labels = pred_labels[keep]
            #     pred_scores_boxes = pred_scores_boxes[keep]
            #     pred_scores_labels = pred_scores_labels[keep]
            #     ≈
            #         keep = pred_scores_labels>score_threshold
            #         pred_boxes = pred_boxes[keep]
            #         pred_labels = pred_labels[keep]
            #         pred_scores_boxes = pred_scores_boxes[keep]
            #         pred_scores_labels = pred_scores_labels[keep]


            
                # pred_scores = pred_scores[keep]
            pred_boxes = pred_boxes.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
            # pred_scores = pred_scores.cpu().numpy() 
            pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
            pred_scores_labels = pred_scores_labels.cpu().numpy() 
            if centroids_xy is not None:
                Z = len(centroids_keep)
            else:
                Z = pred_boxes.shape[0]

            for z in range(Z):
                # prob = probs[z]
                #prob_argmax = int(torch.argmax(prob).cpu().numpy())
                if centroids_xy is not None:
                    idx_z = centroids_keep[z]
                    if z==-99:
                        continue
                    score = pred_scores_labels[idx_z] #prob[prob_argmax].cpu().numpy()
                    pred_box = pred_boxes[idx_z]
                    pred_label = id_to_label[pred_labels[idx_z]]
                else:
                    score = pred_scores_labels[z] #prob[prob_argmax].cpu().numpy()
                    pred_box = pred_boxes[z]
                    pred_label = id_to_label[pred_labels[z]]
                if pred_label=='no_object':
                    continue

                if score < 0.1: # filter not visible class (low confidence for all classes)
                    continue

                if score_threshold is not None:
                    if score < score_threshold:
                        continue

                if do_instance_map:
                    if instance_map_with_cat:
                        pred_label = pred_label + str(z) + '_' + str(self.count) # adjust label by instance index to make it an "instance" category label
                    else:
                        pred_label = str(z) + '_' + str(self.count) # just care that it predicted the correct box for that location
                
                # score = pred_scores[z]
                bbox_params_det = [pred_label, score, pred_box[0], pred_box[1], pred_box[2], pred_box[3]]
                self.allBoundingBoxes, self.allClasses = add_bounding_box(bbox_params_det, self.allBoundingBoxes, self.allClasses, nameOfImage=self.count, isGT=False, imgSize=self.imgSize, Format=self.detFormat, CoordType=self.detCoordType)
            # GT
            # take second view only
            gt_boxes = targets[t][1]['boxes']
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] * W
            gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] * H
            gt_labels = targets[t][1][predictions_label_name]

            # # for openness to remove not openable
            # if remove_not_openable and predictions_label_name=='labels_openness':
            #     keep = gt_labels>0
            #     gt_boxes = gt_boxes[keep]
            #     gt_labels = gt_labels[keep]


            for z in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[z]
                gt_label = id_to_label[int(gt_labels[z])]
                if do_instance_map:
                    filter_ind_z = filter_inds[z]
                    if instance_map_with_cat:
                        gt_label = gt_label + str(filter_ind_z) + '_' + str(self.count) # adjust label by instance index to make it an "instance" category label
                    else:
                        gt_label = str(filter_ind_z) + '_' + str(self.count) # just care that it predicted the correct box for that location
                bbox_params_gt = [gt_label, gt_box[0], gt_box[1], gt_box[2], gt_box[3]]
                self.allBoundingBoxes, self.allClasses = add_bounding_box(bbox_params_gt, self.allBoundingBoxes, self.allClasses, nameOfImage=self.count, isGT=True, imgSize=self.imgSize, Format=self.gtFormat, CoordType=self.gtCoordType)
            self.count += 1

    def get_stats(self, IOU_threshold=0.5):
        mAP, classes, aps, ars, mAR, precision, recall = get_map(self.allBoundingBoxes, self.allClasses, do_st_=False, IOU_threshold=IOU_threshold, consider_nonzero_TP=True) # consider_nonzero_TP=True for instance detection to also consider False positive
        self.mAP[IOU_threshold] = mAP
        self.classes[IOU_threshold] = classes
        self.aps[IOU_threshold] = aps
        self.ars[IOU_threshold] = ars
        self.mAR[IOU_threshold] = mAR
        self.precision[IOU_threshold] = precision
        self.recall[IOU_threshold] = recall
        return mAP


if __name__ == '__main__':
    Ai2Thor()