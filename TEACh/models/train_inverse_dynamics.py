import os
import ipdb
st = ipdb.set_trace
import numpy as np
import random
import cv2
import sys
import numpy as np
import torch
import time
from utils.improc import MetricLogger

from utils.improc import *
import utils.dist
import torch.nn.functional as F 

from arguments import args
import torch.nn as nn
from tqdm import tqdm
from task_base.aithor_base import Base

from nets.inverse_dynamics_model import IDM
from backend import saverloader
import utils.dist

import os
import sys

import os
import cv2
import numpy as np
import time
import random

import sys
import os
import wandb

import torch

# from backend.dataloaders.loader_idm import RICKDataset, my_collate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


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

class Ai2Thor():
    def __init__(self):  

        utils.dist.init_distributed_mode(args) 

        self.initialize_constants()

        print("Getting dataloader...")
        self.initialize_dataloaders()

        num_classes = len(self.include_classes) # minus one to remove no object
        self.score_boxes_name = 'pred1' # only one prediction head so same for both
        self.score_labels_name = 'pred1'

        print(f"Width={self.W}; Height={self.H}")

        self.checkpoint_path = os.path.join(args.checkpoint_path, args.set_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        load_pretrained = True
        self.model = IDM(
            num_classes, 
            load_pretrained, 
            num_actions=len(self.actions2idx),
            actions2idx=self.actions2idx
            )
        self.model.to(device)
        
        if args.distributed:
            print("Running distributed training...")
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu], find_unused_parameters=True)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(name, "requires grad?", param.requires_grad)

        param_dicts = [
            {
                "params":
                    [p for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and not match_name_keywords(n, args.lr_text_encoder_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr * args.lr_backbone_mult,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            },
        ]

        # lr set by arg_parser
        print(f"Learning rate is {args.lr}")
        self.optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_drop = args.lr_drop # every X epochs, drop lr by 0.1
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_drop)

        self.global_step = 0
        self.start_epoch = 0
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
            
            self.global_step, self.start_epoch = saverloader.load_from_path(
                path, 
                self.model, 
                optimizer_load, 
                strict=(not args.load_strict_false), 
                lr_scheduler=lr_scheduler_load,
                )
            self.start_epoch += 1 # need to add one since saving corresponds to trained epoch

        if args.start_one:
            print("Starting at iteration 0 despite checkpoint loaded.")
            self.global_step = 0
            self.start_epoch = 0

        self.log_freq = args.log_freq

        print(f"action mapping is {self.actions2idx}")

        if utils.dist.is_main_process():
            # initialize wandb
            if args.set_name=="test00":
                wandb.init(mode="disabled")
            else:
                wandb.init(project="inverse_dynamics", name=args.set_name)


    def run_train(self):
        print(len(self.train_dataset_loader))
        self.model.train()
        print("Start training")
        for epoch in range(self.start_epoch, args.epochs):
            
            if args.distributed:
                self.train_dataset_loader.sampler.set_epoch(epoch)
        
            self.epoch = epoch

            print("Begin epoch", epoch)
            print("set name:", args.set_name)

            total_loss = self.train_one_epoch()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print(f"loss for epoch {epoch} is: {total_loss}")

            if epoch % args.save_freq_epoch == 0:
                if utils.dist.is_main_process():
                    saverloader.save_checkpoint(
                        self.model, 
                        self.checkpoint_path, 
                        self.global_step, 
                        self.epoch, 
                        self.optimizer, 
                        keep_latest=args.keep_latest, 
                        lr_scheduler=self.lr_scheduler
                        )
            
            self.run_validation(epoch, partial=False)

    def train_one_epoch(self):
        metric_logger = MetricLogger(delimiter="  ")
        header = f'TRAIN | {args.set_name} | Epoch: [{self.epoch}/{args.epochs-1}]'
        for i_batch, sample_batched in enumerate(metric_logger.log_every(self.train_dataset_loader, 10, header)):

            if self.global_step % self.log_freq == 0:
                self.log_iter = True
            else:
                self.log_iter = False

            out_dict = self.run_model(sample_batched)

            total_loss = out_dict['losses']
            loss_value = out_dict['loss_value']
            loss_dict_reduced_unscaled = out_dict['loss_dict_reduced_unscaled'] 
            loss_dict_reduced_scaled = out_dict['loss_dict_reduced_scaled'] 

            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()

            if utils.dist.is_main_process():
                wandb.log({"train/total_loss": loss_value, 'epoch': self.epoch, 'batch': i_batch})
                loss_dict_reduced_unscaled.update({'epoch': self.epoch, 'batch': i_batch})
                loss_dict_reduced_scaled.update({'epoch': self.epoch, 'batch': i_batch})
                loss_dict_reduced_unscaled = {f"train/{k}": v for k, v in loss_dict_reduced_unscaled.items()}
                loss_dict_reduced_scaled = {f"train/{k}": v for k, v in loss_dict_reduced_scaled.items()}
                wandb.log(loss_dict_reduced_unscaled)
                wandb.log(loss_dict_reduced_scaled)
                
            metric_logger.update(loss=loss_value)

            self.global_step += 1

            if self.global_step % args.log_freq == 0:
                self.visualize_predictions(
                    sample_batched["images"],
                    sample_batched["targets"],
                    out_dict,
                    "train",
                )

            # if self.global_step % args.val_freq == 0:
            #     self.run_validation(epoch=self.epoch, partial=True)
    
        return total_loss

    @torch.no_grad()
    def visualize_predictions(
        self,
        images,
        targets,
        predictions,
        split,
        reverse_transform = True,
    ):
        # use first prediction
        images_vis = images[0].cpu().numpy()
        pred_actions = self.actions[int(torch.argmax(nn.Softmax(dim=1)(predictions['outputs']['pred_actions'][0:1].squeeze(1))).squeeze().cpu().numpy())]
        pred_label = self.id_to_name[int(torch.argmax(nn.Softmax(dim=1)(predictions['outputs']['pred_logits'][0:1].squeeze(1))).squeeze().cpu().numpy())]
        gt_action = self.actions[int(targets[0]['expert_action'][0].cpu().numpy())]
        gt_label = self.id_to_name[int(targets[0]['labels'][0].cpu().numpy())]
        rgb1 = images_vis[0]
        rgb2 = images_vis[1]
        if reverse_transform:
            image_mean = np.array([0.485,0.456,0.406]).reshape(3,1,1)
            image_std = np.array([0.229,0.224,0.225]).reshape(3,1,1)
            rgb1 = rgb1 * image_std + image_mean
            rgb2 = rgb2 * image_std + image_mean
        rgb1 = rgb1*255.
        rgb1 = rgb1.transpose(1,2,0).astype(np.float32)
        rgb1 = np.float32(rgb1.copy())
        rgb2 = rgb2*255.
        rgb2 = rgb2.transpose(1,2,0).astype(np.float32)
        rgb2 = np.float32(rgb2.copy())
        cv2.putText(rgb1, f'GT: {gt_action} {gt_label}', (int(40*(self.H/480)),int(40*(self.W/480))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)
        cv2.putText(rgb2, f'Pred: {pred_actions} {pred_label}', (int(40*(self.H/480)),int(40*(self.W/480))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)
        img_torch1 = torch.from_numpy(rgb1).to(device).permute(2,0,1)/ 255.
        img_torch2 = torch.from_numpy(rgb2).to(device).permute(2,0,1)/ 255.
        img_torch = torch.cat([img_torch1, img_torch2], dim=2)
        rgb_x_vis = img_torch.unsqueeze(0) - 0.5
        name = f'{split}/prediction'
        wandb.log({name: wandb.Image(rgb_x_vis)})

    @torch.no_grad()
    def run_validation_split(self, dataloader, split, epoch, do_map_eval=False):
        print(f"Evaluating {split}...")

        self.model.eval()
        total_loss = torch.tensor(0.0).to(device)
        count = 0

        metric_logger = MetricLogger(delimiter="  ")
        header = f'{split} | {args.set_name} | Epoch: [{epoch}/{args.epochs-1}]'
        for i_batch, sample_batched in enumerate(metric_logger.log_every(dataloader, 10, header)):
            out_dict = self.run_model(sample_batched)
            loss_value = out_dict['loss_value']
            loss_dict_reduced_unscaled = out_dict['loss_dict_reduced_unscaled'] 
            loss_dict_reduced_scaled = out_dict['loss_dict_reduced_scaled'] 

            total_loss += loss_value

            if i_batch == 0:
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

            torch.cuda.synchronize()

            count += 1

            metric_logger.update(loss=total_loss/count)

            if args.max_validation_iters is not None:
                if i_batch>=args.max_validation_iters:
                    break

        if utils.dist.is_main_process():
            loss_dict_unscaled = {f"{split}/{k}": v / count for k, v in loss_dict_unscaled.items()}
            loss_dict_scaled = {f"{split}/{k}": v / count for k, v in loss_dict_scaled.items()}
            loss_dict_unscaled.update({'epoch': epoch})
            loss_dict_scaled.update({'epoch': epoch})
            wandb.log({f"{split}/total_loss": total_loss / count, 'epoch': epoch})
            wandb.log(loss_dict_unscaled)
            wandb.log(loss_dict_scaled)

            if self.global_step % args.log_freq == 0:
                self.visualize_predictions(
                    sample_batched["images"],
                    sample_batched["targets"],
                    out_dict,
                    split,
                )

        print(f"{args.set_name}: {split} loss for iter {self.global_step} is: {total_loss/count}")

        self.model.train()

        if utils.dist.is_main_process():
            wandb.log({f'{split}/total_loss_epoch': total_loss, 'epoch': epoch})

        del dataloader # stop dataloader processes

        return total_loss

    def run_model(self, sample_batched):

        # put on cuda
        images = sample_batched["images"].to(device)
        
        targets = sample_batched["targets"]
        for t in targets:
            for k in t.keys():
                t[k] = t[k].to(device)
                
        out_dict = self.model(
            images, 
            targets, 
            )

        return out_dict

    @torch.no_grad()
    def run_validation(self, epoch=0, partial=False, do_map_eval=False):
        if args.run_valid_seen:
            split = "valid_seen" 
            dataloader = self.valid_seen_dataset_loader_full if not partial else self.valid_seen_dataset_loader
            self.run_validation_split(dataloader, split, epoch, do_map_eval)

        if args.run_valid_unseen:
            split = "valid_unseen"
            dataloader = self.valid_unseen_dataset_loader_full if not partial else self.valid_unseen_dataset_loader
            self.run_validation_split(dataloader, split, epoch, do_map_eval)

    def initialize_dataloaders(self):
        if args.train_on_teach:
            from backend.dataloaders.loader_idm_teach import RICKDataset, my_collate
        else:
            from backend.dataloaders.loader_idm import RICKDataset, my_collate
        train_dataset = RICKDataset(
            split="train",
            W=args.W, 
            H=args.H, 
            idx2actions=self.actions,
            id_to_name=self.id_to_name,
            end_index=args.max_episodes_train,
            shuffle=args.shuffle,
        )
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(train_dataset)
        train_dataset_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size, 
            sampler=sampler,
            num_workers=args.num_workers, 
            collate_fn=my_collate,
            pin_memory=True,
            drop_last=True
        )
        self.train_dataset_loader = train_dataset_loader
        print("Size train dataloader:", len(self.train_dataset_loader))

        if args.run_valid_seen:

            valid_seen_dataset = RICKDataset(
                split="valid_seen",
                W=args.W, 
                H=args.H, 
                idx2actions=self.actions,
                id_to_name=self.id_to_name,
                shuffle=args.shuffle,
            )
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(valid_seen_dataset, shuffle=args.validation_shuffle)
            elif args.validation_shuffle:
                sampler = torch.utils.data.RandomSampler(valid_seen_dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(valid_seen_dataset)

            valid_seen_dataset_loader = torch.utils.data.DataLoader(
                valid_seen_dataset,
                batch_size=args.batch_size*args.val_batch_mult, 
                sampler=sampler,
                shuffle=args.validation_shuffle,
                num_workers=4, #args.num_workers, 
                collate_fn=my_collate,
                pin_memory=True,
                drop_last=True
            )
            self.valid_seen_dataset_loader_full = valid_seen_dataset_loader
            self.valid_seen_dataset_loader = valid_seen_dataset_loader

            print("Size valid seen dataloader:", len(self.valid_seen_dataset_loader))
            print("Size valid seen FULL dataloader:", len(self.valid_seen_dataset_loader_full))

        if args.run_valid_unseen:

            valid_unseen_dataset = RICKDataset(
                split="valid_unseen",
                W=args.W, 
                H=args.H, 
                idx2actions=self.actions,
                id_to_name=self.id_to_name,
                shuffle=args.shuffle,
            )
            
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(valid_unseen_dataset, shuffle=args.validation_shuffle)
            elif args.validation_shuffle:
                sampler = torch.utils.data.RandomSampler(valid_unseen_dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(valid_unseen_dataset)
            valid_unseen_dataset_loader = torch.utils.data.DataLoader(
                valid_unseen_dataset,
                batch_size=args.batch_size*args.val_batch_mult, 
                sampler=sampler,
                shuffle=args.validation_shuffle,
                num_workers=4, #args.num_workers, 
                collate_fn=my_collate,
                pin_memory=True,
                drop_last=True
            )
            self.valid_unseen_dataset_loader_full = valid_unseen_dataset_loader
            self.valid_unseen_dataset_loader = valid_unseen_dataset_loader

            print("Size valid unseen dataloader:", len(self.valid_unseen_dataset_loader))
            print("Size valid unseen FULL dataloader:", len(self.valid_unseen_dataset_loader_full))

    def initialize_constants(self):
        self.W, self.H = args.W, args.H

        self.include_classes = [
            'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
            'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
            'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
            'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
            'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
            'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
            'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
            'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
            'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
            'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch']
        
        alfred_objects, object_mapping  = get_alfred_constants()

        for obj in alfred_objects:
            if obj not in self.include_classes:
                self.include_classes.append(obj)

        self.special_classes = ['AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']
        self.include_classes += self.special_classes
        
        self.include_classes += ["no_object"]

        if args.train_on_teach:
            actions = [
                'Forward',
                'Backward', 
                'Turn Right', 
                'Turn Left', 
                'Look Down',
                'Look Up',
                'Pan Left',
                'Pan Right',
                'Pickup', 
                'Place', 
                'Open', 
                'Close', 
                'Slice',
                'ToggleOn',
                'ToggleOff',
                'Pour',
                ]
        else:
            actions = [
                'MoveAhead', 
                'RotateRight', 
                'RotateLeft', 
                'LookDown',
                'LookUp',
                'PickupObject', 
                'PutObject', 
                'OpenObject', 
                'CloseObject', 
                'SliceObject',
                'ToggleObjectOn',
                'ToggleObjectOff',
                # 'Done',
                ]

        subgoals = [
            'PickupObject', 
            'PutObject', 
            'OpenObject', 
            'CloseObject', 
            'SliceObject',
            'GotoLocation',
            'HeatObject',
            "ToggleObject",
            "CleanObject",
            "HeatObject",
            "CoolObject",
            ]
        
        self.actions = {i:actions[i] for i in range(len(actions))}
        self.actions2idx = {actions[i]:i for i in range(len(actions))}
        self.subgoals2idx = {subgoals[i]:i for i in range(len(subgoals))}
        self.idx2subgoals = {i:subgoals[i] for i in range(len(subgoals))}

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.include_classes:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

def get_alfred_constants():

    OBJECTS = [
            'AlarmClock',
            'Apple',
            'ArmChair',
            'BaseballBat',
            'BasketBall',
            'Bathtub',
            'BathtubBasin',
            'Bed',
            'Blinds',
            'Book',
            'Boots',
            'Bowl',
            'Box',
            'Bread',
            'ButterKnife',
            'Cabinet',
            'Candle',
            'Cart',
            'CD',
            'CellPhone',
            'Chair',
            'Cloth',
            'CoffeeMachine',
            'CounterTop',
            'CreditCard',
            'Cup',
            'Curtains',
            'Desk',
            'DeskLamp',
            'DishSponge',
            'Drawer',
            'Dresser',
            'Egg',
            'FloorLamp',
            'Footstool',
            'Fork',
            'Fridge',
            'GarbageCan',
            'Glassbottle',
            'HandTowel',
            'HandTowelHolder',
            'HousePlant',
            'Kettle',
            'KeyChain',
            'Knife',
            'Ladle',
            'Laptop',
            'LaundryHamper',
            'LaundryHamperLid',
            'Lettuce',
            'LightSwitch',
            'Microwave',
            'Mirror',
            'Mug',
            'Newspaper',
            'Ottoman',
            'Painting',
            'Pan',
            'PaperTowel',
            'PaperTowelRoll',
            'Pen',
            'Pencil',
            'PepperShaker',
            'Pillow',
            'Plate',
            'Plunger',
            'Poster',
            'Pot',
            'Potato',
            'RemoteControl',
            'Safe',
            'SaltShaker',
            'ScrubBrush',
            'Shelf',
            'ShowerDoor',
            'ShowerGlass',
            'Sink',
            'SinkBasin',
            'SoapBar',
            'SoapBottle',
            'Sofa',
            'Spatula',
            'Spoon',
            'SprayBottle',
            'Statue',
            'StoveBurner',
            'StoveKnob',
            'DiningTable',
            'CoffeeTable',
            'SideTable',
            'TeddyBear',
            'Television',
            'TennisRacket',
            'TissueBox',
            'Toaster',
            'Toilet',
            'ToiletPaper',
            'ToiletPaperHanger',
            'ToiletPaperRoll',
            'Tomato',
            'Towel',
            'TowelHolder',
            'TVStand',
            'Vase',
            'Watch',
            'WateringCan',
            'Window',
            'WineBottle',
        ]

    # SLICED = [
    #     'AppleSliced',
    #     'BreadSliced',
    #     'LettuceSliced',
    #     'PotatoSliced',
    #     'TomatoSliced'
    # ]

    # OBJECTS += SLICED

    # object parents
    OBJ_PARENTS = {obj: obj for obj in OBJECTS}
    OBJ_PARENTS['AppleSliced'] = 'Apple'
    OBJ_PARENTS['BreadSliced'] = 'Bread'
    OBJ_PARENTS['LettuceSliced'] = 'Lettuce'
    OBJ_PARENTS['PotatoSliced'] = 'Potato'
    OBJ_PARENTS['TomatoSliced'] = 'Tomato'

    return OBJECTS, OBJ_PARENTS

if __name__ == '__main__':
    Ai2Thor()