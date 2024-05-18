import copy
import logging
import random
import numpy as np
import pycocotools.mask as mask_util
import itertools
from imageio import imread
from typing import List, Union
import torch
from torch.nn import functional as F
from operator import itemgetter
from natsort import natsorted
import os
from pathlib import Path
import yaml
import torch.nn as nn

import matplotlib.pyplot as plt

import sys
sys.path.append('Mask2Former3D')
from mask2former_video.modeling.backbone.resnet import build_resnet_backbone_custom
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.layers import ShapeSpec
from torch_scatter import scatter_mean, scatter_min

import copy
import itertools
import logging
import os
import gc
import weakref
import utils.geom

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import ipdb
st = ipdb.set_trace

from detectron2.structures import Boxes, ImageList, Instances

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    launch,
    AMPTrainer,
    SimpleTrainer
)
from detectron2.evaluation import (
    DatasetEvaluator,
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from arguments import args
import numpy as np

# MaskFormer
from mask2former import add_maskformer2_config
from mask2former_video.data_video.dataset_mapper_coco import COCOInstanceNewBaselineDatasetMapper
from mask2former_video.global_vars import NAME_MAP20, AI2THOR_NAME_MAP

from mask2former_video import (
    ScannetDatasetMapper,
    Scannet3DEvaluator,
    ScannetSemantic3DEvaluator,
    COCOEvaluatorMemoryEfficient,
    add_maskformer2_video_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
    build_detection_train_loader_multi_task,
)
from detectron2.data import build_detection_test_loader as build_detection_test_loader_detectron2

from mask2former_video.modeling.backproject.backproject import backprojector_dataloader, multiscsale_voxelize, interpolate_feats_3d

from utils.plot_utils import plot_3d, plot_object_tracks

device = "cuda" if torch.cuda.is_available() else "cpu"

from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, ImageList, Instances
import wandb
import cv2

def add_custom_config(cfg):
    cfg.OUTPUT_DIR = './data'
    cfg.SOLVER.IMS_PER_BATCH = 1 
    cfg.SOLVER.CHECKPOINT_PERIOD = 4000 
    cfg.TEST.EVAL_PERIOD = 8000 
    cfg.INPUT.FRAME_LEFT = 5 
    cfg.INPUT.FRAME_RIGHT = 5 
    cfg.INPUT.SAMPLING_FRAME_NUM = 11 
    cfg.MODEL.SUPERVISE_NO_BATCH = True
    
    
    cfg.SOLVER.BASE_LR = 1e-4 
    cfg.INPUT.IMAGE_SIZE = 512 
    cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE = True
    cfg.MODEL.CROSS_VIEW_METHOD = "panet_lite"
    cfg.INPUT.CAMERA_DROP = True
    cfg.INPUT.STRONG_AUGS = True
    cfg.INPUT.COLOR_AUG = True
    cfg.MODEL.KNN = 8 
    cfg.INPUT.AUGMENT_3D = True
    cfg.INPUT.VOXELIZE = True
    cfg.INPUT.SAMPLE_CHUNK_AUG = True
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 50000 
    cfg.INPUT.VOXELIZE = True
    cfg.MODEL.CROSS_VIEW_BACKBONE = True
    cfg.MODEL.CROSS_VIEW_NUM_LAYERS = [2,2,6,2] 
    cfg.DO_TRILINEAR_INTERPOLATION = True
    cfg.INTERP_NEIGHBORS = 8 
    cfg.DATASETS.TRAIN = ('ai2thor_highres_train_single',)
    cfg.DATASETS.TEST = ('ai2thor_highres_val_single','ai2thor_highres_train_eval_single')
    cfg.MODEL.PIXEL_DECODER_PANET = True
    
    cfg.SKIP_CLASSES = None
    cfg.MODEL.FREEZE_BACKBONE = True
    cfg.SOLVER.TEST_IMS_PER_BATCH = 2 
    cfg.SAMPLING_STRATEGY = "consecutive" 
    cfg.INPUT.CHUNK_AUG_MAX = 5 
    cfg.SAMPLED_CROSS_ATTENTION = True
    cfg.SOLVER.MAX_ITER = 200000 
    cfg.DATALOADER.NUM_WORKERS = 8 
    cfg.DATALOADER.TEST_NUM_WORKERS = 2 
    cfg.MAX_FRAME_NUM = -1 
    cfg.DO_FLIPPING = False 
    cfg.DO_ELASTIC_DISTORTION = False 
    cfg.INPUT.INPAINT_DEPTH = False 
    cfg.INPUT.MIN_SIZE_TEST = 512 
    cfg.INPUT.MAX_SIZE_TEST = 512 
    cfg.TEST.TEST_ALL = True
    cfg.IGNORE_DEPTH_MAX = 15.0 
    cfg.MODEL.SUPERVISE_SPARSE = True
    cfg.TEST.EVAL_SPARSE = True
    cfg.HIGH_RES_SUBSAMPLE = True
    cfg.HIGH_RES = True
    cfg.DO_HIGH_RES_PANET = True
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 13 
    cfg.INPUT.VOXEL_SIZE = [0.04,0.08,0.16,0.32] 
    cfg.SAMPLE_SIZES = [200,800,3200,12800] 
    cfg.HIGH_RES_INPUT = True
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.NO_UPSAMPLE_EVAL = False 

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 123
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 150
    cfg.MODEL.WEIGHTS = args.odin_checkpoint #'./checkpoints/model_0002999.pth'

    return cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file('./Mask2Former3D/configs/scannet_context/3d.yaml')
    cfg = add_custom_config(cfg)
    # cfg.merge_from_file(args.config_file)
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former_video")
    return cfg

def _get_dataset_instances_meta(dataset='ai2thor', ai2thor_name_map=None):
    if dataset == 'ai2thor':
        if ai2thor_name_map is None:
            name_map = AI2THOR_NAME_MAP
        else:
            name_map = ai2thor_name_map
    elif dataset == 'replica':
        name_map = NAME_MAP20
    else:
        assert False, 'dataset not supported: {}'.format(dataset)
    dataset_categories = [
        {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in name_map.items()
    ]
    thing_ids = [k["id"] for k in dataset_categories]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in dataset_categories]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

class ODIN(nn.Module):

    def __init__(self, confidence_threshold):
        super(ODIN, self).__init__()

        cfg = setup(args)

        self.cfg = cfg
        self.W, self.H = args.W, args.H
        self.size_divisibility = 32

        self.pixel_mean = torch.tensor(self.cfg.MODEL.PIXEL_MEAN).to(device).unsqueeze(1).unsqueeze(1)
        self.pixel_std = torch.tensor(self.cfg.MODEL.PIXEL_STD).to(device).unsqueeze(1).unsqueeze(1)

        self.device = device

        self.supervise_sparse = False 
        self.eval_sparse = False 

        self.confidence_threshold = confidence_threshold
        self.target_object_threshold = args.confidence_threshold_searching 

        if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
            self.backbone = build_resnet_backbone_custom(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))).to(device)
        else:
            self.backbone = build_backbone(cfg).to(device)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape()).to(device)

        self.AI2THOR_CLASS_NAMES =  [
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
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch', 'Wall', 'Floor']
        self.AI2THOR_CLASS_NAMES.extend(['AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced'])

        AI2THOR_NAME_MAP  = {i+1: name for i, name in enumerate(self.AI2THOR_CLASS_NAMES)}

        self.metadata = _get_dataset_instances_meta(ai2thor_name_map=AI2THOR_NAME_MAP)

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.AI2THOR_CLASS_NAMES:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

        self.max_odin_images = args.max_odin_images


        self.W = args.W
        self.H = args.H
        self.web_window_size = args.W
        self.fov = args.fov
        print(f"fov: {self.fov}")
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.world_t_weird = torch.from_numpy(np.array(
                                    [
                                        [1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0,-1, 0, 0],
                                        [0, 0, 0, 1]
                                    ],
                                    dtype=np.float32
                                )).float().to(self.device)

    @torch.no_grad()
    def get_masks(
        self, 
        input_dict,
        target_class=None,
        score_threshold=None,
        id_to_mapped_id=None,
        ):

        input_dict_ = {} 
        for k in input_dict.keys(): 
            input_dict_[k] = [torch.tensor(t) for t in input_dict[k][-self.max_odin_images:]]

        for depth_idx in range(len(input_dict_['depths'])):
            input_dict_['depths'][depth_idx][input_dict_['depths'][depth_idx]>15.] = 0.

        dataset_dict = self.get_multiview_xyz(input_dict_)

        dataset_dict['original_xyz'] = dataset_dict['original_xyz'].to(self.device)
        
        # masks: num_queries, num_views, H, W
        # pred_classes: num_queries
        # scores: num_queries
        with torch.no_grad():
            masks, pred_classes, scores = self.forward([dataset_dict])

        if id_to_mapped_id is not None:
            # map desired labels to another label
            for idx in range(len(pred_classes)):
                if pred_classes[idx] in id_to_mapped_id.keys():
                    pred_classes[idx] = id_to_mapped_id[pred_classes[idx]]

        if score_threshold is not None:
            score_threshold_ = self.confidence_threshold
        else:
            if target_class is not None:
                score_threshold_ = self.target_object_threshold
            else:
                score_threshold_ = self.confidence_threshold
        print('score threshold', score_threshold_)
        # print(scores)
        keep = scores>score_threshold_
        masks, pred_classes, scores = masks[keep], pred_classes[keep], scores[keep]

        if target_class is not None:
            if isinstance(target_class, str):
                keep = pred_classes==self.name_to_id[target_class]
            elif isinstance(target_class, int):
                keep = pred_classes==target_class
            else:
                assert(False) # what data type is this?
            masks, pred_classes, scores = masks[keep], pred_classes[keep], scores[keep]

        visualize = False
        if visualize:

            targets = {}
            targets['masks'] = masks
            targets['labels'] = pred_classes
            self.visualize_2d(torch.stack(dataset_dict['images']).unsqueeze(0), [targets], {'masks': 'masks', 'labels': 'labels'})

        centroids = torch.zeros(len(masks), 3)
        # masks_ = torch.zeros(len(masks), self.W, self.H)
        crops = np.zeros((len(masks), 256, 256,3)).astype(np.uint8)
        images = torch.stack(dataset_dict['images']).cpu().numpy()
        masks_ = masks.cpu().numpy()
        to_remove = []
        to_keep = []
        for object_idx in range(len(masks)):
            
            mask_obj = masks[object_idx].unsqueeze(-1).repeat(1,1,1,3)
            
            obj_point_cloud_adjusted = utils.geom.apply_4x4(utils.geom.safe_inverse_single(self.world_t_weird), dataset_dict['original_xyz'][mask_obj].reshape(-1, 3).unsqueeze(0)).squeeze(0)

            c_depth = torch.median(obj_point_cloud_adjusted, dim=0).values
            centroids[object_idx] = c_depth

            mask_idx = torch.argmax(torch.sum(masks[object_idx].flatten(1,2),1))

            if np.sum(masks_[object_idx][mask_idx])<5:
                to_remove.append(object_idx)
                continue
            else:
                to_keep.append(object_idx)

            crop = self.get_crop_from_mask(images[mask_idx].transpose(1,2,0), masks_[object_idx][mask_idx])
            crop = cv2.resize(crop, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            crops[object_idx] = crop

        to_keep = torch.tensor(to_keep)

        if len(to_keep)==0:
            pass
        elif len(to_keep)==1:
            masks_, pred_classes, scores, centroids, crops = masks_[to_keep[0]:to_keep[0]+1], pred_classes[to_keep[0]:to_keep[0]+1].cpu().numpy(), scores[to_keep[0]:to_keep[0]+1].cpu().numpy(), centroids[to_keep[0]:to_keep[0]+1].cpu().numpy(), crops[to_keep[0]:to_keep[0]+1]
        else:
            masks_, pred_classes, scores, centroids, crops = masks_[to_keep], pred_classes[to_keep].cpu().numpy(), scores[to_keep].cpu().numpy(), centroids[to_keep].cpu().numpy(), crops[to_keep]

        return masks_, pred_classes, scores, centroids, crops

    def mask_to_box(self, mask, padding_percent = 0.1):

        segmentation = np.where(mask == True)

        # Bounding Box
        bbox = np.asarray([0, 0, 0, 0])
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))

            padding = max(y_max-y_min, x_max-x_min)
            padding = int(np.ceil(padding * padding_percent))

            bbox = np.asarray([max(0, x_min-padding), min(x_max+padding, self.W-1), max(0, y_min-padding), min(y_max+padding, self.H-1)])

        return bbox

    def get_crop_from_mask(self, rgb, mask):
        box = self.mask_to_box(mask)
        x_min, x_max, y_min, y_max = box
        cropped = rgb[y_min:y_max, x_min:x_max]
        return cropped
    
    def filter_outputs(self, masks, pred_classes, scores):

        where_thresholded = np.where(scores>self.confidence_threshold)

    def visualize_2d(self, vis_images, targets, index_names, field_name='gt', gt_targets=None):
        """
        vis_images: B, V, 3, H, W
        targets: B
        index_names: {'masks': 'gt_masks', 'labels': 'labels'}
        """
        B, V, _, H, W = vis_images.shape
        coco_metadata = self.metadata
        mask_index = index_names['masks']
        label_index = index_names['labels']

        for i in range(B):
            images = []
            for j in range(V):
                im = vis_images[i, j].permute(1, 2, 0).cpu().numpy()
                v = Visualizer(im, coco_metadata)
                predictions = Instances((H, W))
                predictions.pred_masks= targets[i][mask_index][:, j].cpu().numpy()
                predictions.pred_classes = targets[i][label_index].cpu().numpy()
                instance_result = v.draw_instance_predictions(
                    predictions).get_image()
                images.append(instance_result)

            image = np.concatenate(images, axis=1)
        
            if gt_targets is not None:
                gt_images = []
                for j in range(V):
                    im = vis_images[i, j].permute(1, 2, 0).cpu().numpy()
                    v = Visualizer(im, coco_metadata)
                    predictions = Instances((H, W))
                    predictions.pred_masks= gt_targets[i][mask_index][:, j].cpu().numpy()
                    predictions.pred_classes = gt_targets[i][label_index].cpu().numpy()
                    instance_result = v.draw_instance_predictions(
                        predictions).get_image()
                    gt_images.append(instance_result)
            
                gt_image = np.concatenate(gt_images, axis=1)
                image = np.concatenate([image, gt_image], axis=0)
            wandb.log({field_name: [wandb.Image(image)]})

            return image

    def get_multiview_xyz(self, dataset_dict):
        scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        v = len(dataset_dict["images"])
        h, w = dataset_dict["images"][0].shape[:2]
        pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
        pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
        H_padded = h + pad_h
        W_padded = w + pad_w
        features = {
            k: torch.zeros(v, 1, H_padded//s, W_padded//s) for k, s in scales.items()
        }
        depths = torch.stack(dataset_dict["depths"]).float()
        poses = torch.stack(dataset_dict["poses"]).float()
        intrinsics = torch.stack(dataset_dict["intrinsics"]).float()
        augment = False 
        scannet_pc = None
        multi_scale_xyz, scannet_pc, original_xyz = backprojector_dataloader(
            list(features.values()), depths, poses,
            intrinsics, augment=augment, 
            method=self.cfg.MODEL.INTERPOLATION_METHOD, 
            scannet_pc=scannet_pc, padding=(pad_h, pad_w), 
            do_flipping=self.cfg.DO_FLIPPING, 
            mask_valid=self.cfg.MASK_VALID, 
            do_elastic_distortion=self.cfg.DO_ELASTIC_DISTORTION,)

        multi_scale_xyz = multi_scale_xyz[::-1]

        dataset_dict['multi_scale_xyz'] = multi_scale_xyz

        dataset_dict['original_xyz'] = original_xyz[0]

        visualize=False
        if visualize: # and np.random.uniform()<0.05:
            x_data, y_data, z_data = dataset_dict['original_xyz'].reshape(-1, 3).cpu().unbind(-1)
            bmin = dataset_dict['original_xyz'].reshape(-1, 3).min().item()
            bmax = dataset_dict['original_xyz'].reshape(-1, 3).max().item()
            def plot_3d(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45)):
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
                ax.view_init(view[0], view[1])
                ax.set_xlim(b_min, b_max)
                ax.set_ylim(b_min, b_max)
                ax.set_zlim(b_min, b_max)
                ax.scatter3D(xdata, ydata, zdata, c=color, cmap='rgb', s=0.1)
            plot_3d(x_data.numpy(), y_data.numpy(), z_data.numpy(), color=torch.stack(dataset_dict['images']).reshape(-1, 3) / 255.0)
            plt.savefig('output/test.png')
            st()
            
        return dataset_dict
        

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            video["decoder_3d"] = self.cfg.MODEL.DECODER_3D
            video['images'] = [image.permute(2, 0, 1) for image in video["images"]]
            for image in video["images"]:
                images.append(image.to(self.device))
        bs = len(batched_inputs)
        v = len(batched_inputs[0]["images"])
        
        decoder_3d = torch.tensor(sum([batched_input["decoder_3d"] for batched_input in batched_inputs]), device=self.device)
        
        if self.cfg.MULTI_TASK_TRAINING:
            eff_bs = len(batched_inputs)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(decoder_3d)
                eff_bs *= get_world_size()
            decoder_3d = decoder_3d.item()
            assert decoder_3d == eff_bs or decoder_3d == 0, "All videos must have the same decoder_3d value"
        decoder_3d = decoder_3d > 0

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        H_padded, W_padded = images.tensor.shape[-2:]

        depths, poses, intrinsics, valids, segments = None, None, None, None, None
        
        depths, poses, valids, segments, intrinsics, multiview_data = self.load_3d_data(
            batched_inputs,
            images_shape=[bs, v, H_padded, W_padded]
        )
        
        if self.cfg.MODEL.CROSS_VIEW_BACKBONE and decoder_3d:
            features = self.backbone(
                images.tensor,
                multiview_data['multi_scale_xyz'],
                shape=[bs, v, H_padded, W_padded],
                multiview_data=multiview_data, 
                decoder_3d=decoder_3d
            )
        else:
            features = self.backbone(images.tensor, decoder_3d=decoder_3d)
            
        # check for nans in features
        if torch.isnan(features['res4']).any():
            st()
        
        color_features, idxs = None, None
        if decoder_3d:
            color_features, idxs = self.get_color_features(
                multiview_data, images, features, shape=[bs, v, H_padded, W_padded]
            )
    
        outputs = self.sem_seg_head(
            features,
            depths=depths,
            poses=poses,
            intrinsics=intrinsics,
            shape=[bs, v, H_padded, W_padded],
            multiview_data=multiview_data,
            valids=valids,
            decoder_3d=decoder_3d, 
            color_features=color_features,
        )
                
        torch.cuda.empty_cache()

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_pred_results = self.upsample_pred_masks(
            mask_pred_results, decoder_3d, batched_inputs,
            multiview_data, images, [bs, v, H_padded, W_padded]
        )
        
        scores = F.softmax(mask_cls_results[0], dim=-1)[:, :-1]
        num_classes = self.sem_seg_head.num_classes
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES, 1).flatten(0, 1)
        test_topk_per_image = self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES #100
        
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred_results = mask_pred_results[topk_indices]
        
        masks = mask_pred_results > 0.
        mask_scores_per_image = (mask_pred_results.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        mask_scores_per_image = (mask_pred_results.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        
        scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image + 1
        return masks, pred_classes, scores

    def load_3d_data(self, batched_inputs, images_shape):
        depths, poses, segments, valids, intrinsics = None, None, None, None, None
        multiview_data = None
        bs, v = images_shape[:2]
        h, w = batched_inputs[0]['images'][0].shape[-2:]
        decoder_3d = batched_inputs[0]['decoder_3d']

        if self.supervise_sparse or self.eval_sparse:
            valids = torch.stack(
                [torch.stack(video["valids"]).to(self.device) for video in batched_inputs]
            ).reshape(bs, v, h, w)
            # pad the depth image with size divisibility
            if self.size_divisibility > 1:
                pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
                pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)

        if decoder_3d:
            depths = torch.stack(
                [torch.stack(video["depths"]).to(self.device) for video in batched_inputs]
            )
            intrinsics =  torch.stack(
                [torch.stack(video["intrinsics"]).to(self.device) for video in batched_inputs]
            )
            b, v, h, w = depths.shape
            # pad the depth image with size divisibility
            if self.size_divisibility > 1:
                pad_h = int(np.ceil(h / self.size_divisibility) * self.size_divisibility - h)
                pad_w = int(np.ceil(w / self.size_divisibility) * self.size_divisibility - w)
                depths = F.pad(depths, (0, pad_w, 0, pad_h), mode="constant", value=0)
            assert list(depths.shape[-2:]) == images_shape[-2:], "depth and image size mismatch"
             
            poses = torch.stack(
                [torch.stack(video["poses"]).to(self.device) for video in batched_inputs]
            )
            
            multiview_data = {}
            multiview_data["multi_scale_xyz"] = [
                torch.stack([batched_inputs[i]["multi_scale_xyz"][j] for i in range(bs)], dim=0).to(self.device) for j in range(len(batched_inputs[0]["multi_scale_xyz"]))
            ]

            voxel_size = self.cfg.INPUT.VOXEL_SIZE[::-1]

            if self.cfg.HIGH_RES:
                voxel_size.append(0.02)
                multiview_data["multi_scale_xyz"].append(
                    torch.stack([batched_inputs[i]["original_xyz"] for i in range(bs)], dim=0).to(self.device)
                )
            
            if self.cfg.INPUT.VOXELIZE:
                multiview_data["multi_scale_p2v"] = multiscsale_voxelize(
                    multiview_data["multi_scale_xyz"], voxel_size
                )
        
        return depths, poses, valids, segments, intrinsics, multiview_data
    
    def get_color_features(self, multiview_data, images, features, shape):
        bs, v, H_padded, W_padded = shape
        idxs = None
        color_features = None
        # print("HIGH RES", self.cfg.HIGH_RES)
        if self.cfg.HIGH_RES and not self.cfg.USE_GHOST_POINTS:
            color_features = scatter_mean(
                    images.tensor.reshape(bs, v, -1, H_padded, W_padded).permute(0, 1, 3, 4, 2).flatten(1, 3),
                    multiview_data['multi_scale_p2v'][-1],
                    dim=1,
                )
            color_features_xyz = multiview_data['multi_scale_xyz'][-1]
            color_features_p2v = multiview_data['multi_scale_p2v'][-1]

            multiview_data['original_color_xyz'] = multiview_data['multi_scale_xyz'][-1]
            multiview_data['original_color_p2v'] = multiview_data['multi_scale_p2v'][-1]
            
            color_features_xyz = scatter_mean(
                color_features_xyz.flatten(1, 3),
                color_features_p2v,
                dim=1,
            )
            multiview_data['multi_scale_xyz'][-1] = color_features_xyz[:, None, None]
            multiview_data['multi_scale_p2v'][-1] = color_features_p2v

        return color_features, idxs

    def upsample_pred_masks(self, mask_pred_results, decoder_3d, batched_inputs, multiview_data, images, shape):
        bs, v, H_padded, W_padded = shape
        
        target_xyz = torch.stack(
            [batched_input['original_xyz'] for batched_input in batched_inputs]
        ).to(self.device)

        target_p2v = torch.arange(
            target_xyz.flatten(1, 3).shape[1], device=self.device)[None].repeat(target_xyz.shape[0], 1)
        source_xyz = multiview_data['multi_scale_xyz'][-1]
        source_p2v = multiview_data['multi_scale_p2v'][-1]

        mask_pred_results = mask_pred_results[:, :, source_p2v[0]]
        mask_pred_results = mask_pred_results.permute(0, 2, 1)
        source_xyz = source_xyz.flatten(1, 3)
        B, _, Q = mask_pred_results.shape
        assert B == 1, "otherwise later stuff wouldn't work"

        mask_pred_results = mask_pred_results.permute(0, 2, 1).reshape(Q, target_xyz.shape[1], target_xyz.shape[-3], target_xyz.shape[-2])

        return mask_pred_results