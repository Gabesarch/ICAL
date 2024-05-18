import numpy as np
# import quaternion
from argparse import Namespace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 

# from scipy.spatial import distance
# from sklearn.metrics.pairwise import cosine_distances
# from sklearn.metrics.pairwise import euclidean_distances

# from sklearn.decomposition import PCA
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import accuracy_score
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
# import seaborn as sn
import pandas as pd
from pandas.plotting import table
import io
# import tensorflow as tf

import copy
from PIL import Image, ImageDraw

import re

# import shapely
# from shapely.geometry import MultiPoint

# import 

# import detectron2
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.modeling import build_model
# from detectron2.modeling import build_backbone
# from detectron2.modeling.poolers import ROIPooler
# from detectron2.structures import Boxes

# from sklearn.neighbors import KernelDensity

import cv2
import scipy.ndimage as ndimage

# import utils.py
import PIL
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

import random

import ipdb
st = ipdb.set_trace

import torch
# import faiss

# # where the magic happens
# import faiss.contrib.torch_utils
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

class Utils():
    def __init__(self, W, H):  
        self.H = H
        self.W = W
        # self.K = K
        self.Softmax = nn.Softmax(dim=0)

    def revert_color(self, x):
        x = (x + 0.5) * 255.
        return x.astype(np.uint8)
    
    def format_class_name(self, name):
        if name=="TVStand":
            formatted = "television stand"
        elif name=="CounterTop":
            formatted = "countertop"
        else:
            formatted = re.sub(r"(?<=\w)([A-Z])", r" \1", name).lower()
        return formatted

    def crop_image(self, rgb, obj_instance_detection2D, max_len_thresh=5):
        # crop image with some padding
        max_len = np.max(np.array([obj_instance_detection2D[2] - obj_instance_detection2D[0], obj_instance_detection2D[3] - obj_instance_detection2D[1]]))
        max_len = max_len // 2
        pad_len = max_len // 4

        check = np.array([obj_instance_detection2D[0], obj_instance_detection2D[1], self.H - obj_instance_detection2D[2], self.W - obj_instance_detection2D[3]])

        if np.any(check < pad_len):
            pad_len = np.min(check)
            if pad_len < 0:
                pad_len = 0

        if max_len<max_len_thresh:
            # print("max len 0.. continuing")
            return None, None

        x_center = (obj_instance_detection2D[2] + obj_instance_detection2D[0]) // 2
        x_low = int(x_center-max_len-pad_len)
        if x_low < 0:
            x_low = 0
        x_high = int(x_center+max_len+pad_len) #x_low + max_len + 2*pad_len
        if x_high > self.W:
            x_high = self.W - 1

        y_center = (obj_instance_detection2D[3] + obj_instance_detection2D[1]) // 2
        y_low = int(y_center-max_len-pad_len)#-pad_len
        if y_low < 0:
            y_low = 0
        y_high = int(y_center+max_len+pad_len) #y_low + max_len + 2*pad_len
        if y_high > self.H:
            y_high = self.H - 1

        rgb_crop = rgb[x_low:x_high, y_low:y_high,:]

        bounding_box_new = np.array([x_low, y_low, x_high, y_high])
        
        return rgb_crop, bounding_box_new

    def get_center_obj_given_2Dmask(self, masks, obj_mask, len_pad=20):
        # This function outputs the indices of the object close to the center of FOV and with similar mask to provided obj_ask
        # assuming batch size of one here
        W2_low = int(self.W//2 - len_pad)
        W2_high = int(self.W//2 + len_pad)
        H2_low = int(self.H//2 - len_pad)
        H2_high = int(self.H//2 + len_pad)

        N, H, W = masks.shape

        H2, W2 = obj_mask.shape

        assert(H == H2)
        assert(W == W2)

        ind_obj = None
        sum_obj_mask = np.sum(obj_mask)
        mask_sum_thresh = 5000
        for idx in range(N):
            pred_mask_cur = masks[idx]
            pred_masks_center = pred_mask_cur[W2_low:W2_high, H2_low:H2_high]
            sum_pred_mask_cur = np.sum(pred_mask_cur)
            # print(torch.sum(pred_masks_center))
            if np.sum(pred_masks_center) > 0: # must have overlap with center of FOV
                if np.abs(sum_pred_mask_cur - sum_obj_mask) < mask_sum_thresh: # assuming similar size to previous mask
                    ind_obj = idx
                    mask_sum_thresh = np.abs(sum_pred_mask_cur - sum_obj_mask)

        return ind_obj

    def get_crop_norm(self, rgb_s, pred_box, pred_mask, object_category_names, include_classes, mask_2d_camXs, bbox_2d_camXs, ithor_to_maskrcnn, ithor_to_class, normalize):
        target_obj_bbox = np.squeeze(pred_box.detach().cpu().numpy())
        target_obj_bbox = np.array([target_obj_bbox[1], target_obj_bbox[0], target_obj_bbox[3], target_obj_bbox[2]])
        target_obj_mask = np.squeeze(pred_mask.detach().cpu().numpy())

        # get corresponding ground truth object info
        obj_category_gt_s = object_category_names
        masks_obj_gt_s = torch.stack(mask_2d_camXs).squeeze(1)
        bboxs_obj_gt_s = bbox_2d_camXs

        inds_valid_obj = [True if obj_category_gt_s[i][0] in include_classes else False for i in range(masks_obj_gt_s.shape[0])]
        obj_category_gt_s = [i for (i, v) in zip(obj_category_gt_s, inds_valid_obj) if v] 
        masks_obj_gt_s = masks_obj_gt_s[inds_valid_obj].detach().cpu().numpy()
        bboxs_obj_gt_s = torch.stack([i for (i, v) in zip(bboxs_obj_gt_s, inds_valid_obj) if v]).squeeze(1).detach().cpu().numpy()
        gt_match, pred_match, overlaps = self.compute_matches(bboxs_obj_gt_s, np.transpose(masks_obj_gt_s, (1,2,0)),
                np.reshape(target_obj_bbox, (1,4)), np.expand_dims(target_obj_mask, 2),
                iou_threshold=0.3, score_threshold=0.0)
        obj_ind_gt = int(pred_match)
        if obj_ind_gt == -1:
            print("No corresponding gt object found.. continuing..")
            return None, None

        print("GT Object found is: ", obj_category_gt_s[obj_ind_gt][0])
                
        target_obj_category_gt = obj_category_gt_s[obj_ind_gt][0]
        if target_obj_category_gt in ithor_to_class:
            # need to convert same instance labels to class (e.g. "bathtub basin" to "bathtub")
            target_obj_category_gt = ithor_to_class[target_obj_category_gt]
        target_obj_bbox_gt = bboxs_obj_gt_s[obj_ind_gt]
        target_obj_mask_gt = masks_obj_gt_s[obj_ind_gt]

        # crop and normalize 
        rgb_crop, target_obj_bbox_new = self.crop_image(rgb_s, target_obj_bbox)
        if rgb_crop is None:
            print("max len 0.. continuing")
            return None, None

        rgb_crop_PIL = Image.fromarray(rgb_crop)
        rgb_crop_norm = normalize(rgb_crop_PIL).unsqueeze(0)

        return rgb_crop_norm, target_obj_category_gt

    def get_center_obj_given_3Dmask(self, prev_obj_mask, camX_T_camX0, xyz_camX0_mask, pix_T_cams, bboxs_obj_gt_s, iou_threshold=0.7):
        # This function outputs the indices of the object which has high overap with the reprojected mask
        # assuming batch size of one here

        xyz_camX0_mask = xyz_camX0_mask[:,prev_obj_mask.flatten().astype(bool),:]
        xyz_camX_mask = utils.geom.apply_4x4(camX_T_camX0, xyz_camX0_mask)
        mask_modal_xy = utils.geom.Camera2Pixels(xyz_camX_mask, pix_T_cams)
        mask_modal_x, mask_modal_y = torch.unbind(mask_modal_xy, dim=2)
        mask_modal_x = mask_modal_x.reshape(-1).clamp(0, self.W-1)
        mask_modal_y = mask_modal_y.reshape(-1).clamp(0, self.H-1)

        # If mask not in view
        if len(mask_modal_x) == 0:
            return -1
            mask_modal = torch.zeros((self.H, self.W)).float()
        elif torch.max(mask_modal_x) == 0 or torch.max(mask_modal_y) == 0 or torch.min(mask_modal_x)==self.W-1 or torch.min(mask_modal_y)==self.H-1:
            return -1
            mask_modal = torch.zeros((self.H, self.W)).float()
        else:
            # fill in mask
            mask_modal_x = mask_modal_x.long()
            mask_modal_y = mask_modal_y.long()
            mask_modal = torch.zeros((self.H, self.W)).float()
            mask_modal[mask_modal_y, mask_modal_x] = 1

        mask_modal_x = mask_modal_x.detach().cpu().numpy()
        mask_modal_y = mask_modal_y.detach().cpu().numpy()
        reproj_bbox_obj = np.array([[np.min(mask_modal_x), np.min(mask_modal_y), np.max(mask_modal_x), np.max(mask_modal_y)]])
        match, pred_match, overlaps = self.compute_matches(bboxs_obj_gt_s, None,
                                            reproj_bbox_obj, None,
                                            iou_threshold=iou_threshold, score_threshold=0.0)
        obj_ind = int(pred_match)

        # plt.figure(1)
        # plt.clf()
        # plt.imshow(mask_modal)
        # bbox_match = bboxs_obj_gt_s[obj_ind] #np.squeeze(reproj_bbox_obj)
        # plt.plot([bbox_match[1], bbox_match[3]], [bbox_match[0], bbox_match[2]], 'x', color='black')
        # reproj_bbox_obj = np.squeeze(reproj_bbox_obj)
        # plt.plot([reproj_bbox_obj[1], reproj_bbox_obj[3]], [reproj_bbox_obj[0], reproj_bbox_obj[2]], 'x', color='black')
        # plt_name = 'visuals/plot.png'
        # plt.savefig(plt_name)
        
        return obj_ind

    def get_center_obj_intial(self, masks, len_pad=5):
        W2_low = int(self.W//2 - len_pad)
        W2_high = int(self.W//2 + len_pad)
        H2_low = int(self.H//2 - len_pad)
        H2_high = int(self.H//2 + len_pad)

        # take mask which is in center of image - if no box in center, go to next episode
        masks_center = np.where(masks[:,W2_low:W2_high, H2_low:H2_high])
        # if no object in center FOV, discard (NOTE: should instead handle this in data gen)
        if not list(masks_center[0]):
            return None

        values, counts = np.unique(masks_center[0], return_counts=True)
        ind_obj = values[np.argmax(counts)]

        return ind_obj

    def get_gt_object_from_detection(self, masks_gt, mask_detection):
        # This function gives back the gt object that best matches the detection
        # Gabe: maybe there is a better way to do this? 

        obj_ind_gt = None
        best_overlap = self.H * self.W
        for idx in range(masks_gt.shape[0]):
            obj_mask_cur = masks_gt[idx].astype(int)
            non_overlap = np.sum(np.abs(obj_mask_cur - mask_detection.astype(int)))
            overlap = np.sum(np.logical_and(obj_mask_cur, mask_detection))
            if non_overlap < best_overlap: # want most overlap with detection
                if overlap > np.sum(mask_detection * 1/4): # this just makes sure the algorithm doesnt cheat
                    best_overlap = non_overlap
                    obj_ind_gt = idx

        return obj_ind_gt
    
    def compute_matches(self, gt_boxes, gt_masks,
                        pred_boxes, pred_masks,
                        iou_threshold=0.5, score_threshold=0.0):
        """Finds matches between prediction and ground truth instances.
        Returns:
            gt_match: 1-D array. For each GT box it has the index of the matched
                    predicted box.
            pred_match: 1-D array. For each predicted box, it has the index of
                        the matched ground truth box.
            overlaps: [pred_boxes, gt_boxes] IoU overlaps.
        MODIFIED FROM MASKRCNN UTILS: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
        """
        # Trim zero padding
        # TODO: cleaner to do zero unpadding upstream
        gt_boxes = self.trim_zeros(gt_boxes)
        # gt_masks = gt_masks[..., :gt_boxes.shape[0]]
        pred_boxes = self.trim_zeros(pred_boxes)
        pred_scores = np.ones(pred_boxes.shape[0]) #pred_scores[:pred_boxes.shape[0]]
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[indices]
        # pred_class_ids = pred_class_ids[indices]
        pred_scores = pred_scores[indices]
        # pred_masks = pred_masks[..., indices]

        # Compute IoU overlaps [pred_masks, gt_masks]
        # overlaps = self.compute_overlaps_masks(pred_masks, gt_masks)
        overlaps = self.compute_overlaps(pred_boxes, gt_boxes)

        # Loop through predictions and find matching ground truth boxes
        match_count = 0
        pred_match = -1 * np.ones([pred_boxes.shape[0]])
        gt_match = -1 * np.ones([gt_boxes.shape[0]])
        best_iou = 0
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                # if gt_match[j] > -1:
                #     continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_threshold:
                    break
                elif iou > best_iou:
                    match_count += 1
                    gt_match[j] = i
                    pred_match[i] = j
                    best_iou = iou

                # # Do we have a match?
                # if pred_class_ids[i] == gt_class_ids[j]:
                    # match_count += 1
                    # gt_match[j] = i
                    # pred_match[i] = j
                #     break

        return gt_match, pred_match, overlaps
    
    def trim_zeros(self, x):
        """It's common to have tensors larger than the available data and
        pad with zeros. This function removes rows that are all zeros.
        x: [rows, columns].
        MODIFIED FROM MASKRCNN UTILS: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
        """
        assert len(x.shape) == 2
        return x[~np.all(x == 0, axis=1)]
    
    def compute_overlaps_masks(self, masks1, masks2):
        """Computes IoU overlaps between two sets of masks.
        masks1, masks2: [Height, Width, instances]
        MODIFIED FROM MASKRCNN UTILS: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
        """
        
        # If either set of masks is empty return empty result
        if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
            return np.zeros((masks1.shape[-1], masks2.shape[-1]))
        # flatten masks and compute their areas
        masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
        masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
        area1 = np.sum(masks1, axis=0)
        area2 = np.sum(masks2, axis=0)

        # intersections and union
        intersections = np.dot(masks1.T, masks2)
        union = area1[:, None] + area2[None, :] - intersections
        overlaps = intersections / union

        return overlaps

    def compute_overlaps(self,boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        For better performance, pass the largest set first and the smaller second.
        MODIFIED FROM MASKRCNN UTILS: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
        """
        # Areas of anchors and GT boxes
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
        # Each cell contains the IoU value.
        overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
        for i in range(overlaps.shape[1]):
            box2 = boxes2[i]
            overlaps[:, i] = self.compute_iou(box2, boxes1, area2[i], area1)
        return overlaps

    def compute_iou(self, box, boxes, box_area, boxes_area):
        """Calculates IoU of the given box with the array of the given boxes.
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.
        Note: the areas are passed in rather than calculated here for
        efficiency. Calculate once in the caller to avoid duplicate work.
        """
        # Calculate intersection areas
        y1 = np.maximum(box[0], boxes[:, 0])
        y2 = np.minimum(box[2], boxes[:, 2])
        x1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        union = box_area + boxes_area[:] - intersection[:]
        iou = intersection / union
        return iou

    def get_size(self, depth, obj_mask, K, round_am=6):
        # This finds the median depth and take height and width of object at median object depth
        # based on width and height of object mask and intrinsics
        # You can think of this as approximating: cross sectional area at the median depth

        depth_obj = depth[obj_mask]
        depth_obj_median = np.median(depth_obj)

        inds_mask = np.nonzero(obj_mask)

        # get pixel indices on edge of object mask
        ys1 = (np.min(inds_mask[0]) - 256/2) * -1
        xs1 = (inds_mask[1][np.argmin(inds_mask[0])] - 256/2) * -1

        ys2 = (np.max(inds_mask[0]) - 256/2) * -1
        xs2 = (inds_mask[1][np.argmax(inds_mask[0])] - 256/2) * -1

        ys3 = (inds_mask[0][np.argmin(inds_mask[1])] - 256/2) * -1
        xs3 = (np.min(inds_mask[1]) - 256/2) * -1 

        ys4 = (inds_mask[0][np.argmax(inds_mask[1])] - 256/2) * -1
        xs4 = (np.max(inds_mask[1]) - 256/2) * -1 

        xs = np.array([[xs1, xs2, xs3, xs4]])
        ys = np.array([[ys1, ys2, ys3, ys4]])

        depthmap_median = np.ones(xs.shape) * depth_obj_median

        # get 3D camX coords of those for edge points
        xys = np.vstack((xs * depthmap_median , ys * depthmap_median, -depthmap_median, np.ones(depthmap_median.shape)))
        xys = xys.reshape(4, -1)
        xy_c0 = np.matmul(np.linalg.inv(K), xys)
        xyz_obj_masked = xy_c0.T[:,:3]

        # compute x and y range
        range_x = np.round(np.max(xyz_obj_masked[:,0]) - np.min(xyz_obj_masked[:,0]), decimals=round_am)
        range_y = np.round(np.max(xyz_obj_masked[:,1]) - np.min(xyz_obj_masked[:,1]), decimals=round_am)

        # # THIS IS THE OLDER, SLOWER IMPLEMENTATION
        # depthmap_median = np.ones(depth.shape) * depth_obj_median

        # xs, ys = np.meshgrid(np.linspace(-1*256/2.,1*256/2.,256), np.linspace(1*256/2.,-1*256/2., 256))
        # depthmap_median = depthmap_median.reshape(1,256,256)
        # xs = xs.reshape(1,256,256)
        # ys = ys.reshape(1,256,256)

        # xys = np.vstack((xs * depthmap_median , ys * depthmap_median, -depthmap_median, np.ones(depthmap_median.shape)))
        # xys = xys.reshape(4, -1)
        # xy_c0 = np.matmul(np.linalg.inv(self.K), xys)
        # xyz = xy_c0.T[:,:3].reshape(256,256,3)
        # xyz_obj_masked = xyz[obj_mask]

        # range_x = np.max(xyz_obj_masked[:,0]) - np.min(xyz_obj_masked[:,0])
        # range_y = np.max(xyz_obj_masked[:,1]) - np.min(xyz_obj_masked[:,1])

        # # previously we used cross_sectional_area
        # cross_sectional_area = range_x * range_y

        size = np.array([range_x, range_y])

        return size

    def get_knn_softmax(self, obs, data_store, data_store_ids, k=10):

        # distances = np.linalg.norm(data_store-obs, axis=1)

        obs = obs.reshape(1, -1)

        distances = cosine_distances(data_store, obs).reshape(-1)

        ind_knn = list(np.argsort(distances)[:k])

        dist_knn = np.sort(distances)[:k]
        dist_knn_norm = self.Softmax(torch.from_numpy(-dist_knn)).numpy()

        match_knn_id = np.array([data_store_ids[i] for i in ind_knn])

        dist_knn_norm_add, match_knn_id_add = self.agg_softmax(dist_knn_norm, match_knn_id)

        probs = dist_knn_norm_add
        classes = match_knn_id_add
                
        return probs, classes

    def agg_softmax(self, dist_knn_norm, match_knn_id):
        # add softmax values from the same class
        match_knn_id_add = np.unique(match_knn_id)
        dist_knn_norm_add = np.zeros(match_knn_id_add.shape[0])
        for i in range(match_knn_id_add.shape[0]):
            inds_class = match_knn_id == match_knn_id_add[i]
            dist_knn_norm_add[i] = np.sum(dist_knn_norm[inds_class])
        
        return dist_knn_norm_add, match_knn_id_add

    # def kde2D(self, x, y, bandwidth=0.2, xbins=100j, ybins=100j): 
    #     """Build 2D kernel density estimate (KDE)."""

    #     # create grid of sample locations (default: 100x100)
    #     xx, yy = np.mgrid[x.min():x.max():xbins, 
    #                     y.min():y.max():ybins]

    #     xx, yy = np.mgrid[0:1.2:xbins, 
    #                     0:1.2:ybins]

    #     xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    #     xy_train  = np.vstack([y, x]).T

    #     kde_skl = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    #     kde_skl.fit(xy_train)

    #     # score_samples() returns the log-likelihood of the samples
    #     z = np.exp(kde_skl.score_samples(xy_sample))
    #     return xx, yy, np.reshape(z, xx.shape), kde_skl


    def get_rotation_to_obj(self, obj_center, pos_s):
        # YAW calculation - rotate to object
        agent_to_obj = np.squeeze(obj_center) - pos_s 
        agent_local_forward = np.array([0, 0, 1.0]) 
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj

        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

        turn_yaw = np.degrees(turn_angle)

        turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))

        return turn_yaw, turn_pitch

    # def quat_from_angle_axis(self, theta: float, axis: np.ndarray) -> np.quaternion:
    #     r"""Creates a quaternion from angle axis format

    #     :param theta: The angle to rotate about the axis by
    #     :param axis: The axis to rotate about
    #     :return: The quaternion
    #     """
    #     axis = axis.astype(np.float)
    #     axis /= np.linalg.norm(axis)
    #     return quaternion.from_rotation_vector(theta * axis)

    def safe_inverse_single(self,a):
        r, t = self.split_rt_single(a)
        t = np.reshape(t, (3,1))
        r_transpose = r.T
        inv = np.concatenate([r_transpose, -np.matmul(r_transpose, t)], 1)
        bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
        # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
        inv = np.concatenate([inv, bottom_row], 0)
        return inv
    
    def split_rt_single(self,rt):
        r = rt[:3, :3]
        t = np.reshape(rt[:3, 3], 3)
        return r, t

    def eul2rotm(self, rx, ry, rz):
        # inputs are shaped B
        # this func is copied from matlab
        # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
        #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
        #        -sy            cy*sx             cy*cx]

        sinz = np.sin(rz)
        siny = np.sin(ry)
        sinx = np.sin(rx)
        cosz = np.cos(rz)
        cosy = np.cos(ry)
        cosx = np.cos(rx)
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy

        r = np.array([[r11, r12, r13], 
                    [r21, r22, r23],
                    [r31, r32, r33]
                    ])
        return r

    def rotm2eul(self,r):
        # r is Bx3x3, or Bx4x4
        r00 = r[0,0]
        r10 = r[1,0]
        r11 = r[1,1]
        r12 = r[1,2]
        r20 = r[2,0]
        r21 = r[2,1]
        r22 = r[2,2]
        
        sy = np.sqrt(r00*r00 + r10*r10)
        
        cond = (sy > 1e-6)
        rx = np.where(cond, np.arctan2(r21, r22), np.arctan2(-r12, r11))
        ry = np.where(cond, np.arctan2(-r20, sy), np.arctan2(-r20, sy))
        rz = np.where(cond, np.arctan2(r10, r00), np.zeros_like(r20))

        return rx, ry, rz

    def position_to_tuple(self,position):
        return (position["x"], position["y"], position["z"])

    def get_habitat_pix_T_camX(self, fov):
        hfov = float(fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        return pix_T_camX


    def get_agent_map_data(self,c):
        c.step({"action": "ToggleMapView"})
        cam_position = c.last_event.metadata["cameraPosition"]
        cam_orth_size = c.last_event.metadata["cameraOrthSize"]
        pos_translator = ThorPositionTo2DFrameTranslator(
            c.last_event.frame.shape, self.position_to_tuple(cam_position), cam_orth_size
        )
        to_return = {
            "frame": c.last_event.frame,
            "cam_position": cam_position,
            "cam_orth_size": cam_orth_size,
            "pos_translator": pos_translator,
        }
        # plt.figure(figsize=(8,8))
        # plt.imshow(c.last_event.instance_masks['Television|-02.36|+01.21|+06.24'])
        # plt.imshow(c.last_event.instance_segmentation_frame)
        # plt.savefig('images/test.png')
        # st()
        c.step({"action": "ToggleMapView"})
        
        return to_return

    def add_agent_view_triangle(
        self,position, rotation, frame, pos_translator, scale=1.0, opacity=0.7
    ):
        p0 = np.array((position[0], position[2]))
        p1 = copy.copy(p0)
        p2 = copy.copy(p0)

        theta = -2 * math.pi * (rotation / 360.0)
        rotation_mat = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
        offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

        p1 += np.matmul(rotation_mat, offset1)
        p2 += np.matmul(rotation_mat, offset2)

        img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
        img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

        opacity = int(round(255 * opacity))  # Define transparency for the triangle.
        points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
        draw = ImageDraw.Draw(img2)
        draw.polygon(points, fill=(255, 255, 255, opacity))

        img = Image.alpha_composite(img1, img2)
        return np.array(img.convert("RGB"))
    
    def get_overhead_map(self, controller, return_pos_translator=False):
        t = self.get_agent_map_data(controller)
        new_frame = self.add_agent_view_triangle(
            self.position_to_tuple(controller.last_event.metadata["agent"]["position"]),
            controller.last_event.metadata["agent"]["rotation"]["y"],
            t["frame"],
            t["pos_translator"],
            scale=0.1,
            opacity=0.0,
        )
        if not return_pos_translator:
            return new_frame
        else:
            return new_frame, t["pos_translator"]

class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )
    
    def image_to_pos(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position
        
        z = 1.0 - (z / self.frame_shape[0])
        x = x / self.frame_shape[1]
        aa = (np.array((x, z)) * self.span) + self.lower_left

        return aa


class Relations():
    def __init__(self, W, H):  
        self.H = H
        self.W = W
        # self.K = K
        self.Softmax = nn.Softmax(dim=0)

    @staticmethod
    def box2points(box):
        """Convert box min/max coordinates to vertices (8x3)."""
        x_min, y_min, z_min, x_max, y_max, z_max = box
        return np.array([
            [x_min, y_min, z_min], [x_min, y_max, z_min],
            [x_max, y_min, z_min], [x_max, y_max, z_min],
            [x_min, y_min, z_max], [x_min, y_max, z_max],
            [x_max, y_min, z_max], [x_max, y_max, z_max]
        ])

    @staticmethod
    def _compute_dist(points0, points1):
        """Compute minimum distance between two sets of points."""
        dists = ((points0[:, None, :] - points1[None, :, :]) ** 2).sum(2)
        return dists.min()

    def _intersect(self, box_a, box_b):
        return self._intersection_vol(box_a, box_b) > 0

    @staticmethod
    def _intersection_vol(box_a, box_b):
        xA = max(box_a[0] - box_a[3] / 2, box_b[0] - box_b[3] / 2)
        yA = max(box_a[1] - box_a[4] / 2, box_b[1] - box_b[4] / 2)
        zA = max(box_a[2] - box_a[5] / 2, box_b[2] - box_b[5] / 2)
        xB = min(box_a[0] + box_a[3] / 2, box_b[0] + box_b[3] / 2)
        yB = min(box_a[1] + box_a[4] / 2, box_b[1] + box_b[4] / 2)
        zB = min(box_a[2] + box_a[5] / 2, box_b[2] + box_b[5] / 2)
        return max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)

    def _inside(self, box_a, box_b):
        volume_a = box_a[3] * box_a[4] * box_a[5]
        return np.isclose(self._intersection_vol(box_a, box_b), volume_a)

    @staticmethod
    def iou_2d(box0, box1):
        """Compute 2d IoU for two boxes in coordinate format."""
        box_a = np.array(box0)[(0, 1, 3, 4), ]
        box_b = np.array(box1)[(0, 1, 3, 4), ]
        # Intersection
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        # Areas
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        # Return IoU and area ratios
        return (
            inter_area / (box_a_area + box_b_area - inter_area),  # iou
            [inter_area / box_a_area, inter_area / box_b_area],
            [box_a_area / box_b_area, box_b_area / box_a_area]
        )

    @staticmethod
    def volume(box):
        """Compute of box in coordinate format (min, max)."""
        return (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])

    @staticmethod
    def _to_min_max(box):
        return np.concatenate((
            box[:, :3] - box[:, 3:] / 2, box[:, :3] + box[:, 3:] / 2
        ), 1)

    @staticmethod
    def _same_x_range(box, ref_box):
        return (
            min(box[3], ref_box[3]) - max(box[0], ref_box[0])
            > 0.8 * min(box[3] - ref_box[0], box[3] - ref_box[0])
        )

    @staticmethod
    def _same_y_range(box, ref_box):
        return (
            min(box[4], ref_box[4]) - max(box[1], ref_box[1])
            > 0.8 * min(box[4] - ref_box[1], box[4] - ref_box[1])
        )

    @staticmethod
    def _same_z_range(box, ref_box):
        return (
            min(box[5], ref_box[5]) - max(box[2], ref_box[2])
            > 0.3 * (box[5] - box[2])
        )

    def _is_left(self, box, ref_box):
        return (
            box[3] < ref_box[0]  # x_max < x_ref_min
            and self._same_y_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_right(self, box, ref_box):
        return (
            box[0] > ref_box[3]  # x_min > x_ref_max
            and self._same_y_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_front(self, box, ref_box):
        return (
            box[4] < ref_box[1]  # y_max < y_ref_min
            and self._same_x_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )

    def _is_behind(self, box, ref_box):
        return (
            box[1] > ref_box[4]  # y_min > y_ref_max
            and self._same_x_range(box, ref_box)
            and self._same_z_range(box, ref_box)
        )
    
    # def _is_next_to(self, box, ref_box):
    #     center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2])
    #     center_ref_box0 = np.array([(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2, (ref_box[5]+ref_box[2])/2])
    #     squared_dist = np.sum((center_box-center_ref_box0)**2, axis=0)
    #     dist = np.sqrt(squared_dist)

    #     return (
    #         dist < 2.0  
    #         and self._same_z_range(box, ref_box)
    #     )

    def _is_between(self, box, ref_box0, ref_box1):
        # # Get the convex hull of all points of the two anchors
        # convex_hull = MultiPoint(
        #     tuple(map(tuple, self.box2points(ref_box0)[:4, :2]))
        #     + tuple(map(tuple, self.box2points(ref_box1)[:4, :2]))
        # ).convex_hull
        # # Get box as polygons
        # polygon_t = MultiPoint(
        #     tuple(map(tuple, self.box2points(box)[:4, :2]))
        # ).convex_hull
        # # Candidate should fall in the convex_hull polygon
        # return (
        #     convex_hull.intersection(polygon_t).area / polygon_t.area > 0.51
        #     and self._same_z_range(box, ref_box0)
        #     and self._same_z_range(box, ref_box1)
        # )

        # Find distance to of the center of the box to line between centers of ref_box0 and ref_box1

        # get centers of each box
        center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2])
        center_ref_box0 = np.array([(ref_box0[3]+ref_box0[0])/2, (ref_box0[4]+ref_box0[1])/2, (ref_box0[5]+ref_box0[2])/2])
        center_ref_box1 = np.array([(ref_box1[3]+ref_box1[0])/2, (ref_box1[4]+ref_box1[1])/2, (ref_box1[5]+ref_box1[2])/2])

        dist_to_0 = np.sqrt(np.sum((center_box - center_ref_box0)**2, axis=0))
        dist_to_1 = np.sqrt(np.sum((center_box - center_ref_box1)**2, axis=0))

        # # draw line between center_ref_box0 and center_ref_box1, and find distance of center_box to that line
        # p = center_ref_box0
        # q = center_ref_box1
        # r = center_box
        # x = p-q
        # t = np.dot(r-q, x)/np.dot(x, x)
        # dist = np.linalg.norm(t*(p-q)+q-r)
        # print(dist)

        a = center_ref_box0
        b = center_ref_box1
        p = center_box

        # # normalized tangent vector
        # d = np.divide(b - a, np.linalg.norm(b - a))

        # # signed parallel distance components
        # s = np.dot(a - p, d)
        # t = np.dot(p - b, d)

        # # clamped parallel distance
        # h = np.maximum.reduce([s, t, 0])

        # # perpendicular distance component
        # c = np.cross(p - a, d)

        # dist = np.hypot(h, np.linalg.norm(c))
        # print(dist)

        r = b - a
        a2 = a - p
        
        min_t = np.clip(-a2.dot(r) / (r.dot(r)), 0, 1)
        d = a2 + min_t * r
        dist = np.sqrt(d.dot(d))
        # print(dist)

        # distance to each center
        squared_dist = np.sum((p-a)**2, axis=0)
        dist3 = np.sqrt(squared_dist)
        # print(dist3)

        squared_dist = np.sum((p-b)**2, axis=0)
        dist4 = np.sqrt(squared_dist)
        # print(dist4)

        closer_to_point_than_segment = np.logical_or(np.isclose(dist, dist3), np.isclose(dist, dist4))

        # # check if not to the right or left of both ref boxes
        # is_right_0 = self._is_right(box, ref_box0)
        # is_right_1 = self._is_right(box, ref_box1)
        # is_left_0 = self._is_left(box, ref_box0)
        # is_left_1 = self._is_left(box, ref_box1)
        # both_right = is_right_0 and is_right_1
        # both_left = is_left_0 and is_left_1
        # # sign_diff_x = np.sign(center_box[0] - center_ref_box0[0]) != np.sign(center_box[0] - center_ref_box1[0])
        # # sign_diff_y = np.sign(center_box[1] - center_ref_box0[1]) != np.sign(center_box[1] - center_ref_box1[1])
        # print(not both_right, not both_left)

        return dist, (
            dist < 0.1 #0.1 
            and not closer_to_point_than_segment
            and self._same_z_range(box, ref_box0)
            and self._same_z_range(box, ref_box1)
            and not self._is_above(box, ref_box0)
            and not self._is_below(box, ref_box0)
            and not self._is_above(box, ref_box1)
            and not self._is_below(box, ref_box1)
            and max(dist_to_0, dist_to_1) < 3
        )


    def _is_similar_dist_to(self, box, ref_box0, ref_box1):
        center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2])
        center_ref_box0 = np.array([(ref_box0[3]+ref_box0[0])/2, (ref_box0[4]+ref_box0[1])/2, (ref_box0[5]+ref_box0[2])/2])
        center_ref_box1 = np.array([(ref_box1[3]+ref_box1[0])/2, (ref_box1[4]+ref_box1[1])/2, (ref_box1[5]+ref_box1[2])/2])

        dist_to_0 = np.sqrt(np.sum((center_box - center_ref_box0)**2, axis=0))
        dist_to_1 = np.sqrt(np.sum((center_box - center_ref_box1)**2, axis=0))

        return np.abs(dist_to_0 - dist_to_1), np.abs(dist_to_0 - dist_to_1) < min(0.1, 0.1 * min(dist_to_0, dist_to_1)) and max(dist_to_0, dist_to_1) < 2

    def _is_very_close_to(self, box, ref_box):
        '''
        this is a simpler version of _is_next_to.
        it doesn't take into account the intersection ratio. Just finds the closest distance between the points in the box
        this is solely for the superstructure: object type B's around object type A.
        '''
        corner_idx = np.array([[0,1,2],[0,1,5],[0,4,2],[0,4,5],[3,1,2],[3,1,5],[3,4,2],[3,4,5]])
        box_corners = box[corner_idx]
        box_center = np.array([[(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2]])
        box_points = np.concatenate([box_corners, box_center], axis=0)
        ref_box_corners = ref_box[corner_idx]
        ref_box_center = np.array([[(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2, (ref_box[5]+ref_box[2])/2]])
        ref_box_points = np.concatenate([ref_box_corners, ref_box_center], axis=0)
        squared_dists = np.sum((np.expand_dims(box_points, axis=0) - np.expand_dims(ref_box_points, axis=1))**2, axis=-1)
        min_dist = np.amin(np.sqrt(squared_dists))
        return dist < 1.0


    def _is_next_to(self, box, ref_box, ref_is_wall):
        corner_idx = np.array([[0,1,2],[0,1,5],[0,4,2],[0,4,5],[3,1,2],[3,1,5],[3,4,2],[3,4,5]])
        box_corners = box[corner_idx]
        box_center = np.array([[(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2]])
        box_points = np.concatenate([box_corners, box_center], axis=0)
        ref_box_corners = ref_box[corner_idx]
        ref_box_center = np.array([[(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2, (ref_box[5]+ref_box[2])/2]])
        ref_box_points = np.concatenate([ref_box_corners, ref_box_center], axis=0)
        if not ref_is_wall:
            squared_dists = np.sum((np.expand_dims(box_points, axis=0) - np.expand_dims(ref_box_points, axis=1))**2, axis=-1)
            dist = np.amin(np.sqrt(squared_dists))

            iou, intersect_ratios, area_ratios = self.iou_2d(box, ref_box)
            int2box_ratio, int2ref_ratio = intersect_ratios

            return (
                dist < 0.5
                and int2box_ratio < 0.7
                and int2ref_ratio < 0.7
                # and self._is_equal_height(box, ref_box)
                # and not self._is_above(box, ref_box)
                # and not self._is_below(box, ref_box)
            )
        else:
            dist = np.amin(np.abs(box_points[:,0] - ref_box_center[0,0])) if ref_box[3] - ref_box[0] < ref_box[4] - ref_box[1] else np.amin(np.abs(box_points[:,1] - ref_box_center[0,1]))
            return dist < 0.3

    def _is_supported_by(self, box, ref_box, ground_plane_h=None):

        if ground_plane_h is not None:
            box_bottom_ref_top_dist = box[2] - ground_plane_h
            # print(box_bottom_ref_top_dist)
        else:
            box_bottom_ref_top_dist = box[2] - ref_box[5]
        iou, intersect_ratios, area_ratios = self.iou_2d(box, ref_box)
        int2box_ratio, _ = intersect_ratios
        box2ref_ratio, _ = area_ratios

        return (
            int2box_ratio > 0.1  #0.3  # xy intersection
            and abs(box_bottom_ref_top_dist) <= 0.2  # close to surface
            and box2ref_ratio < 1.5  # supporter is usually larger
        )

    # def _is_supporting(self, box, ref_box):
    #     ref_bottom_cox_top_dist = ref_box[2] - box[5]
    #     _, intersect_ratios, area_ratios = self.iou_2d(box, ref_box)
    #     _, int2ref_ratio = intersect_ratios
    #     _, ref2box_ratio = area_ratios
    #     # print(int2ref_ratio, abs(ref_bottom_cox_top_dist), ref2box_ratio)
    #     return (
    #         int2ref_ratio > 0.1 #0.3  # xy intersection
    #         and abs(ref_bottom_cox_top_dist) <= 0.2 #0.01  # close to surface
    #         and ref2box_ratio < 1.5  # supporter is usually larger
    #     )

    def _is_above(self, box, ref_box):
        box_bottom_ref_top_dist = box[2] - ref_box[5]
        _, intersect_ratios, _ = self.iou_2d(box, ref_box)
        int2box_ratio, int2ref_ratio = intersect_ratios
        return (
            box_bottom_ref_top_dist > 0.03  # should be above
            and max(int2box_ratio, int2ref_ratio) > 0.2  # xy intersection
        )

    def _is_below(self, box, ref_box):
        ref_bottom_cox_top_dist = ref_box[2] - box[5]
        _, intersect_ratios, _ = self.iou_2d(box, ref_box)
        int2box_ratio, int2ref_ratio = intersect_ratios
        return (
            ref_bottom_cox_top_dist > 0.03  # should be above
            and max(int2box_ratio, int2ref_ratio) > 0.2  # xy intersection
        )

    def _is_aligned(self, ori, ref_ori, box, ref_box):

        if ori is None or ref_ori is None:
            return False

        ori_dist = np.linalg.norm(ori[:2] - ref_ori[:2])
        # print(ori_dist, ori, ref_ori)

        center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2])
        center_ref_box0 = np.array([(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2])
        squared_dist = np.sum((center_box-center_ref_box0)**2, axis=0)
        dist = np.sqrt(squared_dist)        

        return (
            np.abs(ori_dist) < 0.2
            and dist < 1.5
        )

    def _is_facing(self, overhead_box, ref_facing, overhead_box_ref, box, ref_box):
        # algorithm from https://gamedev.stackexchange.com/questions/109513/how-to-find-if-an-object-is-facing-another-object-given-position-and-direction-a

        if ref_facing is None or overhead_box is None:
            return False

        center = np.array([(overhead_box[0] + overhead_box[2])//2, (overhead_box[1] + overhead_box[3])//2])
        center_ref = np.array([(overhead_box_ref[0] + overhead_box_ref[2])//2, (overhead_box_ref[1] + overhead_box_ref[3])//2])

        ref_facing_unit_vec = np.array([np.cos(ref_facing-np.pi/2), np.sin(ref_facing-np.pi/2)])
        facing = -np.dot((center_ref-center)/np.linalg.norm(center_ref-center), ref_facing_unit_vec/np.linalg.norm(ref_facing_unit_vec))

        center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2])
        center_ref_box0 = np.array([(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2])
        squared_dist = np.sum((center_box-center_ref_box0)**2, axis=0)
        dist = np.sqrt(squared_dist)    

        # print(dist)
        # print(facing)

        return (
            facing > 0.9 # 0.9
            and dist < 7.0 # distance in meters
        )


    def get_facing_dir(self, depth_masked, mask_binary, line):
        ''' 
        TODO: Optimize the algorithm here
        Obtains the facing direciton with respect to the overhead map.
        For reference: 0 degrees is facing directly "up"/north, 90 degrees is eat, 180 degrees south, etc. 
        looks at the depth distribution and identifies the "back" of the object by which direction has a larger median depth (wrt a fitted line)
        Inputs: 
        - depth_masked: depth map of overhead view masked for the object
        - mask_binary: binary mask of the object overhead view
        - line: 4x1 array: with parameters of line from cv2.fitLine
        '''

        vx,vy,x,y = list(line)
        
        x_mask, y_mask = np.where(mask_binary)

        for q in [1,2]: # also consider perpendicular line

            # here we need to check the original line and the line perpendicular to that
            if q == 1:
                    # get two points on the fitted line
                lx1 = x + vx*256
                ly1 = y + vy*256
                lx2 = x + vx*0
                ly2 = y + vy*0
            elif q==2:
                lx1 = x + vy*256
                ly1 = y - vx*256
                lx2 = x + vy*0
                ly2 = y - vx*0

            depths_above = []
            depths_below = []
            xs_above = []
            ys_above = []
            xs_below = []
            ys_below = []
            for i in range(depth_masked.shape[0]):

                # get image coords of mask 
                yA = x_mask[i]
                xA = y_mask[i]

                # Check if the point falls above or below the fitted line
                v1 = (lx2-lx1, ly2-ly1)   # Vector 1
                v2 = (lx2-xA, ly2-yA)   # Vector 1
                xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product
                if xp > 0: # fell 'above' the line (may not be in the literal sense here)
                    depths_above.append(depth_masked[i])
                    xs_above.append(xA)
                    ys_above.append(yA)
                    # print('on one side')
                elif xp < 0: # fell 'below' the line
                    depths_below.append(depth_masked[i])
                    xs_below.append(xA)
                    ys_below.append(yA)
                    # print('on the other')
                # else:
                #     print('on the same line!')
            
            if q==1:
                # get median depth of points that fall on either side of line
                d_ab_median1 = np.median(np.array(depths_above))
                d_be_median1 = np.median(np.array(depths_below))

                # want largest difference (back of chair vs front)
                median_diff1 = np.abs(d_ab_median1 - d_be_median1)
            elif q==2:
                # get median depth of points that fall on either side of line
                d_ab_median2 = np.median(np.array(depths_above))
                d_be_median2 = np.median(np.array(depths_below))

                # want largest difference (back of chair vs front)
                median_diff2 = np.abs(d_ab_median2 - d_be_median2)
            
                if median_diff2 > median_diff1: # we want bigger difference in median depth to detect the back vs front
                    line_consider = 'perpendicular'

                    # find distance to y axis
                    y_axis      = np.array([0, 1])    # unit vector in the same direction as the x axis
                    your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
                    dot_product = np.dot(y_axis, your_line)
                    angle_2_y   = np.arccos(dot_product)

                    # if angle_2_y > np.pi/2:
                    #     angle_2_y = np.pi - angle_2_y

                    # if angle_2_y > np.pi/4:
                    #     angle_2_y = np.pi/2 - angle_2_y

                    # get sign of slope
                    sign_slope = np.sign(vx/vy)

                    if d_ab_median2 < d_be_median2: # shorter distance to camera means higher depth of part from floor
                        if sign_slope > 0:
                            facing_orientation = np.pi - angle_2_y
                        else:
                            facing_orientation = 2*np.pi - angle_2_y
                        # side_consider = 'above'
                    else:
                        if sign_slope > 0:
                            facing_orientation = np.pi - angle_2_y
                            # facing_orientation = np.pi - angle_2_y
                        else:
                            facing_orientation = 2*np.pi - angle_2_y
                        # side_consider = 'below'
                else:

                    # find angle to y axis
                    y_axis      = np.array([0, 1])    # unit vector in the same direction as the x axis
                    your_line   = np.array([-vy, vx])  # unit vector in the same direction as your line
                    dot_product = np.dot(y_axis, your_line)
                    angle_2_y   = np.arccos(dot_product)

                    # if angle_2_y > np.pi/2:
                    #     angle_2_y = np.pi - angle_2_y

                    # if angle_2_y > np.pi/4:
                    #     angle_2_y = np.pi/2 - angle_2_y

                    # get sign of slope
                    sign_slope = np.sign(-vy/vx)

                    if d_ab_median1 < d_be_median1: # shorter distance to camera means higher depth of part from floor
                        if sign_slope > 0:
                            facing_orientation = 2*np.pi - angle_2_y # f
                        else:
                            facing_orientation = 2*np.pi - angle_2_y # f
                        # side_consider = 'above'
                    else:
                        if sign_slope > 0:
                            facing_orientation = np.pi - angle_2_y
                            # facing_orientation = np.pi - angle_2_y
                        else:
                            facing_orientation = 2*np.pi - angle_2_y # f

        return facing_orientation

    @staticmethod
    def _is_higher(box, ref_box):
        return box[2] - ref_box[5] > 0.1 #0.03

    @staticmethod
    def _is_lower(box, ref_box):
        return ref_box[2] - box[5] > 0.1 #0.03

    def _is_equal_height(self, box, ref_box):
        return min(box[5], ref_box[5]) - max(box[2], ref_box[2]) > 0.3 * (max(box[5], ref_box[5]) - min(box[2], ref_box[2]))

    def _is_larger(self, box, ref_box):
        return self.volume(box) > 1.1 * self.volume(ref_box)

    def _is_smaller(self, box, ref_box):
        return self.volume(ref_box) > 1.1 * self.volume(box)

    def _is_equal_size(self, box, ref_box):
        return (
            not self._is_larger(box, ref_box)
            and not self._is_smaller(box, ref_box)
            and 0.9 < (box[3] - box[0]) / (ref_box[3] - ref_box[0]) < 1.1
            and 0.9 < (box[4] - box[1]) / (ref_box[4] - ref_box[1]) < 1.1
            and 0.9 < (box[5] - box[2]) / (ref_box[5] - ref_box[2]) < 1.1
        )

    def _get_closest(self, boxes, ref_box):
        dists = np.array([
            self._compute_dist(self.box2points(box), self.box2points(ref_box))
            for box in boxes
        ])
        # dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
        return dists.argmin()

    def _get_furthest(self, boxes, ref_box):
        dists = np.array([
            self._compute_dist(self.box2points(box), self.box2points(ref_box))
            for box in boxes
        ])
        # dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
        return dists.argmax()

    def _get_largest(self, boxes, ref_box=None):
        return np.array([self.volume(box) for box in boxes]).argmax()

    def _get_smallest(self, boxes, ref_box=None):
        return np.array([self.volume(box) for box in boxes]).argmin()



class Relations_CenterOnly():
    def __init__(self, W, H):  
        self.H = H
        self.W = W
        # self.K = K
        self.Softmax = nn.Softmax(dim=0)

    def _closest(self, ref_centroid, centroids):
        dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
        return dists.argmin()

    def _farthest(self, ref_centroid, centroids):
        dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
        return dists.argmax()

    def _is_above(self, ref_center, center):
        y_diff = ref_center[1] - center[1]
        is_relation = y_diff > 0.03
        return is_relation

    def _is_below(self, ref_center, center):
        y_diff = center[1] - ref_center[1]
        is_relation = y_diff > 0.03
        return is_relation
        
    def _is_similar_height(self, ref_center, center):
        y_diff_abs = np.abs(ref_center[1] - center[1])
        is_relation = y_diff_abs <= 0.06
        return is_relation

    def _is_supported_by(self, ref_centroid, centroids, ground_plane_h=None):

        # first check supported by floor
        floor_dist = ref_centroid[1] - ground_plane_h
        if floor_dist<0.1:
            return -1 # floor

        # must be below 
        obj_below = (centroids[:,1] - ref_centroid[1]) < 0.03
        dists = np.sqrt(np.sum((centroids[obj_below] - np.expand_dims(ref_centroid, axis=0))**2, axis=1))

        if len(dists)==0:
            return -2

        argmin_dist = dists.argmin()
        argmin_ = np.arange(centroids.shape[0])[obj_below][argmin_dist]

        return argmin_

    def _is_next_to(self, ref_center, center):
        dist = np.sqrt(np.sum((center - ref_center)**2))
        is_relation = dist < 1.2
        return is_relation

    

