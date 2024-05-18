import cv2
import numpy as np
import torch
import torch.nn.functional as F 
from PIL import Image
from arguments import args
# from nets.REPLAY.util import box_ops
import warnings

from utils.Detection_Metrics.pascalvoc_nofiles import get_map, ValidateFormats, ValidateCoordinatesTypes, add_bounding_box
import glob
from utils.Detection_Metrics.lib.BoundingBox import BoundingBox
from utils.Detection_Metrics.lib.BoundingBoxes import BoundingBoxes
from utils.Detection_Metrics.lib.Evaluator import *
from utils.Detection_Metrics.lib.utils_pascal import BBFormat

import ipdb
st = ipdb.set_trace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize_boxes(img_, pred_boxes, pred_labels, score_img, iou_img, precision_img, recall_img,vis_name, obj_cur, gt_boxes, gt_labels, id_to_name, summ_writer, class_agnostic=False):
    confidence=0.5; rect_th=1; text_size=0.5; text_th=1
    img = img_.copy()
    for i in range(len(pred_boxes)):
        # rgb_mask = self.get_coloured_mask(pred_masks[i])
        if class_agnostic: 
            pred_class_name = 'Object'
        else:
            pred_class_name = id_to_name[int(pred_labels[i])]

        if pred_class_name != obj_cur:
            continue

        # pred_score = pred_scores[i]
        # alpha = 0.7
        # beta = (1.0 - alpha)
        # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
        # where_mask = rgb_mask>0
        # img[where_mask] = rgb_mask[where_mask]
        cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),(0, 255, 0), rect_th)
        cv2.putText(img,pred_class_name, (int(pred_boxes[i][0:1]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'score: ' + str(float(score_img)), (int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'best iou: ' + str(iou_img), (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'precision: ' + str(precision_img), (int(20), int(60)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'recall: ' + str(recall_img), (int(20), int(80)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    img_torch1 = torch.from_numpy(img).cuda().permute(2,0,1) / 255. - 0.5
    img_torch1 = F.interpolate(img_torch1.unsqueeze(0), scale_factor=0.5, mode='bilinear')

    img = img_.copy()
    for i in range(len(pred_boxes)):
        # rgb_mask = self.get_coloured_mask(pred_masks[i])
        if class_agnostic: 
            pred_class_name = 'Object'
        else:
            pred_class_name = id_to_name[int(pred_labels[i])]

        # if pred_class_name != obj_cur:
        #     continue

        # pred_score = pred_scores[i]
        # alpha = 0.7
        # beta = (1.0 - alpha)
        # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
        # where_mask = rgb_mask>0
        # img[where_mask] = rgb_mask[where_mask]
        cv2.rectangle(img, (int(pred_boxes[i][0]), int(pred_boxes[i][1])), (int(pred_boxes[i][2]), int(pred_boxes[i][3])),(0, 255, 0), rect_th)
        cv2.putText(img,pred_class_name, (int(pred_boxes[i][0:1]), int(pred_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'score: ' + str(float(score_img)), (int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'best iou: ' + str(iou_img), (int(20), int(40)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'precision: ' + str(precision_img), (int(20), int(60)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    cv2.putText(img,'recall: ' + str(recall_img), (int(20), int(80)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    img_torch2 = torch.from_numpy(img).cuda().permute(2,0,1) / 255. - 0.5
    img_torch2 = F.interpolate(img_torch2.unsqueeze(0), scale_factor=0.5, mode='bilinear')

    img = img_.copy()
    for i in range(len(gt_boxes)):
        # rgb_mask = self.get_coloured_mask(gt_masks[i])
        if class_agnostic: 
            gt_class_name = 'Object'
        else:
            gt_class_name = id_to_name[int(gt_labels[i])]

        if gt_class_name != obj_cur:
            continue

        # alpha = 0.7
        # beta = (1.0 - alpha)
        # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
        # where_mask = rgb_mask>0
        # img[where_mask] = rgb_mask[where_mask]
        cv2.rectangle(img, (int(gt_boxes[i][0]), int(gt_boxes[i][1])), (int(gt_boxes[i][2]), int(gt_boxes[i][3])),color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,gt_class_name, (int(gt_boxes[i][0:1]), int(gt_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    img_torch3 = torch.from_numpy(img).cuda().permute(2,0,1) / 255. - 0.5
    img_torch3 = F.interpolate(img_torch3.unsqueeze(0), scale_factor=0.5, mode='bilinear')

    

    # img = img_.copy()
    # for i in range(len(gt_boxes)):
    #     # rgb_mask = self.get_coloured_mask(gt_masks[i])
    #     if class_agnostic: 
    #         gt_class_name = 'Object'
    #     else:
    #         gt_class_name = id_to_name[int(gt_labels[i])]

    #     # if gt_class_name != obj_cur:
    #     #     continue

    #     # alpha = 0.7
    #     # beta = (1.0 - alpha)
    #     # img = cv2.addWeighted(img.astype(np.int32), 1.0, rgb_mask.astype(np.int32), 0.5, 0.0)
    #     # where_mask = rgb_mask>0
    #     # img[where_mask] = rgb_mask[where_mask]
    #     cv2.rectangle(img, (int(gt_boxes[i][0]), int(gt_boxes[i][1])), (int(gt_boxes[i][2]), int(gt_boxes[i][3])),color=(0, 255, 0), thickness=rect_th)
    #     cv2.putText(img,gt_class_name, (int(gt_boxes[i][0:1]), int(gt_boxes[i][1:2])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    # img_torch3 = torch.from_numpy(img).cuda().permute(2,0,1).unsqueeze(0) / 255. - 0.5

    img_torch = torch.cat([img_torch1, img_torch2, img_torch3], dim=3)
    summ_writer.summ_rgb(vis_name, img_torch)

@torch.no_grad()
def check_for_detections_local_policy(
    outputs,
    W, H, 
    score_labels_name, 
    score_boxes_name, 
    score_threshold_ddetr=0.0, 
    do_nms=True, 
    return_features=False, 
    target_object=None, 
    target_object_score_threshold=None, 
    solq=False, 
    return_masks=False, 
    nms_threshold=0.2, 
    id_to_mapped_id=None,
    return_policy_output=False,
    idx=0,
    ):
    '''
    rgb: single rgb image
    NOTE: currently this only handles a single RGB
    '''
    
    # rgb_PIL = Image.fromarray(rgb)
    # rgb_norm = rgb.astype(np.float32) * 1./255
    # rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).cuda()
    
    # # self.ddetr.cuda()
    # if solq:
    #     outputs = ddetr(rgb_torch.unsqueeze(0), do_loss=False, return_features=return_features)
    # else:
    #     outputs = ddetr([rgb_torch], do_loss=False, return_features=return_features)
    # self.ddetr.cpu()
    
    out = outputs['outputs']
    if return_features:
        features = out['features']
    else:
        features = None
    postprocessors = outputs['postprocessors']

    # if return_features:
    #     predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=True)
    #     features = features[0]
    # else:
    if return_features:
        predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
        features = features[idx]
        # if hyp.do_segmentation:
        #     predictions = postprocessors['segm'](predictions, out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())  
    else:
        predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
    pred_boxes = predictions[score_labels_name][idx]['boxes']
    pred_labels = predictions[score_labels_name][idx]['labels']
    pred_scores_boxes = predictions[score_boxes_name][idx]['scores']
    pred_scores_labels = predictions[score_labels_name][idx]['scores']
    pred_interm_labels = predictions[score_labels_name][idx]['labels_interm']
    pred_interm_scores = predictions[score_labels_name][idx]['scores_interm']
    pred_action = predictions[score_labels_name][idx]['labels_action']
    if return_masks:
        pred_masks = predictions[score_labels_name][idx]['masks']
        
        
    if pred_boxes.shape[0]>1:
        
        if do_nms:
            keep, count = nms(pred_boxes, pred_scores_boxes, nms_threshold, top_k=100)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            # pred_scores = pred_scores[keep]
            pred_scores_boxes = pred_scores_boxes[keep]
            pred_scores_labels = pred_scores_labels[keep]
            if return_masks:
                pred_masks = pred_masks[keep]
            if return_features:
                features = features[keep]
            pred_interm_labels = pred_interm_labels[keep]
            pred_interm_scores = pred_interm_scores[keep]

        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
        pred_scores_labels = pred_scores_labels.cpu().numpy() 
        if return_masks:
            pred_masks = pred_masks.squeeze(1).cpu().numpy() 
        pred_interm_labels = pred_interm_labels.cpu().numpy() 
        pred_interm_scores = pred_interm_scores.cpu().numpy() 

        # print(pred_scores_labels)

        # pred_boxes = pred_boxes.cpu().numpy()
        # pred_labels = pred_labels.cpu().numpy()
        # pred_scores = pred_scores.cpu().numpy()

        # above score threshold
        if target_object is not None:
            if type(target_object)==int:
                keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            elif type(target_object)==list:
                # keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
                keep_target = []
                for lab_i in range(len(pred_labels)):
                    check1 = pred_scores_labels[lab_i]>target_object_score_threshold
                    check2 = pred_labels[lab_i] in target_object
                    # check if above score and label in target_object list
                    if check1 and check2:
                        keep_target.append(True)
                    else:
                        keep_target.append(False)
                keep_target = np.array(keep_target)
            else:
                assert(False) # target_object should be int or list of ints
            keep = keep_target #np.logical_and(keep_target, keep)
        else:
            keep = pred_scores_labels>score_threshold_ddetr
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores_labels = pred_scores_labels[keep]
        if return_features:
            features = features[keep]
        if return_masks:
            pred_masks = pred_masks[keep]
        pred_interm_labels = pred_interm_labels[keep]
        pred_interm_scores = pred_interm_scores[keep]

        if id_to_mapped_id is not None:
            # map desired labels to another label
            for idx in range(len(pred_labels)):
                if pred_labels[idx] in id_to_mapped_id.keys():
                    pred_labels[idx] = id_to_mapped_id[pred_labels[idx]]

        # labels_in_view = [self.id_to_name[pred_labels_] for pred_labels_ in list(pred_labels)]
        # print("labels in view:")
        # print(labels_in_view)

    out = {}
    out["pred_labels"] = pred_labels
    out["pred_boxes"] = pred_boxes
    out["pred_scores"] = pred_scores_labels
    if return_masks:
        out["pred_masks"] = pred_masks
    if return_features:
        out["features"] = features
    out["pred_interms"] = pred_interm_labels
    out["pred_interms_scores"] = pred_interm_scores
    out["pred_action"] = pred_action

    return out

@torch.no_grad()
def check_for_detections_local_policy_actionsonly(
    outputs,
    W, H, 
    score_labels_name, 
    score_boxes_name, 
    score_threshold_ddetr=0.0, 
    do_nms=True, 
    return_features=False, 
    target_object=None, 
    target_object_score_threshold=None, 
    solq=False, 
    return_masks=False, 
    nms_threshold=0.2, 
    id_to_mapped_id=None,
    return_policy_output=False,
    idx=0,
    ):
    '''
    rgb: single rgb image
    NOTE: currently this only handles a single RGB
    '''
    
    out = outputs['outputs']
    if return_features:
        features = out['features']
    else:
        features = None
    postprocessors = outputs['postprocessors']

    predictions = postprocessors['bbox'](out, None, features=features, return_features=return_features)
    pred_action = predictions[score_labels_name][idx]['labels_action']

    out = {}
    out["pred_action"] = pred_action[0]

    return out

@torch.no_grad()
def check_for_detections(
    rgb, 
    ddetr, 
    W, H, 
    score_labels_name, 
    score_boxes_name, 
    score_threshold_ddetr=0.0, 
    do_nms=True, 
    return_features=False, 
    target_object=None, 
    target_object_score_threshold=None, 
    only_keep_target=True, # only keep target object in 
    solq=True, 
    return_masks=False, 
    nms_threshold=0.2, 
    id_to_mapped_id=None,
    ):
    '''
    rgb: single rgb image
    NOTE: currently this only handles a single RGB
    '''
    
    rgb_PIL = Image.fromarray(rgb)
    rgb_norm = rgb.astype(np.float32) * 1./255
    rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).cuda()
    
    # self.ddetr.cuda()
    if solq:
        outputs = ddetr(rgb_torch.unsqueeze(0), do_loss=False, return_features=return_features)
    else:
        outputs = ddetr([rgb_torch], do_loss=False, return_features=return_features)
    # self.ddetr.cpu()
    
    out = outputs['outputs']
    if return_features:
        features = out['features']
    else:
        features = None
    postprocessors = outputs['postprocessors']

    # if return_features:
    #     predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=True)
    #     features = features[0]
    # else:
    if return_features:
        predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
        features = features[0]
        # if hyp.do_segmentation:
        #     predictions = postprocessors['segm'](predictions, out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())  
    else:
        predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
    
    pred_boxes = predictions[score_labels_name][0]['boxes']
    pred_labels = predictions[score_labels_name][0]['labels']
    pred_scores_boxes = predictions[score_boxes_name][0]['scores']
    pred_scores_labels = predictions[score_labels_name][0]['scores']
    if return_masks:
        pred_masks = predictions[score_labels_name][0]['masks']

    if id_to_mapped_id is not None:
        # map desired labels to another label
        for idx in range(len(pred_labels)):
            if pred_labels[idx] in id_to_mapped_id.keys():
                pred_labels[idx] = id_to_mapped_id[pred_labels[idx]]

    # above score threshold
    if target_object is not None:
        if type(target_object)==int:
            # print(len(pred_scores_labels), target_object)
            # print(pred_scores_labels[pred_labels==target_object])
            keep_target = torch.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
        elif type(target_object)==list:
            # keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            keep_target = []
            for lab_i in range(len(pred_labels)):
                check1 = pred_scores_labels[lab_i]>target_object_score_threshold
                check2 = pred_labels[lab_i] in target_object
                # check if above score and label in target_object list
                if check1 and check2:
                    keep_target.append(True)
                else:
                    keep_target.append(False)
            keep_target = np.array(keep_target)
        else:
            assert(False) # target_object should be int or list of ints
        if only_keep_target:
            keep = keep_target #np.logical_and(keep_target, keep)
        else:
            keep = torch.logical_or(pred_scores_labels>score_threshold_ddetr, keep_target)
    else:
        keep = pred_scores_labels>score_threshold_ddetr
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores_labels = pred_scores_labels[keep]
    pred_scores_boxes = pred_scores_boxes[keep]
    if return_features:
        features = features[keep]
    if return_masks:
        keep = keep.to(pred_masks.device)
        pred_masks = pred_masks[keep]
        
    if pred_boxes.shape[0]>1:
        
        if do_nms:
            keep, count = nms(pred_boxes, pred_scores_boxes, nms_threshold, top_k=100)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            # pred_scores = pred_scores[keep]
            pred_scores_boxes = pred_scores_boxes[keep]
            pred_scores_labels = pred_scores_labels[keep]
            if return_masks:
                pred_masks = pred_masks[keep]
            if return_features:
                features = features[keep]

    pred_boxes = pred_boxes.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()
    pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
    pred_scores_labels = pred_scores_labels.cpu().numpy() 
    if return_masks:
        pred_masks = pred_masks.squeeze(1).cpu().numpy() 

    # pred_boxes = pred_boxes.cpu().numpy()
    # pred_labels = pred_labels.cpu().numpy()
    # pred_scores = pred_scores.cpu().numpy()

    # labels_in_view = [self.id_to_name[pred_labels_] for pred_labels_ in list(pred_labels)]
    # print("labels in view:")
    # print(labels_in_view)

    out = {}
    out["pred_labels"] = pred_labels
    out["pred_boxes"] = pred_boxes
    out["pred_scores"] = pred_scores_labels
    if return_masks:
        out["pred_masks"] = pred_masks
    if return_features:
        out["features"] = features

    return out
    # if return_features:
    #     return pred_scores_labels, pred_labels, pred_boxes, features
    # else:
    #     return pred_scores_labels, pred_labels, pred_boxes

@torch.no_grad()
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

@torch.no_grad()
def check_for_detections_two_head(
    rgb, 
    ddetr,
     W, H, 
     score_labels_name, 
     score_labels_name2, 
     score_threshold_head1=0.0, 
     score_threshold_head2=0.0, 
     do_nms=True, 
     return_features=False, 
     target_object=None, 
     target_object_score_threshold=None, 
     solq=False, 
     return_masks=False, 
     nms_threshold=0.2, 
     id_to_mapped_id=None
     ):
    '''
    rgb: single rgb image
    NOTE: currently this only handles a single RGB
    '''
    
    rgb_PIL = Image.fromarray(rgb)
    rgb_norm = rgb.astype(np.float32) * 1./255
    rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).cuda()
    
    # self.ddetr.cuda()
    if solq:
        outputs = ddetr(rgb_torch.unsqueeze(0), do_loss=False, return_features=return_features)
    else:
        outputs = ddetr([rgb_torch], do_loss=False, return_features=return_features)
    # self.ddetr.cpu()
    
    out = outputs['outputs']
    if return_features:
        features = out['features']
    else:
        features = None
    postprocessors = outputs['postprocessors']

    # if return_features:
    #     predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=True)
    #     features = features[0]
    # else:
    if return_features:
        predictions, features = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
        features = features[0]
        # if hyp.do_segmentation:
        #     predictions = postprocessors['segm'](predictions, out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda())  
    else:
        predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), features=features, return_features=return_features)
    
    count = 0

    pred_boxes = predictions[score_labels_name][0]['boxes']
    pred_labels = predictions[score_labels_name][0]['labels']
    pred_scores_boxes = predictions[score_labels_name][0]['scores']
    pred_scores_labels = predictions[score_labels_name][0]['scores']
    if return_masks:
        pred_masks = predictions[score_labels_name][0]['masks']

    pred_boxes2 = predictions[score_labels_name2][0]['boxes']
    pred_labels2 = predictions[score_labels_name2][0]['labels']
    pred_scores_boxes2 = predictions[score_labels_name2][0]['scores']
    pred_scores_labels2 = predictions[score_labels_name2][0]['scores']
    if return_masks:
        pred_masks2 = predictions[score_labels_name2][0]['masks']
        
        
    if pred_boxes.shape[0]>1:
        
        if do_nms:
            keep, count = nms(pred_boxes, pred_scores_boxes, nms_threshold, top_k=100)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            # pred_scores = pred_scores[keep]
            pred_scores_boxes = pred_scores_boxes[keep]
            pred_scores_labels = pred_scores_labels[keep]
            if return_masks:
                pred_masks = pred_masks[keep]
            if return_features:
                features = features[keep]

            pred_boxes2 = pred_boxes2[keep]
            pred_labels2 = pred_labels2[keep]
            pred_scores_boxes2 = pred_scores_boxes2[keep]
            pred_scores_labels2 = pred_scores_labels2[keep]
            if return_masks:
                pred_masks2 = pred_masks2[keep]

        pred_boxes = pred_boxes.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
        pred_scores_labels = pred_scores_labels.cpu().numpy() 
        if return_masks:
            pred_masks = pred_masks.squeeze(1).cpu().numpy() 

        pred_boxes2 = pred_boxes2.cpu().numpy()
        pred_labels2 = pred_labels2.cpu().numpy()
        pred_scores_boxes2 = pred_scores_boxes2.cpu().numpy()
        pred_scores_labels2 = pred_scores_labels2.cpu().numpy()
        if return_masks:
            pred_masks2 = pred_masks2.cpu().numpy()

        # pred_boxes = pred_boxes.cpu().numpy()
        # pred_labels = pred_labels.cpu().numpy()
        # pred_scores = pred_scores.cpu().numpy()

        # above score threshold
        keep = pred_scores_labels>score_threshold_head1
        if target_object is not None:
            keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            keep = np.logical_or(keep_target, keep)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores_labels = pred_scores_labels[keep]
        if return_features:
            features = features[keep]
        if return_masks:
            pred_masks = pred_masks[keep]

        pred_boxes2 = pred_boxes2[keep]
        pred_labels2 = pred_labels2[keep]
        pred_scores_boxes2 = pred_scores_boxes2[keep]
        pred_scores_labels2 = pred_scores_labels2[keep]
        if return_masks:
            pred_masks2 = pred_masks2[keep]


        # above score threshold head 2
        keep = pred_scores_labels2>score_threshold_head2
        if target_object is not None:
            keep_target = np.logical_and(pred_scores_labels>target_object_score_threshold, pred_labels==target_object)
            keep = np.logical_or(keep_target, keep)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores_labels = pred_scores_labels[keep]
        if return_features:
            features = features[keep]
        if return_masks:
            pred_masks = pred_masks[keep]

        pred_boxes2 = pred_boxes2[keep]
        pred_labels2 = pred_labels2[keep]
        pred_scores_boxes2 = pred_scores_boxes2[keep]
        pred_scores_labels2 = pred_scores_labels2[keep]
        if return_masks:
            pred_masks2 = pred_masks2[keep]

        if id_to_mapped_id is not None:
            # map desired labels to another label
            for idx in range(len(pred_labels)):
                if pred_labels[idx] in id_to_mapped_id.keys():
                    pred_labels[idx] = id_to_mapped_id[pred_labels[idx]]

        # labels_in_view = [self.id_to_name[pred_labels_] for pred_labels_ in list(pred_labels)]
        # print("labels in view:")
        # print(labels_in_view)

        out = {}
        out["pred_labels"] = pred_labels
        out["pred_boxes"] = pred_boxes
        out["pred_scores"] = pred_scores_labels
        if return_masks:
            out["pred_masks"] = pred_masks
        if return_features:
            out["features"] = features
        out["pred_labels2"] = pred_labels2
        out["pred_boxes2"] = pred_boxes2
        out["pred_scores2"] = pred_scores_labels2
        if return_masks:
            out["pred_masks2"] = pred_masks2

    return out


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    # keep = scores.new(scores.size(0)).zero_().long()
    keep = []
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        # keep[count] = i
        keep.append(i)
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter.float() / union.float()  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    keep = torch.tensor(keep)
    return keep, count


def draw_ddetr_predictions(
    out_dict,
    images,
    targets,
    text,
    H,W,
    id_to_mapped_id,
    actions,
    reverse_transform=True
    ):

    out = check_for_detections_local_policy(
        out_dict, W, H, 
        'pred1', 'pred1', 
        score_threshold_ddetr=args.confidence_threshold,
        do_nms=True,
        target_object=None,
        target_object_score_threshold=args.confidence_threshold,
        solq=True, 
        return_masks=True, 
        nms_threshold=args.nms_threshold, 
        id_to_mapped_id=id_to_mapped_id, 
        return_features=False,
        )
    pred_labels = out["pred_labels"]
    pred_scores = out["pred_scores"]
    pred_masks = out["pred_masks"] 
    pred_boxes = out["pred_boxes"]
    pred_interms = out["pred_interms"] 
    pred_action = out["pred_action"] 
    pred_interms_scores = out["pred_interms_scores"]

    subgoal_vis = text[0]

    rgb = images[0][-1].cpu().numpy()
    if reverse_transform:
        image_mean = np.array([0.485,0.456,0.406]).reshape(3,1,1)
        image_std = np.array([0.229,0.224,0.225]).reshape(3,1,1)
        rgb = rgb * image_std + image_mean
    rgb = rgb*255.
    rgb = rgb.transpose(1,2,0).astype(np.float32)
    rgb_ = np.float32(rgb.copy())
    rect_th = 1
    for b in range(len(pred_boxes)):
        box = pred_boxes[b]
        if args.do_intermediate_loss:
            if pred_interms[b]==1:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
        else:
            color = (0, 255, 0)
        if len(box)==4:
            cv2.rectangle(rgb_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, rect_th)
        else:
            box = np.squeeze(box)
            masked_img = np.where(box[...,None], color, rgb_)
            rgb_ = cv2.addWeighted(rgb_, 0.8, np.float32(masked_img), 0.2,0)

    # add subgoal text
    cv2.putText(rgb_,subgoal_vis, (int(40*(H/480)),int(40*(W/480))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)

    # write action
    action_text = actions[int(pred_action.cpu().numpy())]
    cv2.putText(rgb_,action_text, (int(40*(H/480)),int(440*(W/480))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),thickness=1)
    
    # vis.add_frame(rgb_, text=f"{samples['subgoal']}: {cmd['action']}")
    img_torch = torch.from_numpy(rgb_).to(device).permute(2,0,1)/ 255.

    gt_boxes = targets[0]['boxes']
    gt_interms = list(targets[0]['obj_targets'].cpu().numpy())
    gt_action = actions[int(targets[0]['expert_action'].cpu().numpy())]
    gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
    gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] * W
    gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] * H

    

    rgb_ = np.float32(rgb.copy())
    for z in range(gt_boxes.shape[0]):
        box = gt_boxes[z]
        if args.do_intermediate_loss:
            if z in gt_interms:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
        else:
            color = (0, 255, 0)
        if len(box)==4:
            cv2.rectangle(rgb_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, rect_th)
        else:
            box = np.squeeze(box)
            masked_img = np.where(box[...,None], color, rgb_)
            rgb_ = cv2.addWeighted(rgb_, 0.8, np.float32(masked_img), 0.2,0)

    # add subgoal text
    cv2.putText(rgb_,subgoal_vis, (int(40*(H/480)),int(40*(W/480))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=1)

    # write action
    cv2.putText(rgb_,gt_action, (int(40*(H/480)),int(440*(W/480))), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),thickness=1)

    # plt.figure()
    # plt.imshow(rgb_.astype(int))
    # plt.savefig('data/images/test.png')
    # st()

    img_torch2 = torch.from_numpy(rgb_).to(device).permute(2,0,1)/ 255.

    img_torch = torch.cat([img_torch, img_torch2], dim=2)        

    return img_torch

class mAP():
    '''
    Util for computing mAP in COCO format
    '''
    def __init__(self, W, H, id_to_mapped_id): 
        '''
        This function tracks boxes and computes mAP
        '''
        self.W, self.H = W,H
        self.id_to_mapped_id = id_to_mapped_id

        self.labels = {0:'not_intermediate', 1:'intermediate'}

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

        # self.score_boxes_name = score_boxes_name # which prediction head output to use for scoring boxes
        # self.score_labels_name = score_labels_name # which prediction head to use for label scores

        self.mAP = {}
        self.classes = {}
        self.aps = {}
        self.ars = {}
        self.mAR = {}
        self.precision = {}
        self.recall = {}


    def add_boxes(
        self, 
        out_dict, 
        targets, 
        do_nms=True, 
        ):
        '''
        do_instance_map: Do instance-wise mAP rather than category-wise
        instance_map_with_cat: Should instances also predict the correct class to count as true positive?
        '''
        

        for t in range(len(targets)):

            out = check_for_detections_local_policy(
                out_dict, self.W, self.H, 
                'pred1', 'pred1', 
                score_threshold_ddetr=args.confidence_threshold,
                do_nms=do_nms,
                target_object=None,
                target_object_score_threshold=args.confidence_threshold,
                solq=True, 
                return_masks=True, 
                nms_threshold=args.nms_threshold, 
                id_to_mapped_id=self.id_to_mapped_id, 
                return_features=False,
                idx=t,
                )
            pred_labels = out["pred_labels"]
            pred_scores = out["pred_scores"]
            pred_masks = out["pred_masks"] 
            pred_boxes = out["pred_boxes"]
            pred_interms = out["pred_interms"] 
            pred_action = out["pred_action"] 
            pred_interms_scores = out["pred_interms_scores"]

            # pred_boxes = pred_boxes.cpu().numpy()
            # pred_labels = pred_labels.cpu().numpy()
            # pred_scores_boxes = pred_scores_boxes.cpu().numpy() 
            # pred_scores_labels = pred_scores_labels.cpu().numpy() 

            # Z = len(pred_interms_scores)

            for z in range(len(pred_interms_scores)):
                score = pred_interms_scores[z] #prob[prob_argmax].cpu().numpy()
                pred_box = pred_boxes[z]
                pred_label = pred_interms[z]
                pred_label = str(pred_label)
                
                bbox_params_det = [pred_label, score, pred_box[0], pred_box[1], pred_box[2], pred_box[3]]
                self.allBoundingBoxes, self.allClasses = add_bounding_box(bbox_params_det, self.allBoundingBoxes, self.allClasses, nameOfImage=self.count, isGT=False, imgSize=self.imgSize, Format=self.detFormat, CoordType=self.detCoordType)
            
            # GT
            gt_boxes = targets[t]['boxes']
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            gt_boxes[:,[0,2]] = gt_boxes[:,[0,2]] * self.W
            gt_boxes[:,[1,3]] = gt_boxes[:,[1,3]] * self.H
            gt_labels = np.ones(len(gt_boxes), dtype=int) # 1 is intermediate label - all GT boxes are intermediate   #targets[t][obj_targets]

            for z in range(len(gt_boxes)):
                gt_box = gt_boxes[z]
                gt_label = gt_labels[z]
                gt_label = str(gt_labels[z])
                bbox_params_gt = [gt_label, gt_box[0], gt_box[1], gt_box[2], gt_box[3]]
                self.allBoundingBoxes, self.allClasses = add_bounding_box(bbox_params_gt, self.allBoundingBoxes, self.allClasses, nameOfImage=self.count, isGT=True, imgSize=self.imgSize, Format=self.gtFormat, CoordType=self.gtCoordType)
            self.count += 1

    def get_stats(self, IOU_threshold=0.5):
        mAP, classes, aps, ars, mAR, precision, recall = get_map(self.allBoundingBoxes, self.allClasses, IOU_threshold=IOU_threshold) # consider_nonzero_TP=True for instance detection to also consider False positive
        self.mAP[IOU_threshold] = mAP
        self.classes[IOU_threshold] = classes
        self.aps[IOU_threshold] = aps
        self.ars[IOU_threshold] = ars
        self.mAR[IOU_threshold] = mAR
        self.precision[IOU_threshold] = precision
        self.recall[IOU_threshold] = recall
        return mAP