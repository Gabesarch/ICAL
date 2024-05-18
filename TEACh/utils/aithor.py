import torch 
import numpy as np
import utils.geom
import math
import quaternion

from scipy.spatial.transform import Rotation as R

import ipdb
st = ipdb.set_trace

import cv2

import matplotlib.pyplot as plt
from PIL import Image

import utils.box

# def get_origin_T_camX(event, standing=True, add_camera_height=True):
#     if add_camera_height:
#         position = np.array(list(event.metadata["agent"]["position"].values())) + np.array([0.0, 0.675, 0.0]) # adjust for camera height from agent
#     else:
#         position = np.array(list(event.metadata['cameraPosition'].values()))
#     rotation = np.array(list(event.metadata["agent"]["rotation"].values()))
#     rx_deg = event.metadata["agent"]["cameraHorizon"]
#     if rx_deg < 0: # numpy takes range 0-360
#         rx_deg = 360 + rx_deg
#     ry_deg = rotation[1]
#     if ry_deg < 0:
#         ry_deg = 360 + ry_deg
#     rx = -np.radians(rx_deg) # pitch
#     ry = np.radians(ry_deg) # yaw
#     rz = 0. # roll is always 0
#     # r = R.from_euler('yxz', [rx,ry,rz])
#     # rotm = r.as_matrix()
#     # eul = r.as_euler('yxz', degrees=True)
#     # print(0, rx_deg, ry_deg, rz)
#     # print(1, rx,ry,rz)
#     rotm = utils.geom.eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
#     # rx,ry,rz = utils.geom.rotm2eul(torch.from_numpy(rotm))
#     # print(2, eul[0],eul[1],eul[2])
#     origin_T_camX = np.eye(4)
#     origin_T_camX[0:3,0:3] = rotm
#     origin_T_camX[0:3,3] = position
#     origin_T_camX = torch.from_numpy(origin_T_camX)
#     return origin_T_camX

def compute_metrics(success, reward, task, t, pcs):
    '''
    compute metrics for task evaluation
    taken from Episodic Transformer codebase
    '''
    # goal_conditions
    goal_condition_success_rate = pcs[0] / float(pcs[1])
    # SPL
    path_len_weight = len(task['plan']['low_actions'])
    s_spl = (1 if success else 0) * min(1., path_len_weight / float(t))
    pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))
    # path length weighted SPL
    plw_s_spl = s_spl * path_len_weight
    plw_pc_spl = pc_spl * path_len_weight
    metrics = {'completed_goal_conditions': int(pcs[0]),
               'total_goal_conditions': int(pcs[1]),
               'goal_condition_success': float(goal_condition_success_rate),
               'success_spl': float(s_spl),
               'path_len_weighted_success_spl': float(plw_s_spl),
               'goal_condition_spl': float(pc_spl),
               'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
               'path_len_weight': int(path_len_weight),
               'reward': float(reward),
               'success': success}
    return metrics

def get_centroids_from_masks(pitch, xyz, mask_nopad, W, H, camX0_T_camX=None):
        # adjust point cloud for agent pitch
        rx = -torch.tensor([targets['agent_pose'][-1]]) # pitch - positive is down, negative is up in aithor
        ry = torch.zeros(1) #.to(self.device) # yaw
        rz = torch.zeros(1) #.to(self.device) # roll  
        rot = utils.geom.eul2rotm(rx, ry, rz)
        rotm = utils.geom.eye_4x4(1, device='cpu').to(xyz.dtype) #.to(self.device)
        rotm[:,0:3,0:3] = rot
        xyz = utils.geom.apply_4x4(rotm, xyz)
        xyz[:,:,1] = -xyz[:,:,1]
        if camX0_T_camX is None:
            xyz_camX0 = xyz
        else:
            xyz_camX0 = utils.geom.apply_4x4(camX0_T_camX.float(), xyz).squeeze().numpy()
        xyz_camX0 = xyz_camX0.reshape(W,H,3) 
        obj_centroid = np.zeros((len(mask_nopad), 3))
        for m_i in range(len(mask_nopad)):
            centroid = get_centroid_from_detection_no_controller(
                            mask_nopad[m_i], 
                            xyz_camX0, 
                            W, H, 
                            centroid_mode='median', 
                            # min_depth=0.0, 
                            # max_depth=50.0, 
                            # num_valid_thresh=1,
            )
            obj_centroid[m_i] = np.squeeze(centroid)
        return obj_centroid

def read_task_data(task, subgoal_idx=None):
    '''
    read data from the traj_json
    taken from Episodic Transformer codebase
    '''
    # read general task info
    st()
    repeat_idx = task['repeat_idx']
    task_dict = {'repeat_idx': repeat_idx,
                 'type': task['task_type'],
                 'task': '/'.join(task['root'].split('/')[-3:-1])}
    # read subgoal info
    if subgoal_idx is not None:
        task_dict['subgoal_idx'] = subgoal_idx
        task_dict['subgoal_action'] = task['plan']['high_pddl'][
            subgoal_idx]['discrete_action']['action']
    return task_dict

def get_origin_T_camX(event, invert_pitch, standing=True, add_camera_height=True):
    if isinstance(event, dict):
        position = np.array(list(event["position"].values())) + np.array([0.0, 0.675, 0.0])
        rotation = np.array(list(event["rotation"].values()))
        rx = np.radians(event["horizon"]) # pitch
    else:
        if add_camera_height:
            position = np.array(list(event.metadata["agent"]["position"].values())) + np.array([0.0, 0.675, 0.0]) # adjust for camera height from agent
        else:
            position = np.array(list(event.metadata['cameraPosition'].values()))
        rotation = np.array(list(event.metadata["agent"]["rotation"].values()))
        rx = np.radians(event.metadata["agent"]["cameraHorizon"]) # pitch
    if invert_pitch: # in aithor negative pitch is up - turn this on if need the reverse
       rx = -rx 
    ry = np.radians(rotation[1]) # yaw
    rz = 0. # roll is always 0
    rotm = utils.geom.eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
    origin_T_camX = np.eye(4)
    origin_T_camX[0:3,0:3] = rotm
    origin_T_camX[0:3,3] = position
    origin_T_camX = torch.from_numpy(origin_T_camX)
    # st()
    # utils.geom.apply_4x4(origin_T_camX.float(), torch.tensor([0.,0.,0.]).unsqueeze(0).unsqueeze(0).float())
    return origin_T_camX

def get_origin_T_camX_from_xyz_rot(position, yaw, pitch, standing=True, add_camera_height=True):
    '''
    add_camera_height: in aithor, camera height = agent height + 0.675
    '''
    if add_camera_height:
        position += np.array([0.0, 0.675, 0.0])
    rx = np.radians(pitch) #np.radians(rotation[1]) # yaw
    ry = np.radians(yaw) #np.radians(rotation[1]) # yaw
    rz = 0. # roll is always 0
    rotm = utils.geom.eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
    # rotm = quaternion.as_rotation_matrix(quaternion.from_euler_angles(np.concatenate([[rx, ry, rz]],axis=0), beta=None, gamma=None))
    # rotm = quaternion.as_rotation_matrix(quaternion.from_euler_angles(np.concatenate([[ry, rx, rz]],axis=0), beta=None, gamma=None)) # convention is yaw-pitch-roll or alpha-beta-gamma
    origin_T_camX = np.eye(4)
    origin_T_camX[0:3,0:3] = rotm
    origin_T_camX[0:3,3] = position
    origin_T_camX = torch.from_numpy(origin_T_camX)
    # st()
    # utils.geom.apply_4x4(origin_T_camX.float(), torch.tensor([0.,0.,0.]).unsqueeze(0).unsqueeze(0).float())
    return origin_T_camX

def get_origin_T_camX_from_xyz_rot_batch(position, yaw, pitch, standing=True, add_camera_height=True):
    B = len(yaw)
    if add_camera_height:
        position += np.expand_dims(np.array([0.0, 0.675, 0.0]), 0).repeat(B, 0)
    rx = np.radians(pitch) #np.radians(rotation[1]) # yaw
    ry = np.radians(yaw) #np.radians(rotation[1]) # yaw
    rz = np.zeros(B) # roll is always 0
    # rotm = quaternion.as_rotation_matrix(quaternion.from_euler_angles(np.stack([ry, rx, rz],axis=1), beta=None, gamma=None))
    rotm = utils.geom.eul2rotm_py(rx, ry, rz)
    origin_T_camX = np.expand_dims(np.eye(4), 0).repeat(B, 0)
    origin_T_camX[:,0:3,0:3] = rotm
    origin_T_camX[:,0:3,3] = position
    origin_T_camX = torch.from_numpy(origin_T_camX)
    # st()
    # utils.geom.apply_4x4(origin_T_camX.float(), torch.tensor([0.,0.,0.]).unsqueeze(0).unsqueeze(0).float())
    return origin_T_camX

def get_xyz_from_depth(depth, H, W):
    # # get pointcloud
    xs, ys = np.meshgrid(np.linspace(-1*256/2.,1*256/2.,256), np.linspace(1*256/2.,-1*256/2., 256))
    depth = depth.reshape(1,256,256)
    xs = xs.reshape(1,256,256)
    ys = ys.reshape(1,256,256)
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(self.K), xys)
    xyz = xy_c0.T[:,:3].reshape(256,256,3)
    return xyz

def get_3dbox_in_geom_format(obj_meta):
    if obj_meta['objectOrientedBoundingBox'] is not None:
        obj_3dbox_origin = np.array(obj_meta['objectOrientedBoundingBox']['cornerPoints'])
        obj_3dbox_origin = torch.from_numpy(obj_3dbox_origin).unsqueeze(0).unsqueeze(0)
    else:
        obj_3dbox_origin = np.array(obj_meta['axisAlignedBoundingBox']['cornerPoints'])
        # reorder_x = np.array([0,1,4,5,2,3,6,7]) # switch between aithor and utils.geom format
        # reorder_y = np.array([0,1,4,5,2,3,6,7])
        # reorder_z = np.array([0,2,3,1,4,6,7,5])
        # obj_3dbox_origin[:,0] = obj_3dbox_origin[reorder_x,0]
        # obj_3dbox_origin[:,1] = obj_3dbox_origin[reorder_y,1]
        # obj_3dbox_origin[:,2] = obj_3dbox_origin[reorder_z,2]

        ry = 0. #obj_rot['x']) 
        rx = 0. #np.radians(obj_rot['y'])
        rz = 0. #np.radians(obj_rot['z'])
        xc = np.mean(obj_3dbox_origin[:,0])
        yc = np.mean(obj_3dbox_origin[:,1])
        zc = np.mean(obj_3dbox_origin[:,2])
        lx = np.max(obj_3dbox_origin[:,0]) - np.min(obj_3dbox_origin[:,0])
        ly = np.max(obj_3dbox_origin[:,1]) - np.min(obj_3dbox_origin[:,1])
        lz = np.max(obj_3dbox_origin[:,2]) - np.min(obj_3dbox_origin[:,2])
        box_origin = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
        box_origin = torch.from_numpy(box_origin).unsqueeze(0).unsqueeze(0)
        obj_3dbox_origin = utils.geom.transform_boxes_to_corners(box_origin.cuda().float()).cpu().double()

    return obj_3dbox_origin

def get_masks_from_seg(
    segm, 
    mask_colors,
    W,H,
    max_objs,
    add_whole_image_mask=True,
    filter_inds=None
    ):
    '''
    get masks from segmentation image and mask color path
    segm: segmentation mask
    '''
        
    if len(mask_colors)==0: # no objects
        mask = np.ones((0, W, H), dtype=bool) 
    elif len(mask_colors.shape)==3: # input is masks already
        mask = mask_colors
    else:
        mask = np.asarray([np.all(segm==np.tile(seg_color, [W, H, 1]), axis=2) for seg_color in mask_colors], dtype=bool)

    if filter_inds is not None:
        # filter masks if we want
        if len(filter_inds)==0:
            mask = np.ones((0, W, H), dtype=bool) 
        else:
            mask = mask[filter_inds]

    mask_nopad = mask
    # targets["masks"] = mask
    
    if add_whole_image_mask or len(mask)==0: # Also, take whole image as context if no masks (else can be unstable)
        mask = np.concatenate([np.ones((1, W, H), dtype=bool), mask]) # add whole image as context
    
    if mask.shape[0] > max_objs: # check if over max objs
        mask = mask[:max_objs]

    # pad masks to max objects
    npad = ((0, max_objs-mask.shape[0]), (0, 0), (0, 0))
    mask = np.pad(mask, pad_width=npad, mode='constant', constant_values=False)

    return mask, mask_nopad

def get_3dbox_in_geom_format_alfred(obj_meta):

    obj_3dbox_origin = obj_meta['objectBounds']['objectBoundsCorners']
    obj_3dbox_origin_ = []
    for o in obj_3dbox_origin:
        obj_3dbox_origin_.append([o['x'], o['y'], o['z']])
    obj_3dbox_origin_ = np.array(obj_3dbox_origin_)
    obj_3dbox_origin = obj_3dbox_origin_

    ry = 0. 
    rx = 0. 
    rz = 0. 
    xc = np.mean(obj_3dbox_origin[:,0])
    yc = np.mean(obj_3dbox_origin[:,1])
    zc = np.mean(obj_3dbox_origin[:,2])
    lx = np.max(obj_3dbox_origin[:,0]) - np.min(obj_3dbox_origin[:,0])
    ly = np.max(obj_3dbox_origin[:,1]) - np.min(obj_3dbox_origin[:,1])
    lz = np.max(obj_3dbox_origin[:,2]) - np.min(obj_3dbox_origin[:,2])
    box_origin = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
    box_origin = torch.from_numpy(box_origin).unsqueeze(0).unsqueeze(0)
    obj_3dbox_origin = utils.geom.transform_boxes_to_corners(box_origin.cuda().float()).cpu().double()

    return obj_3dbox_origin

def aithor3D_to_2D(controller, aithor_3D_point_origin, pix_T_cam):
    origin_T_camX = get_origin_T_camX(controller.last_event, False)
    camX_T_origin = utils.geom.safe_inverse_single(origin_T_camX)
    aithor_3D_point_camX = utils.geom.apply_4x4(camX_T_origin.unsqueeze(0), aithor_3D_point_origin) # tranform point from origin to agent reference frame - Should be B x N_points x 3
    point_2D = utils.geom.apply_pix_T_cam(pix_T_cam, aithor_3D_point_camX) # transform from agent frame to pixel coords
    return point_2D

def get_amodal2d(origin_T_camX, obj_3dbox_origin, pix_T_camX, H, W):
    camX_T_origin = utils.geom.safe_inverse_single(origin_T_camX)
    obj_3dbox_camX = utils.geom.apply_4x4_to_corners(camX_T_origin.unsqueeze(0), obj_3dbox_origin)
    boxlist2d_amodal = utils.geom.get_boxlist2d_from_corners_aithor(pix_T_camX, obj_3dbox_camX, H, W)[0][0]
    return boxlist2d_amodal, obj_3dbox_camX

def get_floor_height(controller, W, H):
    times_moved = 0
    while True:
        controller.step("LookDown")
        if not controller.last_event.metadata["lastActionSuccess"]:
            break
        times_moved += 1

    # ray cast
    query = controller.step(
        action="GetCoordinateFromRaycast",
        x=0.5,
        y=1,
    )
    coordinate = query.metadata["actionReturn"]
    c_depth = np.array(list(coordinate.values()))

    # move back
    for _ in range(times_moved):
        controller.step("LookUp")

    floor_height = c_depth[1]
    return floor_height

def get_centroid_from_detection_no_controller(
    box, 
    depth, 
    W, H, 
    centroid_mode='median', 
    pix_T_camX=None, 
    origin_T_camX=None, 
    min_depth=0.0, 
    max_depth=12.0, 
    num_valid_thresh=50,
    xyz_origin=None,
    ):
    '''
    box: 1x4 bbox or mask HxW
    depth: depth image or point cloud shape H,W or H,W,3
    centroid mode: "median" takes centroid at median depth of bounding box, "middle" takes center point of bbox
    input_xyz: Is depth input a pointcloud? If True then yes, if False then it is depth map
    '''

    input_xyz = True if len(depth.shape)==3 else False

    is_box = len(box)==4
    if is_box:
        x_min, y_min, x_max, y_max = list(np.round(box).astype(np.int))
        if x_max>=W:
            x_max = W-1
        if y_max>=H:
            y_max = H-1
        if x_min<0:
            x_min = 0
        if y_min<0:
            y_min = 0

    if centroid_mode=='median':

        if is_box:
            x_inds = np.arange(x_min, x_max)
            y_inds = np.arange(y_min, y_max)
            xv, yv = np.meshgrid(x_inds, y_inds)
            xv = xv.flatten()
            yv = yv.flatten()
        else:
            # print("Using Mask")
            yv, xv = np.where(box)

        depth = np.squeeze(depth)
        depth_box = depth[yv, xv]

        if not input_xyz:
            valid_depth = np.ones_like(depth_box).astype(bool)
            valid_depth = np.logical_and(valid_depth, ~np.isnan(depth_box))
            if max_depth is not None:
                valid_depth = np.logical_and(valid_depth, depth_box < max_depth)
            if min_depth is not None:
                valid_depth = np.logical_and(valid_depth, depth_box > min_depth)
            where_valid = np.where(valid_depth)[0]
            if len(where_valid) < num_valid_thresh:
                return None
            depth_box_valid = depth_box[valid_depth]
            argmedian_valid = np.argsort(depth_box_valid)[len(depth_box_valid)//2] 
            argmedian = where_valid[argmedian_valid]
            # ray casting to get centroid
            # this accomplishes the same thing as raycasting without calling the command
            xv_median = xv[argmedian]
            yv_median = yv[argmedian]
            if xyz_origin is None:
                depth_ = torch.from_numpy(depth).cuda().unsqueeze(0).unsqueeze(0)
                xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(pix_T_camX).cuda().unsqueeze(0).float())
                xyz_origin = utils.geom.apply_4x4(origin_T_camX.cuda().float(), xyz).squeeze().cpu().numpy()
                xyz_origin = xyz_origin.reshape(1,W,H,3) 
            c_depth = np.squeeze(xyz_origin[:,yv_median,xv_median,:])
        else:
            c_depth = np.expand_dims(np.median(depth_box, axis=0), axis=0)     
    else:
        assert False # not implemented     

    return c_depth

def get_centroid_from_detection(controller, box, depth, W, H, centroid_mode='median', use_ray_cast=True, pix_T_camX=None, origin_T_camX0=None, min_depth=None, max_depth=None, num_valid_thresh=50):
    '''
    controller: optional if not using ray casting
    box: 1x4 bbox
    depth: depth image
    centroid mode: "median" takes centroid at median depth of bounding box, "middle" takes center point of bbox
    min_depth: minimum depth allowable in the box
    max_depth: maximum depth allowable in the box
    num_valid_thresh: if number of valid depth less than this, throw out centroid
    '''

    x_min, y_min, x_max, y_max = list(np.round(box).astype(np.int))
    if x_max>=W:
        x_max = W-1
    if y_max>=H:
        y_max = H-1
    if x_min<0:
        x_min = 0
    if y_min<0:
        y_min = 0

    if centroid_mode=='median':
        x_inds = np.arange(x_min, x_max)
        y_inds = np.arange(y_min, y_max)
        xv, yv = np.meshgrid(x_inds, y_inds)
        xv = xv.flatten()
        yv = yv.flatten()
        depth = np.squeeze(depth)
        depth_box = depth[yv, xv]
        # plt.figure()
        # plt.imshow(depth_box.reshape(y_inds.shape[0], x_inds.shape[0]))
        # plt.savefig('images/test.png')
        # plt.figure()
        # depth_plt = depth_.cpu().numpy()
        # depth_plt[depth_plt>5] = 1.
        # plt.imshow(depth_plt)
        # plt.savefig('images/test2.png')
        valid_depth = np.ones_like(depth_box).astype(bool)
        if max_depth is not None:
            valid_depth = np.logical_and(valid_depth, depth_box < max_depth)
        if min_depth is not None:
            valid_depth = np.logical_and(valid_depth, depth_box > min_depth)
        where_valid = np.where(valid_depth)[0]
        if len(where_valid) < num_valid_thresh:
            return None
        depth_box_valid = depth_box[valid_depth]
        argmedian_valid = np.argsort(depth_box_valid)[len(depth_box_valid)//2] 
        argmedian = where_valid[argmedian_valid]
        
        xv_median = xv[argmedian]
        yv_median = yv[argmedian]

        # ray casting to get centroid
        if use_ray_cast:
            # Note: if fed in dictionary, must not use raycasting
            query = controller.step(
                action="GetCoordinateFromRaycast",
                x=xv_median/W,
                y=yv_median/H,
            )
            coordinate = query.metadata["actionReturn"]
            c_depth = np.array(list(coordinate.values()))
        else:
            # this accomplishes the same thing as raycasting without calling the command
            depth_ = torch.from_numpy(depth).cuda().unsqueeze(0).unsqueeze(0)
            xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(pix_T_camX).cuda().unsqueeze(0).float())
            if isinstance(controller, dict):
                origin_T_camX = utils.aithor.get_origin_T_camX(controller, True)
            else:
                origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, True)
            origin_T_camX[1,3] = -origin_T_camX[1,3]
            xyz_origin = utils.geom.apply_4x4(origin_T_camX.cuda().float(), xyz).squeeze().cpu().numpy()
            xyz_origin = xyz_origin.reshape(1,W,H,3) 
            c_depth = np.squeeze(xyz_origin[:,yv_median,xv_median,:])
            c_depth[1] = -c_depth[1]
    elif centroid_mode=='middle':
        c_x, c_y = int(np.round((x_max+x_min)/2)), int(np.round((y_max+y_min)/2))
        query = controller.step(
            action="GetCoordinateFromRaycast",
            x=c_x/W,
            y=c_y/H
        )
        coordinate = query.metadata["actionReturn"]
        c_depth = np.array(list(coordinate.values()))
    else:
        assert(False)

    return c_depth

def get_centroid_from_detection_torch(boxes, depths, W, H, origin_T_camX, centroid_mode='median', use_ray_cast=True, pix_T_camX=None):
    '''
    controller: optional if not using ray casting
    box: 1x4 bbox
    depth: depth image
    centroid mode: "median" takes centroid at median depth of bounding box, "middle" takes center point of bbox
    '''
    with torch.no_grad():
        B = depths.shape[0]
        xyz = utils.geom.depth2pointcloud(depths.unsqueeze(1), pix_T_camX.repeat((B,1,1)))
        origin_T_camX[1,3] = -origin_T_camX[1,3]
        xyz_origin = utils.geom.apply_4x4(origin_T_camX.cuda().float(), xyz).squeeze()
        xyz_origin = xyz_origin.reshape(B,W,H,3) 

        print("NOTE: ASSUMING ENTIRE BATCH IS THE SAME SCENE FOR FLOOR HEIGHT")
        floor_height = torch.min(xyz_origin.reshape(B*W*H, 3)[:,1]).cpu().numpy()

        centroids = []
        for b_i in range(B):
            boxes_ = boxes[b_i]
            xyz_origin_ = xyz_origin[b_i]
            depth_ = depths[b_i].squeeze()
            centroids_ = []
            for box in boxes_:

                x_min, y_min, x_max, y_max = list(torch.clip(torch.round(box), 0, W-1).cpu().numpy())
                if x_min==x_max:
                    if x_max<W-1:
                        x_max += 1
                    else:
                        x_min -= 1
                if y_min==y_max:
                    if y_max<H-1:
                        y_max += 1
                    else:
                        y_min -= 1

                if y_max < y_min: # just in case
                    y_max_ = y_max
                    y_min_ = y_min
                    y_max = y_min_
                    y_min = y_max_

                if x_max < x_min: # just in case
                    x_max_ = x_max
                    x_min_ = x_min
                    x_max = x_min_
                    x_min = x_max_

                if centroid_mode=='median':
                    x_inds = np.arange(x_min, x_max)
                    y_inds = np.arange(y_min, y_max)
                    xv, yv = np.meshgrid(x_inds, y_inds)
                    xv = xv.flatten()
                    yv = yv.flatten()
                    depth_box = depth_[yv, xv]
                    argmedian = torch.argsort(depth_box)[len(depth_box)//2]              
                    xv_median = int(xv[argmedian])
                    yv_median = int(yv[argmedian])
                else:
                    assert(False)

                # this accomplishes the same thing as raycasting without calling the command
                c_depth = xyz_origin_[yv_median,xv_median,:].squeeze()
                c_depth[1] = -c_depth[1]
                centroids_.append(c_depth)
            centroids_ = torch.stack(centroids_)
            centroids.append(centroids_)

    return centroids, floor_height

def get_obj_inf_center_box(
    pred_boxes, pred_scores, depth, 
    controller, W, H, include_classes, 
    objects, 
    obj_category=None, receptacles_only=False, use_iou=True, 
    first_search_class=None, centroid_mode='median'
    ):
    # get estimated centroid of object
    # c_depth = depth[c_x, c_y]
    # depth_ = torch.from_numpy(depth).cuda().unsqueeze(0).unsqueeze(0)
    # xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(pix_T_camX).cuda().unsqueeze(0).float())
    # origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, True)
    # xyz_origin = utils.geom.apply_4x4(origin_T_camX.cuda().float(), xyz).squeeze().cpu().numpy()
    # xyz_origin = xyz_origin.reshape(1,W,H,3)    
    c_depth = get_centroid_from_detection(controller, pred_boxes, depth, W, H, centroid_mode=centroid_mode)
    
    obj_thisone = {}
    if not use_iou:
        query = controller.step(
            action="GetObjectInFrame",
            x=c_x/W,
            y=c_y/H,
            checkVisible=False
        )
        object_id = query.metadata["actionReturn"]
        # print(object_id)
        for obj in objects:
            if obj['objectId']==object_id:
                obj_thisone = obj
    elif use_iou: # this is used just for placing + verification so we'll use gt for getting ID
        object_dict = {}
        for obj in objects:
            object_dict[obj['objectId']] = obj
        
        detections = controller.last_event.instance_detections2D

        # if 'CounterTop' in first_search_class:
        #     st()

        # first search if class is in view, then do IOU - we just use this GT info for placement
        if first_search_class is not None:
            for key in list(detections.keys()):
                if first_search_class in key:
                    box_ = detections[key]
                    iou_inview = utils.box.boxlist_2d_iou(box_.reshape(1,4), pred_boxes.reshape(1,4))
                    iou_inview = np.squeeze(iou_inview)
                    if iou_inview>0.2 and key in object_dict:
                        obj_thisone = object_dict[key]
                        break

        

        # if didn't find one, search again
        if len(obj_thisone)==0:
            boxes = []
            IDs = []
            for key in list(detections.keys()):
                if key.split('|')[0] not in include_classes:
                    continue
                if key not in object_dict:
                    continue
                if (not object_dict[key]['receptacle'] or object_dict[key]['openable']) and receptacles_only: # must be a receptacle and dont have support for openable items yet
                    continue
                box_ = detections[key]
                IDs.append(key)
                boxes.append(box_)
            if len(boxes)==0:
                return {}
            boxes = np.stack(boxes)
            iou_inview = utils.box.boxlist_2d_iou(boxes, pred_boxes.reshape(1,4))
            if True:
                argmax_iou = np.argmax(np.squeeze(iou_inview))
                best_iou_ID = IDs[argmax_iou]
                obj_thisone = object_dict[best_iou_ID]
            else:
                obj_thisone = []
                for iou_i in range(len(iou_inview)):
                    if iou_inview[iou_i]>0.2:
                        ID = IDs[iou_i]
                        obj_thisone.append(object_dict[ID])


        # for obj in objects:
        #     if obj['objectId']==best_iou_ID:
        #         obj_thisone = obj
            # print(obj['objectType'], obj_type)
            # if obj['objectType']==obj_type and obj['visible']:
            #     obj_thisone = obj
    if len(obj_thisone)==0:
        st()
        return {}
    # print("Object detected is", obj_thisone['name'])
    # if 'Floor' in obj_thisone['name']:
    #     return {}
    in_view = {}
    in_view['box'] = pred_boxes
    in_view['score'] = pred_scores
    in_view['obj_center'] = c_depth #np.array(list(obj_thisone['axisAlignedBoundingBox']['center'].values())) #c_depth
    in_view['objectId'] = obj_thisone['objectId']
    in_view['objectType'] = obj_thisone['objectType']
    in_view['receptacle'] = obj_thisone['receptacle']
    in_view['name'] = obj_thisone['name']
    return in_view

def move_held_obj_out_of_view(controller, action="MoveHeldObjectUp"):
    # sum_ = 0
    while True:
        controller.step(
            action=action,
            moveMagnitude=0.05,
            forceVisible=False
        )
        # sum_ += 0.05
        if not controller.last_event.metadata["lastActionSuccess"]:
            break
    # print("magnitude of movement:", sum_)

def change_pose(objects, obj_name=None, obj_pos=None, obj_rot=None):
    '''
    outputs object poses to change pose of one specified object for use in action="SetObjectPoses"
    '''
    objectPoses = []
    for obj in objects:
        if not obj["pickupable"] and not obj["moveable"]:
            continue
        if obj_name is not None and obj['name']==obj_name:
            objectPoses.append({"objectName":obj_name, "rotation":obj_rot, "position":obj_pos})
        else:
            objectPoses.append({"objectName":obj["name"], "rotation":obj["rotation"], "position":obj["position"]})
    return objectPoses

def get_map_type(mapname):
    map_num = mapname.split('FloorPlan')
    map_num = int(map_num[1])
    if map_num<100:
        map_type = 'kitchen'
    elif map_num>200 and map_num<300:
        map_type = 'living_room'
    elif map_num>300 and map_num<400:
        map_type = 'bedroom'
    elif map_num>400:
        map_type = 'bathroom'
    return map_type

def get_closest_navigable_point(positions, nav_points):
    '''
    positions = Nx3 positions to get closest match
    nav_points = N2x3 all nav points
    '''
    positions_ = positions[:,[0,2]].unsqueeze(0).float()
    nav_points_ = nav_points[:,[0,2]].unsqueeze(0).float()
    # if not positions_.shape[0]==nav_points_.shape[0] or not positions_.shape[2]==nav_points_.shape[2]:
    #     st()
    dist = torch.cdist(positions_, nav_points_).squeeze(0)
    min_dists, argmins = torch.min(dist, dim=1)
    positions_navigable = nav_points[argmins,:]
    # positions_navigable = []
    # min_dists = []
    # st()
    # for p in positions:
    #     # only use z and x (since y is the same anyways)
    #     dist = np.linalg.norm(nav_points[:,[0,2]] - p[[0,2]], axis=1)
    #     amin = np.argmin(dist)
    #     min_dist = np.min(dist)
    #     min_dists.append(min_dists)
    #     pos_closest = nav_points[amin]
    #     positions_navigable.append(pos_closest)
    # positions_navigable = np.array(positions_navigable)
    # min_dists = np.array(min_dists)
    return positions_navigable, min_dists

def get_yaw_pitch_to_nearest_obj(controller, current_position, object_class):

    centers = []
    for obj in controller.last_event.metadata['objects']:
        if obj['objectType'] != object_class:
            continue
        obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))         
        obj_center = np.expand_dims(obj_center, axis=0)
        centers.append(obj_center)
    centers = np.stack(centers)
    centers = torch.from_numpy(centers).cuda()
    
    current_position = torch.from_numpy(current_position).cuda()
    closest_center,_ = get_closest_navigable_point(current_position.unsqueeze(0), centers.squeeze(0).squeeze(1))
    closest_center = closest_center.squeeze(0).cpu().numpy()
    current_position = current_position.squeeze(0).cpu().numpy()

    # YAW calculation - rotate to object
    agent_to_obj = np.squeeze(closest_center) - current_position 
    agent_local_forward = np.array([0, 0, 1.0]) 
    flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
    flat_dist_to_obj = np.linalg.norm(flat_to_obj)
    flat_to_obj /= flat_dist_to_obj

    det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
    turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

    # add noise so not right in the center
    # noise = np.random.normal(0, 2, size=2)

    turn_yaw = np.degrees(turn_angle)
    turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))

    return turn_yaw, turn_pitch

def get_scene_bounds(controller):
    # get rough scene boundaries
    all_obj_xmin = 1000
    all_obj_xmax = -1000
    all_obj_zmin = 1000
    all_obj_zmax = -1000
    for obj in controller.last_event.metadata['objects']:
        if obj['objectType'] in ['Floor']:
            continue
        bbox = np.array(obj['axisAlignedBoundingBox']['cornerPoints'])
        try:
            xmin, xmax = np.min(bbox[:,0]), np.max(bbox[:,0])
            ymin, ymax = np.min(bbox[:,1]), np.max(bbox[:,1])
            zmin, zmax = np.min(bbox[:,2]), np.max(bbox[:,2])
        except:
            continue

        all_obj_xmin = min(all_obj_xmin, xmin)
        all_obj_xmax = max(all_obj_xmax, xmax)
        all_obj_zmin = min(all_obj_zmin, zmin)
        all_obj_zmax = max(all_obj_zmax, zmax)
        # if all_obj_xmin==xmin:
        #     obj_xmin = obj['objectType']
        # if all_obj_xmax==xmax:
        #     obj_xmax = obj['objectType']
        # if all_obj_zmin==zmin:
        #     obj_zmin = obj['objectType']
        # if all_obj_zmax==zmax:
        #     obj_zmax = obj['objectType']

    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    reachable_pos = np.array([[pos['x'], pos['z']] for pos in reachable_positions])
    xmin, zmin = np.min(reachable_pos, axis=0)
    xmax, zmax = np.max(reachable_pos, axis=0)
    xmin = min(xmin - 0.3, all_obj_xmin)
    xmax = max(xmax + 0.3, all_obj_xmax)
    zmin = min(zmin - 0.3, all_obj_zmin)
    zmax = max(zmax + 0.3, all_obj_zmax)
    bounds = [xmin, xmax, zmin, zmax]
    return bounds

def get_all_category_instances(controller, category):
    instances = []
    for obj in controller.last_event.metadata['objects']:
        if not (obj['objectType']==category):
            continue
        instances.append(obj['objectId'])
    instances = np.array(instances)
    return instances

def format_class_name(name):
    if name=="TVStand":
        formatted = "television stand"
    elif name=="CounterTop":
        formatted = "countertop"
    else:
        formatted = re.sub(r"(?<=\w)([A-Z])", r" \1", name).lower()
    return formatted

def get_images_of_objects(controller, objects, target_names, H,W, do_zoom_in_video=False, do_third_party_image=True):
    '''
    objects: object meta
    target_names: obj['names'] of what you want images of
    '''



    event = controller.step(
                        action="GetReachablePositions"
                    ) #.metadata["actionReturn"]
    nav_pts = event.metadata["actionReturn"]
    nav_pts = np.array([list(d.values()) for d in nav_pts])

    if do_zoom_in_video or do_third_party_image:
        event_test = controller.step(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=0,
            position=dict(x=-1.25, y=1, z=-1),
            rotation=dict(x=90, y=0, z=0),
            fieldOfView=90
        )
        if not event_test.metadata["lastActionSuccess"]:
            third_party_event = controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=-1.25, y=1, z=-1),
                rotation=dict(x=90, y=0, z=0),
                fieldOfView=90
            )

    image_dict = {}
    for obj in objects:
        if obj['name'] not in target_names:
            continue

        obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

        print(f"Getting image for {obj['name']}")
        # print(obj['axisAlignedBoundingBox']['center'])

        dists = np.sqrt(np.sum((nav_pts - obj_center)**2, axis=1))
        argmin_pos = np.argmin(dists)
        closest_pos= nav_pts[argmin_pos] 

        # YAW calculation - rotate to object
        agent_to_obj = np.squeeze(obj_center) - (closest_pos + np.array([0.0, 0.675, 0.0]))
        agent_local_forward = np.array([0, 0, 1.0]) 
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj

        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

        # # add noise so not right in the center
        # noise = np.random.normal(0, 2, size=2)

        turn_yaw = np.degrees(turn_angle) #+ noise[0]

        turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) #+ noise[1]

        event = controller.step('TeleportFull', position=dict(x=closest_pos[0], y=closest_pos[1], z=closest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
        origin_T_camX = get_origin_T_camX(controller.last_event, False)

        if do_zoom_in_video:
            # now move agent out of the way
            argmax_pos = np.argmax(np.sqrt(np.sum((nav_pts - obj_center)**2, axis=1)))
            farthest_pos = nav_pts[argmax_pos] 
            controller.step('TeleportFull', position=dict(x=farthest_pos[0], y=farthest_pos[1], z=farthest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)

            rgbs = []
            fovs = np.flip(np.arange(90,150,5))
            for fov in list(fovs):
                third_party_event = controller.step(
                    action="UpdateThirdPartyCamera",
                    thirdPartyCameraId=0,
                    position=dict(x=closest_pos[0], y=closest_pos[1]+0.675, z=closest_pos[2]),
                    rotation=dict(x=turn_pitch, y=turn_yaw, z=0),
                    fieldOfView=fov,
                )
                rgb = third_party_event.third_party_camera_frames[0]

                hfov = float(fov) * np.pi / 180.
                pix_T_camX = np.array([
                    [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
                    [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
                    [0., 0.,  1, 0],
                    [0., 0., 0, 1]])
                pix_T_camX[0,2] = W/2.
                pix_T_camX[1,2] = H/2.

                obj_3dbox_origin = get_3dbox_in_geom_format(obj)
                # get amodal box
                # origin_T_camX = get_origin_T_camX(controller.last_event, False)
                boxlist2d_amodal, obj_3dbox_camX = get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(pix_T_camX).unsqueeze(0).cuda(), H, W)
                boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
                boxlist2d_amodal[[0,1]] = boxlist2d_amodal[[0,1]] - 10
                boxlist2d_amodal[[2,3]] = boxlist2d_amodal[[2,3]] + 10

                rect_th = 1
                img = rgb.copy()
                cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])),(0, 255, 0), rect_th)
                # plt.figure(1); plt.clf()
                # plt.imshow(img)
                # plt.savefig('images/test.png')
                
                rgbs.append(Image.fromarray(img))
            rgbs[0].save('images/test.gif', save_all=True,optimize=False, append_images=rgbs[1:], duration=400, loop=0)
            st()
        elif do_third_party_image:
            
            # move agent far away
            argmax_pos = np.argmax(np.sqrt(np.sum((nav_pts - obj_center)**2, axis=1)))
            farthest_pos = nav_pts[argmax_pos] 
            controller.step('TeleportFull', position=dict(x=farthest_pos[0], y=farthest_pos[1], z=farthest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)

            pos_visit = [closest_pos]
            select = dists<=1.5
            dists2 = dists[select]
            nav_pts2 = nav_pts[select]
            if len(nav_pts2)==0:
                pos_visit += [closest_pos]
            else:
                argmin_pos = np.argsort(dists2)[len(dists2)//2]
                closest_pos = nav_pts2[argmin_pos]
                pos_visit += [closest_pos]
            select = dists<=3.0
            dists2 = dists[select]
            nav_pts2 = nav_pts[select]
            if len(nav_pts2)==0:
                pos_visit += [closest_pos]
            else:
                argmin_pos = np.argmax(dists2)
                closest_pos = nav_pts2[argmin_pos]
                pos_visit += [closest_pos]
                # pos_visit = pos_visit[[2,1,0]]
            # print("visited:", pos_visit)


            rgbs = []
            for p_i in range(len(pos_visit)):

                closest_pos = pos_visit[p_i]

                # YAW calculation - rotate to object
                agent_to_obj = np.squeeze(obj_center) - (closest_pos + np.array([0.0, 0.675, 0.0]))
                agent_local_forward = np.array([0, 0, 1.0]) 
                flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                flat_to_obj /= flat_dist_to_obj

                det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

                # # add noise so not right in the center
                # noise = np.random.normal(0, 2, size=2)

                turn_yaw = np.degrees(turn_angle) #+ noise[0]

                turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) #+ noise[1]

                controller.step('TeleportFull', position=dict(x=closest_pos[0], y=closest_pos[1], z=closest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
                origin_T_camX = get_origin_T_camX(controller.last_event, False)
                controller.step('TeleportFull', position=dict(x=farthest_pos[0], y=farthest_pos[1], z=farthest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
            
                # rgbs = []
                # fovs = [120, 100, 90]
                # for fov in list(fovs):
                fov = 100
                third_party_event = controller.step(
                    action="UpdateThirdPartyCamera",
                    thirdPartyCameraId=0,
                    position=dict(x=closest_pos[0], y=closest_pos[1]+0.675, z=closest_pos[2]),
                    rotation=dict(x=turn_pitch, y=turn_yaw, z=0),
                    fieldOfView=fov,
                )
                rgb = third_party_event.third_party_camera_frames[0]

                hfov = float(fov) * np.pi / 180.
                pix_T_camX = np.array([
                    [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
                    [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
                    [0., 0.,  1, 0],
                    [0., 0., 0, 1]])
                pix_T_camX[0,2] = W/2.
                pix_T_camX[1,2] = H/2.

                obj_3dbox_origin = get_3dbox_in_geom_format(obj)
                # get amodal box
                # origin_T_camX = get_origin_T_camX(controller.last_event, False)
                boxlist2d_amodal, obj_3dbox_camX = get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(pix_T_camX).unsqueeze(0).cuda(), H, W)
                boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
                boxlist2d_amodal[[0,1]] = boxlist2d_amodal[[0,1]] - 5
                boxlist2d_amodal[[2,3]] = boxlist2d_amodal[[2,3]] + 5

                rect_th = 1
                img = rgb.copy()
                cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])),(0, 255, 0), rect_th)

                img2 = np.zeros((img.shape[0]+5*2, img.shape[1]+5*2, 3)).astype(int)
                for i_i in range(3):
                    img2[:,:,i_i] = np.pad(img[:,:,i_i], pad_width=5, constant_values=255)
                rgbs.append(img2)

            # st()
            img = np.concatenate(rgbs, axis=1)
            # plt.figure(1); plt.clf()
            # plt.imshow(img)
            # plt.savefig('images/test.png')
            # st()

            # np.pad(img, pad_width=5, constant_values=255)
            # plt.figure()
            # plt.imshow(img)
            # plt.savefig('images/test.png')
            # st()
        else:
            
            if not controller.last_event.metadata['lastActionSuccess']:
                print(controller.last_event.metadata["errorMessage"])

            rgb = event.frame

            obj_3dbox_origin = get_3dbox_in_geom_format(obj)
            # get amodal box
            boxlist2d_amodal, obj_3dbox_camX = get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(pix_T_camX).unsqueeze(0).cuda(), H, W)
            boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
            boxlist2d_amodal[[0,1]] = boxlist2d_amodal[[0,1]] - 10
            boxlist2d_amodal[[2,3]] = boxlist2d_amodal[[2,3]] + 10

            rect_th = 1
            img = rgb.copy()
            cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])),(0, 255, 0), rect_th)

            

        name = obj['name']
        if obj['parentReceptacles'] is None:
            receptacle = 'Floor'
        else:
            receptacle = obj['parentReceptacles'][-1]
        image_dict[name] = {}
        image_dict[name]['rgb'] = img
        image_dict[name]['receptacle'] = receptacle.split('|')[0]

    return image_dict

def get_amodal_targets(controller, pix_T_camX, H, W, name_to_id, amodal_boxes, classes_to_save=None, class_agnostic=False):
    origin_T_camX = get_origin_T_camX(controller.last_event, False)
    semantic = controller.last_event.instance_segmentation_frame
    object_id_to_color = controller.last_event.object_id_to_color
    color_to_object_id = controller.last_event.color_to_object_id

    obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)
    
    obj_metadata_IDs = []
    for obj_m in controller.last_event.metadata['objects']: #objects:
        obj_metadata_IDs.append(obj_m['objectId'])

    instance_masks = controller.last_event.instance_masks
    instance_detections2d = controller.last_event.instance_detections2D
    masks = []
    bboxes = []
    labels = []
    instances = []

    for obj_idx_loop in range(obj_ids.shape[0]): # skip target object

        # sometimes this fails?
        try:
            obj_color = tuple(obj_ids[obj_idx_loop])
            object_id = color_to_object_id[obj_color]
        except:
            continue

        if object_id not in obj_metadata_IDs:
            continue

        obj_meta_index = obj_metadata_IDs.index(object_id)
        obj_meta = controller.last_event.metadata['objects'][obj_meta_index]

        obj_category_name = obj_meta['objectType']
        obj_instance_name = obj_meta['objectId']
        
        # print(obj_category_name)
        if classes_to_save is not None:
            if obj_category_name not in classes_to_save:
                continue

        if obj_category_name not in name_to_id:
            continue

        # print(obj_category_name)

        i_mask = instance_masks[object_id]
        num_points = np.sum(i_mask)
        if num_points < 20:
            # print("num points < 20")
            continue
        obj_bbox = instance_detections2d[object_id] #[[0,2,1,3]]
        if obj_bbox[2] - obj_bbox[0] < 5 or obj_bbox[3] - obj_bbox[1] < 5:
            # print("box too small")
            continue
        obj_3dbox_origin = get_3dbox_in_geom_format(obj_meta)
        # get amodal box
        boxlist2d_amodal, obj_3dbox_camX = get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(pix_T_camX).unsqueeze(0).cuda(), H, W)
        boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
        
        boxlist2d_amodal_clip = np.zeros(4)
        boxlist2d_amodal_clip[[0,2]] = np.clip(boxlist2d_amodal[[0,2]], 0, W)
        boxlist2d_amodal_clip[[1,3]] = np.clip(boxlist2d_amodal[[1,3]], 0, H)
        iou_inview = utils.box.boxlist_2d_iou(boxlist2d_amodal.reshape(1,4), boxlist2d_amodal_clip.reshape(1,4))

        inview_iou_threshold = 0.3
        if iou_inview < inview_iou_threshold:
            # not enough of the object in view
            # print("iou thresh broken")
            continue

        # print("all good")

        if amodal_boxes:
            obj_bbox = boxlist2d_amodal_clip
        else:
            obj_bbox = obj_bbox # [start_x, start_y, end_x, end_y]                      

        center_x = ((obj_bbox[0] + obj_bbox[2]) / 2) / W
        center_y = ((obj_bbox[1] + obj_bbox[3]) / 2) / H
        width = (obj_bbox[2] - obj_bbox[0]) / W
        height = (obj_bbox[3] - obj_bbox[1]) / H
        obj_bbox_coco_format = torch.from_numpy(np.array([center_x, center_y, width, height]))

        masks.append(i_mask)
        bboxes.append(obj_bbox_coco_format)
        labels.append(name_to_id[obj_category_name])
        instances.append(obj_instance_name)

    if not bboxes:
        target_frame = {}
        target_frame['boxes'] = torch.tensor([])
        target_frame['labels'] = torch.tensor([])
        target_frame['instance_ids'] = torch.tensor([])
        return target_frame

    # bboxes = torch.as_tensor(np.stack(bboxes), dtype=torch.float32).cuda()
    bboxes = torch.stack(bboxes).cuda()
    if class_agnostic:
        labels = torch.ones((len(labels),), dtype=torch.int64).cuda()
    else:
        labels = torch.as_tensor(labels, dtype=torch.int64).cuda()
    masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)

    num_objs = bboxes.shape[0]

    target_frame = {}
    target_frame['boxes'] = bboxes.float()
    target_frame['labels'] = labels.long()
    target_frame['instance_ids'] = torch.from_numpy(np.arange(len(labels)))
    target_frame['instance_names'] = np.array(instances)

    return target_frame
