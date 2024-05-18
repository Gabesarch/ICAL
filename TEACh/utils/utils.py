import numpy as np

import ipdb
st = ipdb.set_trace

import utils.geom
import torch

def get_pointcloud(observations, fov):

    depth = observations['depth']

    H = depth.shape[0]
    W = depth.shape[1]

    hfov = float(fov) * np.pi / 180.
    K = np.array([
        [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0.,  1, 0],
        [0., 0., 0, 1]])
    
    # get pointcloud
    xs, ys = np.meshgrid(np.linspace(-1*H/2.,1*H/2.,H), np.linspace(1*W/2.,-1*W/2., W))
    depth = depth.reshape(1,H,W)
    xs = xs.reshape(1,H,W)
    ys = ys.reshape(1,H,W)
    xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)
    xyz = xy_c0.T[:,:3].reshape(H,W,3)
    xyz = xyz.reshape(-1, 3)

    return xyz

def eul2rotm(rx, ry, rz):
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

def undo_rotation_translation(xyz, rel_pos_tx_t0):

    # translate = np.array([-rel_pos_tx_t0[0], 0.0, -rel_pos_tx_t0[1]])
    # xyz = xyz + translate

    rx = 0.0
    ry = np.radians(rel_pos_tx_t0[2]) #0.0 #eulers_xyz_rad[1]
    rz = 0.0 #eulers_xyz_rad[2]
    rotation_r_matrix = eul2rotm(rx, ry, rz)
    xyz = (np.linalg.inv(rotation_r_matrix) @ xyz.T).T 

    if rel_pos_tx_t0[0]!=0.:
        tx = -rel_pos_tx_t0[0]
    else:
        tx = 0.0
    if rel_pos_tx_t0[1]!=0.:
        tz = -rel_pos_tx_t0[1]
    else:
        tz = 0.0
    # direction = np.array([tx, 0.0, tz])
    # M = np.identity(4)
    # M[:3, 3] = direction[:3]
    # M[:3, :3] = rotation_r_matrix

    # M = torch.from_numpy(M)
    # M_inv = utils.geom.safe_inverse_single(M)

    # xyz = torch.from_numpy(xyz)
    # xyz = utils.geom.apply_4x4(M_inv, xyz.unsqueeze(0))

    # xyz = xyz.squeeze(0).cpu().numpy()

    translate = np.array([tx, 0.0, tz])
    xyz = xyz + translate


    # xyz = torch.cat([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    # xyz = (M @ xyz.T).T
    # xyz = xyz[:,:3]

    return xyz
