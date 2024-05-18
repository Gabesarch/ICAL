import torch
import utils.basic
import utils.box
# import utils.vox
import numpy as np

import ipdb
st = ipdb.set_trace

def eye_2x2(B, device='cuda'):
    rt = torch.eye(2, device=torch.device(device)).view(1,2,2).repeat([B, 1, 1])
    return rt

def eye_3x3(B, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,3,3).repeat([B, 1, 1])
    return rt

def eye_3x3s(B, S, device='cuda'):
    rt = torch.eye(3, device=torch.device(device)).view(1,1,3,3).repeat([B, S, 1, 1])
    return rt

def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def eye_4x4s(B, S, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,1,4,4).repeat([B, S, 1, 1])
    return rt

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt

def merge_rt_single(r, t):
    # r is 3 x 3
    # t is 3
    C, D = list(r.shape)
    assert(C==3)
    assert(D==3)
    t = t.view(3)
    rt = eye_4x4(1).squeeze(0)
    rt[:3,:3] = r
    rt[:3,3] = t
    return rt

def merge_rt_py(r, t):
    # r is B x 3 x 3
    # t is B x 3

    if r is None and t is None:
        assert(False) # you have to provide either r or t
        
    if r is None:
        shape = t.shape
        B = int(shape[0])
        r = np.tile(np.eye(3)[np.newaxis,:,:], (B,1,1))
    elif t is None:
        shape = r.shape
        B = int(shape[0])
        
        t = np.zeros((B, 3))
    else:
        shape = r.shape
        B = int(shape[0])
        
    bottom_row = np.tile(np.reshape(np.array([0.,0.,0.,1.], dtype=np.float32),[1,1,4]),
                         [B,1,1])
    rt = np.concatenate([r,np.expand_dims(t,2)], axis=2)
    rt = np.concatenate([rt,bottom_row], axis=1)
    return rt

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def apply_4x4_py(RT, XYZ):
    # RT is B x 4 x 4
    # XYZ is B x N x 3

    # put into homogeneous coords
    X, Y, Z = np.split(XYZ, 3, axis=2)
    ones = np.ones_like(X)
    XYZ1 = np.concatenate([X, Y, Z, ones], axis=2)
    # XYZ1 is B x N x 4

    XYZ1_t = np.transpose(XYZ1, (0,2,1))
    # this is B x 4 x N

    XYZ2_t = np.matmul(RT, XYZ1_t)
    # this is B x 4 x N
    
    XYZ2 = np.transpose(XYZ2_t, (0,2,1))
    # this is B x N x 4
    
    XYZ2 = XYZ2[:,:,:3]
    # this is B x N x 3
    
    return XYZ2

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
    inv = torch.cat([inv, bottom_row], 0)
    return inv

# def safe_inverse(a):
#     B, _, _ = list(a.shape)
#     inv = torch.zeros(B, 4, 4).cuda()
#     for b in list(range(B)):
#         inv[b] = safe_inverse_single(a[b])
#     return inv

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def get_cami_T_camXs(origin_T_cami, origin_T_camXs):
    B, S = list(origin_T_camXs.shape)[0:2]
    cami_T_camXs = torch.zeros_like(origin_T_camXs)
    cami_T_origin = safe_inverse(origin_T_cami)
    for b in list(range(B)):
        for s in list(range(S)):
            cami_T_camXs[b,s] = torch.matmul(cami_T_origin[b], origin_T_camXs[b,s])
    return cami_T_camXs

def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = pack_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def pack_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=torch.device('cuda'))
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def depth2pointcloud(z, pix_T_cam, device='cuda'):
    B, C, H, W = list(z.shape)
    y, x = utils.basic.meshgrid2d(B, H, W, device=device)
    z = torch.reshape(z, [B, H, W])
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = Pixels2Camera(x, y, z, fx, fy, x0, y0)
    return xyz

def Pixels2Camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth image in meters
    # their shapes are B x H x W
    # fx, fy, x0, y0 are scalar camera intrinsics
    # returns xyz, sized [B,H*W,3]
    
    B, H, W = list(z.shape)

    fx = torch.reshape(fx, [B,1,1])
    fy = torch.reshape(fy, [B,1,1])
    x0 = torch.reshape(x0, [B,1,1])
    y0 = torch.reshape(y0, [B,1,1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    xyz = torch.stack([x,y,z], dim=2)
    return xyz

def pixels2camera(x,y,z,fx,fy,x0,y0):
    # x and y are locations in pixel coordinates, z is a depth in meters
    # they can be images or pointclouds
    # fx, fy, x0, y0 are camera intrinsics
    # returns xyz, sized B x N x 3

    B = x.shape[0]
    
    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])

    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])
    
    # unproject
    x = (z/fx)*(x-x0)
    y = (z/fy)*(y-y0)
    
    xyz = torch.stack([x,y,z], dim=2)
    # B x N x 3
    return xyz

def Camera2Pixels(xyz, pix_T_cam):
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B,1])
    fy = torch.reshape(fy, [B,1])
    x0 = torch.reshape(x0, [B,1])
    y0 = torch.reshape(y0, [B,1])
    x = torch.reshape(x, [B,-1])
    y = torch.reshape(y, [B,-1])
    z = torch.reshape(z, [B,-1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/z + x0
    y = (y*fy)/z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy

def eul2rotm(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = torch.unsqueeze(rx, dim=1)
    ry = torch.unsqueeze(ry, dim=1)
    rz = torch.unsqueeze(rz, dim=1)
    # these are B x 1
    sinz = torch.sin(rz)
    siny = torch.sin(ry)
    sinx = torch.sin(rx)
    cosz = torch.cos(rz)
    cosy = torch.cos(ry)
    cosx = torch.cos(rx)
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=2)
    r2 = torch.stack([r21,r22,r23],dim=2)
    r3 = torch.stack([r31,r32,r33],dim=2)
    r = torch.cat([r1,r2,r3],dim=1)
    return r

def eul2rotm_py(rx, ry, rz):
    # inputs are shaped B
    # this func is copied from matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    rx = rx[:,np.newaxis]
    ry = ry[:,np.newaxis]
    rz = rz[:,np.newaxis]
    # these are B x 1
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
    r1 = np.stack([r11,r12,r13],axis=2)
    r2 = np.stack([r21,r22,r23],axis=2)
    r3 = np.stack([r31,r32,r33],axis=2)
    r = np.concatenate([r1,r2,r3],axis=1)
    return r

def rotm2eul(r):
    # r is Bx3x3, or Bx4x4
    r00 = r[:,0,0]
    r10 = r[:,1,0]
    r11 = r[:,1,1]
    r12 = r[:,1,2]
    r20 = r[:,2,0]
    r21 = r[:,2,1]
    r22 = r[:,2,2]
    
    ## python guide:
    # if sy > 1e-6: # singular
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else:
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0
    
    sy = torch.sqrt(r00*r00 + r10*r10)
    
    cond = (sy > 1e-6)
    rx = torch.where(cond, torch.atan2(r21, r22), torch.atan2(-r12, r11))
    ry = torch.where(cond, torch.atan2(-r20, sy), torch.atan2(-r20, sy))
    rz = torch.where(cond, torch.atan2(r10, r00), torch.zeros_like(r20))

    # rx = torch.atan2(r21, r22)
    # ry = torch.atan2(-r20, sy)
    # rz = torch.atan2(r10, r00)
    # rx[cond] = torch.atan2(-r12, r11)
    # ry[cond] = torch.atan2(-r20, sy)
    # rz[cond] = 0.0
    return rx, ry, rz

def get_random_rt(B,
                  r_amount=5.0,
                  t_amount=1.0,
                  sometimes_zero=False,
                  return_pieces=False):
    # t_amount is in meters
    # r_amount is in degrees
    
    r_amount = np.pi/180.0*r_amount

    ## translation
    tx = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)
    ty = np.random.uniform(-t_amount/2.0, t_amount/2.0, size=B).astype(np.float32)
    tz = np.random.uniform(-t_amount, t_amount, size=B).astype(np.float32)
    
    ## rotation
    rx = np.random.uniform(-r_amount/2.0, r_amount/2.0, size=B).astype(np.float32)
    ry = np.random.uniform(-r_amount, r_amount, size=B).astype(np.float32)
    rz = np.random.uniform(-r_amount/2.0, r_amount/2.0, size=B).astype(np.float32)

    if sometimes_zero:
        rand = np.random.uniform(0.0, 1.0, size=B).astype(np.float32)
        prob_of_zero = 0.5
        rx = np.where(np.greater(rand, prob_of_zero), rx, np.zeros_like(rx))
        ry = np.where(np.greater(rand, prob_of_zero), ry, np.zeros_like(ry))
        rz = np.where(np.greater(rand, prob_of_zero), rz, np.zeros_like(rz))
        tx = np.where(np.greater(rand, prob_of_zero), tx, np.zeros_like(tx))
        ty = np.where(np.greater(rand, prob_of_zero), ty, np.zeros_like(ty))
        tz = np.where(np.greater(rand, prob_of_zero), tz, np.zeros_like(tz))
        
    t = np.stack([tx, ty, tz], axis=1)
    t = torch.from_numpy(t)
    rx = torch.from_numpy(rx)
    ry = torch.from_numpy(ry)
    rz = torch.from_numpy(rz)
    r = eul2rotm(rx, ry, rz)
    rt = merge_rt(r, t).cuda()

    if return_pieces:
        return t.cuda(), rx.cuda(), ry.cuda(), rz.cuda()
    else:
        return rt

def convert_boxlist_to_lrtlist(boxlist):
    B, N, D = list(boxlist.shape)
    assert(D==9)
    boxlist_ = boxlist.view(B*N, D)
    rtlist_ = convert_box_to_ref_T_obj(boxlist_)
    rtlist = rtlist_.view(B, N, 4, 4)
    lenlist = boxlist[:,:,3:6].reshape(B, N, 3)
    lenlist = lenlist.clamp(min=0.01)
    lrtlist = merge_lrtlist(lenlist, rtlist)
    return lrtlist
    
def convert_box_to_ref_T_obj(box3D):
    # turn the box into obj_T_ref (i.e., obj_T_cam)
    B = list(box3D.shape)[0]
    
    # box3D is B x 9
    x, y, z, lx, ly, lz, rx, ry, rz = torch.unbind(box3D, axis=1)
    rot0 = eye_3x3(B)
    tra = torch.stack([x, y, z], axis=1)
    center_T_ref = merge_rt(rot0, -tra)
    # center_T_ref is B x 4 x 4
    
    t0 = torch.zeros([B, 3])
    rot = eul2rotm(rx, ry, rz)
    rot = torch.transpose(rot, 1, 2) # other dir
    obj_T_center = merge_rt(rot, t0)
    # this is B x 4 x 4

    # we want obj_T_ref
    # first we to translate to center,
    # and then rotate around the origin
    obj_T_ref = utils.basic.matmul2(obj_T_center, center_T_ref)

    # return the inverse of this, so that we can transform obj corners into cam coords
    ref_T_obj = obj_T_ref.inverse()
    return ref_T_obj

def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    # frustum/train/provider.py
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=2)
    # ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=2)
    # zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=2)


    xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=2)
    ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=2)
    zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=2)
    
    # these are B x N x 8
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 8 x 3
    return xyzlist

def get_xyzlist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = get_xyzlist_from_lenlist(lenlist)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 8, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)
    return xyzlist_cam

def get_clist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros(B, N, 1, 3).cuda()
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def get_rlist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)
    rlist_, tlist_ = split_rt(__p(rtlist))
    # rlist, tlist = __u(rlist_), __u(tlist_)

    rx_, ry_, rz_ = rotm2eul(rlist_)
    rx, ry, rz = __u(rx_), __u(ry_), __u(rz_)

    # ok now, the funny thing is that these rotations may be wrt the camera origin, not wrt the object
    # so, an object parallel to our car but to the right of us will have a different pose than an object parallel in front

    # maybe that's entirely false

    rlist = torch.stack([rx, ry, rz], dim=2)

    return rlist

def transform_boxes_to_corners_single(boxes, legacy_format=False):
    N, D = list(boxes.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = torch.unbind(boxes, axis=1)
    # these are each shaped N

    ref_T_obj = convert_box_to_ref_T_obj(boxes)
    
    if legacy_format:
        xs = torch.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        ys = torch.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        zs = torch.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
    else:
        xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=1)
        ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=1)
        zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=1)
    
    xyz_obj = torch.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref

def transform_boxes_to_corners(boxes, legacy_format=False):
    # returns corners, shaped B x N x 8 x 3
    B, N, D = list(boxes.shape)
    assert(D==9)
    
    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    boxes_ = __p(boxes)
    corners_ = transform_boxes_to_corners_single(boxes_, legacy_format=legacy_format)
    corners = __u(corners_)
    return corners

def transform_boxes3D_to_corners_py(boxes3D):
    N, D = list(boxes3D.shape)
    assert(D==9)
    
    xc,yc,zc,lx,ly,lz,rx,ry,rz = boxes3D[:,0], boxes3D[:,1], boxes3D[:,2], boxes3D[:,3], boxes3D[:,4], boxes3D[:,5], boxes3D[:,6], boxes3D[:,7], boxes3D[:,8]

    # these are each shaped N

    rotation_mat = eul2rotm_py(rx, ry, rz)
    translation = np.stack([xc, yc, zc], axis=1) 
    ref_T_obj = merge_rt_py(rotation_mat, translation)

    xs = np.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=1)
    ys = np.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=1)
    zs = np.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=1)

    # xs = tf.stack([-lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2.], axis=1)
    # ys = tf.stack([ly/2., -ly/2., ly/2., -ly/2., ly/2., -ly/2., ly/2., -ly/2.], axis=1)
    # zs = tf.stack([-lz/2., -lz/2., -lz/2., -lz/2., lz/2., lz/2., lz/2., lz/2.], axis=1)

    xyz_obj = np.stack([xs, ys, zs], axis=2)
    # centered_box is N x 8 x 3

    xyz_ref = apply_4x4_py(ref_T_obj, xyz_obj)
    # xyz_ref is N x 8 x 3
    return xyz_ref

def apply_pix_T_cam(pix_T_cam, xyz):

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2
    
    B, N, C = list(xyz.shape)
    assert(C==3)
    
    x, y, z = torch.unbind(xyz, axis=-1)

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x*fx)/(z)+x0
    y = (y*fy)/(z)+y0
    xy = torch.stack([x, y], axis=-1)
    return xy

# def apply_4x4_to_boxes(Y_T_X, boxes_X):
#     B, N, C = boxes_X.get_shape().as_list()
#     assert(C==9)
#     corners_X = transform_boxes_to_corners(boxes_X) # corners is B x N x 8 x 3
#     corners_X_ = tf.reshape(corners_X, [B, N*8, 3])
#     corners_Y_ = apply_4x4(Y_T_X, corners_X_)
#     corners_Y = tf.reshape(corners_Y_, [B, N, 8, 3])
#     boxes_Y = corners_to_boxes(corners_Y)
#     return boxes_Y

def apply_4x4_to_corners(Y_T_X, corners_X):
    B, N, C, D = list(corners_X.shape)
    assert(C==8)
    assert(D==3)
    corners_X_ = torch.reshape(corners_X, [B, N*8, 3])
    corners_Y_ = apply_4x4(Y_T_X, corners_X_)
    corners_Y = torch.reshape(corners_Y_, [B, N, 8, 3])
    return corners_Y

def split_lrt(lrt):
    # splits a B x 19 tensor
    # into B x 3 (lens)
    # and B x 4 x 4 (rts)
    B, D = list(lrt.shape)
    assert(D==19)
    lrt = lrt.unsqueeze(1)
    l, rt = split_lrtlist(lrt)
    l = l.squeeze(1)
    rt = rt.squeeze(1)
    return l, rt

def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:,:,3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list

def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert(D==3)
    B2, N2, E, F = list(rtlist.shape)
    assert(B==B2)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist

def merge_lrt(l, rt):
    # l is B x 3
    # rt is B x 4 x 4
    # merges these into a B x 19 tensor
    B, D = list(l.shape)
    assert(D==3)
    B2, E, F = list(rt.shape)
    assert(B==B2)
    assert(E==4 and F==4)
    rt = rt.reshape(B, 16)
    lrt = torch.cat([l, rt], axis=1)
    return lrt

def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrtlist_X)
    # rtlist_X is B x N x 4 x 4

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B*N, 4, 4)
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rtlist_Y_ = utils.basic.matmul2(Y_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def apply_4x4s_to_lrts(Ys_T_Xs, lrt_Xs):
    B, S, D = list(lrt_Xs.shape)
    assert(D==19)
    B2, S2, E, F = list(Ys_T_Xs.shape)
    assert(B2==B)
    assert(S2==S)
    assert(E==4 and F==4)
    
    lenlist, rtlist_X = split_lrtlist(lrt_Xs)
    # rtlist_X is B x N x 4 x 4

    Ys_T_Xs_ = Ys_T_Xs.view(B*S, 4, 4)
    rtlist_X_ = rtlist_X.view(B*S, 4, 4)
    rtlist_Y_ = utils.basic.matmul2(Ys_T_Xs_, rtlist_X_)
    rtlist_Y = rtlist_Y_.view(B, S, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

# import time
# if __name__ == "__main__":
#     input = torch.rand(10, 4, 4).cuda()
#     cur_time = time.time()
#     out_1 = safe_inverse(input)
#     print('time for non-parallel:{}'.format(time.time() - cur_time))

#     print(out_1[0])

#     cur_time = time.time()
#     out_2 = safe_inverse_parallel(input)
#     print('time for parallel:{}'.format(time.time() - cur_time))

#     print(out_2[0])

def create_depth_image_single(xy, z, H, W):
    # turn the xy coordinates into image inds
    xy = torch.round(xy).long()
    depth = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
    
    # lidar reports a sphere of measurements
    # only use the inds that are within the image bounds
    # also, only use forward-pointing depths (z > 0)
    valid = (xy[:,0] <= W-1) & (xy[:,1] <= H-1) & (xy[:,0] >= 0) & (xy[:,1] >= 0) & (z[:] > 0)

    # gather these up
    xy = xy[valid]
    z = z[valid]

    inds = utils.basic.sub2ind(H, W, xy[:,1], xy[:,0]).long()
    depth[inds] = z
    valid = (depth > 0.0).float()
    depth[torch.where(depth == 0.0)] = 100.0
    depth = torch.reshape(depth, [1, H, W])
    valid = torch.reshape(valid, [1, H, W])
    return depth, valid

def create_depth_image(pix_T_cam, xyz_cam, H, W):
    B, N, D = list(xyz_cam.shape)
    assert(D==3)
    xy = apply_pix_T_cam(pix_T_cam, xyz_cam)
    z = xyz_cam[:,:,2]

    depth = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, 1, H, W, dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        depth[b], valid[b] = create_depth_image_single(xy[b], z[b], H, W)
    return depth, valid

def get_iou_from_corresponded_lrtlists(lrtlist_a, lrtlist_b):
    B, N, D = list(lrtlist_a.shape)
    assert(D==19)
    B2, N2, D2 = list(lrtlist_b.shape)
    assert(B2==B, N2==N)
    
    xyzlist_a = get_xyzlist_from_lrtlist(lrtlist_a)
    xyzlist_b = get_xyzlist_from_lrtlist(lrtlist_b)
    # these are B x N x 8 x 3

    xyzlist_a = xyzlist_a.detach().cpu().numpy()
    xyzlist_b = xyzlist_b.detach().cpu().numpy()

    # ious = np.zeros((B, N), np.float32)
    ioulist_3d = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    ioulist_2d = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        for n in list(range(N)):
            iou_3d, iou_2d = utils.box.box3d_iou(xyzlist_a[b,n], xyzlist_b[b,n]+1e-4)
            # print('computed iou %d,%d: %.2f' % (b, n, iou))
            ioulist_3d[b,n] = iou_3d
            ioulist_2d[b,n] = iou_2d
            
    # print('ioulist_3d', ioulist_3d)
    # print('ioulist_2d', ioulist_2d)
    return ioulist_3d, ioulist_2d

def get_centroid_from_box2d(box2d):
    ymin = box2d[:,0]
    xmin = box2d[:,1]
    ymax = box2d[:,2]
    xmax = box2d[:,3]
    x = (xmin+xmax)/2.0
    y = (ymin+ymax)/2.0
    return y, x

def normalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin /= float(H)
    ymax /= float(H)
    xmin /= float(W)
    xmax /= float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def unnormalize_boxlist2d(boxlist2d, H, W):
    boxlist2d = boxlist2d.clone()
    ymin, xmin, ymax, xmax = torch.unbind(boxlist2d, dim=2)
    ymin *= float(H)
    ymax *= float(H)
    xmin *= float(W)
    xmax *= float(W)
    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    return boxlist2d

def get_size_from_box2d(box2d):
    ymin = box2d[:,0]
    xmin = box2d[:,1]
    ymax = box2d[:,2]
    xmax = box2d[:,3]
    height = ymax-ymin
    width = xmax-xmin
    return height, width

def get_box2d_from_centroid_and_size(cy, cx, h, w, clip=True):
    # centroids is B x N x 2
    # dims is B x N x 2
    # both are in normalized coords
    
    ymin = cy - h/2
    ymax = cy + h/2
    xmin = cx - w/2
    xmax = cx + w/2

    box = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    if clip:
        box = torch.clamp(box, 0, 1)
    return box

def convert_box2d_to_intrinsics(box2d, pix_T_cam, H, W, use_image_aspect_ratio=True, mult_padding=1.0):
    # box2d is B x 4, with ymin, xmin, ymax, xmax in normalized coords
    # ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)
    # H, W is the original size of the image
    # mult_padding is relative to object size in pixels

    # i assume we're rendering an image the same size as the original (H, W)

    if not mult_padding==1.0:
        y, x = get_centroid_from_box2d(box2d)
        h, w = get_size_from_box2d(box2d)
        box2d = get_box2d_from_centroid_and_size(
            y, x, h*mult_padding, w*mult_padding, clip=False)
        
    if use_image_aspect_ratio:
        h, w = get_size_from_box2d(box2d)
        y, x = get_centroid_from_box2d(box2d)

        # note h,w are relative right now
        # we need to undo this, to see the real ratio

        h = h*float(H)
        w = w*float(W)
        box_ratio = h/w
        im_ratio = H/float(W)

        # print('box_ratio:', box_ratio)
        # print('im_ratio:', im_ratio)

        if box_ratio >= im_ratio:
            w = h/im_ratio
            # print('setting w:', h/im_ratio)
        else:
            h = w*im_ratio
            # print('setting h:', w*im_ratio)
            
        box2d = get_box2d_from_centroid_and_size(
            y, x, h/float(H), w/float(W), clip=False)

    assert(h > 1e-4)
    assert(w > 1e-4)
        
    ymin, xmin, ymax, xmax = torch.unbind(box2d, dim=1)

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)

    # the topleft of the new image will now have a different offset from the center of projection
    
    new_x0 = x0 - xmin*W
    new_y0 = y0 - ymin*H

    pix_T_cam = pack_intrinsics(fx, fy, new_x0, new_y0)
    # this alone will give me an image in original resolution,
    # with its topleft at the box corner

    box_h, box_w = get_size_from_box2d(box2d)
    # these are normalized, and shaped B. (e.g., [0.4], [0.3])

    # we are going to scale the image by the inverse of this,
    # since we are zooming into this area

    sy = 1./box_h
    sx = 1./box_w

    pix_T_cam = scale_intrinsics(pix_T_cam, sx, sy)
    return pix_T_cam, box2d

def rad2deg(rad):
    return rad*180.0/np.pi

def deg2rad(deg):
    return deg/180.0*np.pi

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return torch.atan2(torch.sin(rad_angle), torch.cos(rad_angle))

def corners_to_boxes(corners, legacy_format=False):
    # corners is B x N x 8 x 3
    B, N, C, D = list(corners.shape)
    assert(C==8)
    assert(D==3)
    assert(legacy_format) # you need to the corners in legacy (non-clockwise) format and acknowledge this
    # do them all at once
    corners_ = corners.reshape(B*N, 8, 3)
    boxes_ = corners_to_boxes_py(corners_.detach().cpu().numpy(), legacy_format=legacy_format)
    boxes_ = torch.from_numpy(boxes_).float().to('cuda')
    # reshape
    boxes = boxes_.reshape(B, N, 9)
    return boxes

def corners_to_boxes_py(corners, legacy_format=False):
    # corners is B x 8 x 3

    assert(legacy_format) # you need to the corners in legacy (non-clockwise) format and acknowledge this
 
    # assert(False) # this function has a flaw; use rigid_transform_boxes instead, or fix it.
    # # i believe you can fix it using what i noticed in rigid_transform_boxes:
    # # if we are looking at the box backwards, the rx/rz dirs flip

    # we want to transform each one to a box
    # note that the rotation may flip 180deg, since corners do not have this info
    
    boxes = []
    for ind, corner_set in enumerate(corners):
        xs = corner_set[:,0]
        ys = corner_set[:,1]
        zs = corner_set[:,2]
        # these are 8 each
        
        xc = np.mean(xs)
        yc = np.mean(ys)
        zc = np.mean(zs)

        # we constructed the corners like this:
        # xs = tf.stack([-lx/2., -lx/2., -lx/2., -lx/2., lx/2., lx/2., lx/2., lx/2.], axis=1)
        # ys = tf.stack([-ly/2., -ly/2., ly/2., ly/2., -ly/2., -ly/2., ly/2., ly/2.], axis=1)
        # zs = tf.stack([-lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2., -lz/2., lz/2.], axis=1)
        # # so we can recover lengths like this:
        # lx = np.linalg.norm(xs[-1] - xs[0])
        # ly = np.linalg.norm(ys[-1] - ys[0])
        # lz = np.linalg.norm(zs[-1] - zs[0])
        # but that's a noisy estimate apparently. let's try all pairs
        
        # rotations are a bit more interesting...

        # defining the corners as: clockwise backcar face, clockwise frontcar face:
        #   E -------- F
        #  /|         /|
        # A -------- B .
        # | |        | |
        # . H -------- G
        # |/         |/
        # D -------- C

        # the ordered eight indices are:
        # A E D H B F C G

        # unstack on first dim
        A, E, D, H, B, F, C, G = corner_set

        back = [A, B, C, D] # back of car is closer to us
        front = [E, F, G, H]
        top = [A, E, B, F]
        bottom = [D, C, H, G]

        front = np.stack(front, axis=0)
        back = np.stack(back, axis=0)
        top = np.stack(top, axis=0)
        bottom = np.stack(bottom, axis=0)
        # these are 4 x 3

        back_z = np.mean(back[:,2])
        front_z = np.mean(front[:,2])
        # usually the front has bigger coords than back
        backwards = not (front_z > back_z)

        front_y = np.mean(front[:,1])
        back_y = np.mean(back[:,1])
        # someetimes the front dips down
        dips_down = front_y > back_y
        
        
        # the bottom should have bigger y coords than the bottom (since y increases down)
        top_y = np.mean(top[:,2])
        bottom_y = np.mean(bottom[:,2])
        upside_down = not (top_y < bottom_y)
        
        # rx: i need anything but x-aligned bars
        # there are 8 of these
        # atan2 wants the y part then the x part; here this means y then z

        x_bars = [[A, B], [D, C], [E, F], [H, G]]
        y_bars = [[A, D], [B, C], [E, H], [F, G]]
        z_bars = [[A, E], [B, F], [D, H], [C, G]]

        lx = 0.0
        for x_bar in x_bars:
            x0, x1 = x_bar
            lx += np.linalg.norm(x1-x0)
        lx /= 4.0
        
        ly = 0.0
        for y_bar in y_bars:
            y0, y1 = y_bar
            ly += np.linalg.norm(y1-y0)
        ly /= 4.0
        
        lz = 0.0
        for z_bar in z_bars:
            z0, z1 = z_bar
            lz += np.linalg.norm(z1-z0)
        lz /= 4.0
        
        # rx = 0.0
        # for pair in [z_bar:
            # rx += np.arctan2(A[1] - E[1], A[2] - E[2])
        # rx = rx / 8.0

        # x: we want atan2(y,z)
        # rx = np.arctan2(A[1] - E[1], A[2] - E[2])
        rx = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            # intermed = np.arctan2(np.abs(pt1[1] - pt2[1]), np.abs(pt1[2] - pt2[2]))
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[2] - pt2[2]))
            rx += intermed
            # if ind==0:
            #     print 'temp rx = %.2f' % intermed
        # for bar in y_bars:
        #     pt1, pt2 = bar
        #     rx += np.arctan2(pt1[1] - pt2[1], pt1[2] - pt2[2])
        # rx /= 8.0
        rx /= 4.0

        ry = 0.0
        for bar in z_bars:
            pt1, pt2 = bar
            # intermed = np.arctan2(np.abs(pt1[2] - pt2[2]), np.abs(pt1[0] - pt2[0]))
            intermed = np.arctan2((pt1[2] - pt2[2]), (pt1[0] - pt2[0]))
            ry += intermed
            # if ind==0:
            #     print 'temp ry = %.2f' % intermed
        # for bar in x_bars:
        #     pt1, pt2 = bar
        #     ry += np.arctan2(pt1[2] - pt2[2], pt1[0] - pt2[0])
        #     if ind==0:
        #         print 'temp ry = %.2f' % np.arctan2(pt1[2] - pt2[2], pt1[0] - pt2[0])
        ry /= 4.0
        
        rz = 0.0
        for bar in x_bars:
            pt1, pt2 = bar
            # intermed = np.arctan2(np.abs(pt1[1] - pt2[1]), np.abs(pt1[0] - pt2[0]))
            intermed = np.arctan2((pt1[1] - pt2[1]), (pt1[0] - pt2[0]))
            rz += intermed
            # if ind==0:
            #     print 'temp rz = %.2f' % intermed
        # for bar in y_bars:
        #     pt1, pt2 = bar
        #     rz += np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
        # rz /= 8.0
        rz /= 4.0


        # # ry: i need anything but y-aligned bars
        # # y: we want atan2(z,x)
        # ry = np.arctan2(A[2] - E[2], A[0] - E[0])

        # rz: anything but z-aligned bars
        # z: we want atan2(y,x)
        # rz = np.arctan2(A[1] - B[1], A[0] - B[0])

        ry += np.pi/2.0

        # handle axis flips
            
        # if ind==0 or ind==1:
        #     # print 'rx = %.2f' % rx
        #     # print 'ry = %.2f' % ry
        #     # print 'rz = %.2f' % rz
        #     print 'rx = %.2f; ry = %.2f; rz = %.2f, backwards = %s; dips_down = %s, front %.2f, back %.2f, upside_down = %s' % (
        #         rx, ry, rz, backwards, dips_down,
        #         front_y, back_y, upside_down)
        if backwards:
            ry = -ry
        if not backwards:
            ry = ry - np.pi

        # rx = 0.0
        # rz = 0.0
        
        #     # rx = rx - np.pi
        #     rz = -rz 
           
        # if np.abs(rz) > np.pi/2.0:
        #     # rx = -rx
        #     rx = wrap2halfpi_single_py(rx)
        #     rz = wrap2halfpi_single_py(rz)

        # # hack
        # if np.abs(ry) < np.pi/2.0:
        #     rx = -rx

        
        #     rx = rx - np.pi
        # else:
        #     ry = ry - np.pi
        # # rx = -rx
        # if dips_down:
        #     rx = -rx
            
            # ry = -(ry - np.pi)
            # ry = -(ry - np.pi)
            # ry = -(ry - np.pi)
        # ry = wrap2pi_py(ry)
        #     if not dips_down:
        #         rx = -rx
        # if dips_down and not backwards:
        #     rx = -rx
        # if dips_down:
        #     rx = -rx
            
            # rx = -rx
            # rz = -rz
        # if backwards_x:
        #     rz = -rz
            
        box = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
        boxes.append(box)
    boxes = np.stack(boxes, axis=0).astype(np.float32)
    return boxes
    
    
def corners_to_box3D_single_py(corners):
    # corners is N x 8 x 3

    # boxes_new, tids_new, scores_new = tf.py_function(sink_invalid_boxes_py, (boxes, tids, scores),
    #                                                  (tf.float32, tf.int32, tf.float32))
    
    
    # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry or rz
    if coordinate == 'lidar':
        for idx in list(range(len(boxes_corner))):
            boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx], rect_T_cam, cam_T_velo)
    ret = []

    
    for roi in boxes_corner:
        roi = np.array(roi)
        h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4.0)
        w = np.sum(
            np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
        ) / 4
        l = np.sum(
            np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
            np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
        ) / 4
        x = np.sum(roi[:, 0], axis=0) / 8.0
        y = np.sum(roi[0:4, 1], axis=0) / 4.0
        z = np.sum(roi[:, 2], axis=0) / 8.0
        ry = np.sum(
            np.arctan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
            np.arctan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
            np.arctan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
            np.arctan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
            np.arctan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
            np.arctan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
            np.arctan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
            np.arctan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
        ) / 8.0
        if w > l:
            w, l = l, w
            ry = angle_in_limit(ry + np.pi / 2.0)
        ret.append([x, y, z, h, w, l, ry])

    return np.array(ret).astype(np.float32)
    
def inflate_to_axis_aligned_boxlist(boxlist):
    B, N, D = list(boxlist.shape)
    assert(D==9)

    corners = transform_boxes_to_corners(boxlist) # corners is B x N x 8 x 3
    corners_max = torch.max(corners, dim=2)[0]
    corners_min = torch.min(corners, dim=2)[0]

    centers = (corners_max + corners_min)/2.0
    sizes = corners_max - corners_min
    rots = torch.zeros_like(sizes)

    # xc, yc, zc, lx, ly, lz, rx, ry, rz
    boxlist_norot = torch.cat([centers, sizes, rots], dim=2)
    # boxlist_norot is B x N x 9

    return boxlist_norot

def depthrt2flow(depth_cam0, cam1_T_cam0, pix_T_cam):
    B, C, H, W = list(depth_cam0.shape)
    assert(C==1)

    # get the two pointclouds
    xyz_cam0 = depth2pointcloud(depth_cam0, pix_T_cam)
    xyz_cam1 = apply_4x4(cam1_T_cam0, xyz_cam0)

    # project, and get 2d flow
    flow = pointcloud2flow(xyz_cam1, pix_T_cam, H, W)
    return flow

def pointcloud2flow(xyz1, pix_T_cam, H, W):
    # project xyz1 down, so that we get the 2d location of all of these pixels,
    # then subtract these 2d locations from the original ones to get optical flow
    
    B, N, C = list(xyz1.shape)
    assert(N==H*W)
    assert(C==3)
    
    # we assume xyz1 is the unprojection of the regular grid
    grid_y0, grid_x0 = utils.basic.meshgrid2d(B, H, W)

    xy1 = Camera2Pixels(xyz1, pix_T_cam)
    x1, y1 = torch.unbind(xy1, dim=2)
    x1 = x1.reshape(B, H, W)
    y1 = y1.reshape(B, H, W)

    flow_x = x1 - grid_x0
    flow_y = y1 - grid_y0
    flow = torch.stack([flow_x, flow_y], axis=1)
    # flow is B x 2 x H x W
    return flow

def get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_cam, H, W):
    B, N, D = list(lrtlist_cam.shape)
    assert(D==19)
    corners_cam = get_xyzlist_from_lrtlist(lrtlist_cam)
    # this is B x N x 8 x 3
    corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])

    xmin = torch.min(corners_pix[:,:,:,0], dim=2)[0]
    xmax = torch.max(corners_pix[:,:,:,0], dim=2)[0]
    ymin = torch.min(corners_pix[:,:,:,1], dim=2)[0]
    ymax = torch.max(corners_pix[:,:,:,1], dim=2)[0]
    # these are B x N

    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    boxlist2d = normalize_boxlist2d(boxlist2d, H, W)
    return boxlist2d

def get_boxlist2d_from_corners(pix_T_cam, corners_cam, H, W):
    B, N, C, D = list(corners_cam.shape)
    assert(C==8)
    assert(D==3)
    # this is B x N x 8 x 3
    corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])

    xmin = torch.min(corners_pix[:,:,:,0], dim=2)[0]
    xmax = torch.max(corners_pix[:,:,:,0], dim=2)[0]
    ymin = torch.min(corners_pix[:,:,:,1], dim=2)[0]
    ymax = torch.max(corners_pix[:,:,:,1], dim=2)[0]
    # these are B x N

    boxlist2d = torch.stack([ymin, xmin, ymax, xmax], dim=2)
    # boxlist2d = normalize_boxlist2d(boxlist2d, H, W)
    return boxlist2d

def get_boxlist2d_from_corners_aithor(pix_T_cam, corners_cam, H, W):
    B, N, C, D = list(corners_cam.shape)
    assert(C==8)
    assert(D==3)
    # this is B x N x 8 x 3
    corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
    corners_pix_ = apply_pix_T_cam(pix_T_cam, corners_cam_)
    corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
    
    # need this for aithor
    corners_pix[:,:,:,1] = H - corners_pix[:,:,:,1]

    xmin = torch.min(corners_pix[:,:,:,0], dim=2)[0]
    xmax = torch.max(corners_pix[:,:,:,0], dim=2)[0]
    ymin = torch.min(corners_pix[:,:,:,1], dim=2)[0]
    ymax = torch.max(corners_pix[:,:,:,1], dim=2)[0]
    # these are B x N

    boxlist2d = torch.stack([xmin, ymin, xmax, ymax], dim=2)
    # boxlist2d = normalize_boxlist2d(boxlist2d, H, W)
    return boxlist2d
    
def sincos_norm(sin, cos):
    both = torch.stack([sin, cos], dim=-1)
    both = utils.basic.l2_normalize(both, dim=-1)
    sin, cos = torch.unbind(both, dim=-1)
    return sin, cos
                
def sincos2rotm(sinz, siny, sinx, cosz, cosy, cosx):
    # copy of matlab
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
    #        -sy            cy*sx             cy*cx]
    r11 = cosy*cosz
    r12 = sinx*siny*cosz - cosx*sinz
    r13 = cosx*siny*cosz + sinx*sinz
    r21 = cosy*sinz
    r22 = sinx*siny*sinz + cosx*cosz
    r23 = cosx*siny*sinz - sinx*cosz
    r31 = -siny
    r32 = sinx*cosy
    r33 = cosx*cosy
    r1 = torch.stack([r11,r12,r13],dim=-1)
    r2 = torch.stack([r21,r22,r23],dim=-1)
    r3 = torch.stack([r31,r32,r33],dim=-1)
    r = torch.stack([r1,r2,r3],dim=-2)
    return r
                                                        
def convert_clist_to_lrtlist(clist, len0, angle0=None, smooth=3):
    B, S, D = list(clist.shape)
    B, E = list(len0.shape)
    assert(D==3)
    assert(E==3)

    boxlist = torch.zeros(B, S, 9).float().cuda()
    for s in list(range(S)):
        s_a = max(s-smooth, 0)
        s_b = min(s+smooth, S)
        xyz0 = torch.mean(clist[:,s_a:s+1], dim=1)
        xyz1 = torch.mean(clist[:,s:s_b+1], dim=1)

        delta = xyz1-xyz0
        delta_norm = torch.norm(delta, dim=1)
        invalid_NY = delta_norm < 0.0001

        if invalid_NY.sum() > 0:
            assert(False) # this shouldn't really happen
        #     import pdb; pdb.set_trace()

        delta = delta.detach().cpu().numpy()
        dx = delta[:,0]
        dy = delta[:,1]
        dz = delta[:,2]
        yaw = -np.arctan2(dz, dx) + np.pi/2.0

        # if s==0:
        #     print('input angle[:,1]', angle0[:,1])
        #     print('raw arctan', np.arctan2(dz, dx))
        #     print('subbed yaw on step %d:' % s, yaw)

        yaw = torch.from_numpy(yaw).float().cuda()

        # # the car must begin straight, but we quickly allow it to curve
        # coeff = 1.0-np.exp(-s) # this is 0 on the first iter, and quickly increases to 1
        # yaw = coeff*yaw + (1.0-coeff)*angle0[:,1]

        zero = torch.zeros_like(yaw)
        angles = torch.stack([zero, yaw, zero], dim=1)
        # angles = torch.stack([angle0[:,0], yaw, angle0[:,2]], dim=1)
        # this is B x 3

        boxlist[:,s] = torch.cat([clist[:,s], len0, angles], dim=1)

    lrtlist = convert_boxlist_to_lrtlist(boxlist)
    return lrtlist

def angular_l1_norm(e, g, dim=1, keepdim=False):
    # inputs are shaped B x N
    # returns a tensor sized B x N, with the dist in every slot
    
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    # now our angles are in [0, 360]
    l = torch.abs(np.pi - torch.abs(torch.abs(e-g) - np.pi))
    return torch.sum(l, dim=dim, keepdim=keepdim)

def angular_l1_dist(e, g):
    # inputs are shaped B x N
    # returns a tensor sized B x N, with the dist in every slot
    
    # if our angles are in [0, 360] we can follow this stack overflow answer:
    # https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    # wrap2pi brings the angles to [-180, 180]; adding pi puts them in [0, 360]
    e = wrap2pi(e)+np.pi
    g = wrap2pi(g)+np.pi
    # now our angles are in [0, 360]
    l = torch.abs(np.pi - torch.abs(torch.abs(e-g) - np.pi))
    return l

def get_image_inbounds(pix_T_cam, xyz_cam, H, W, padding=0.0):
    # pix_T_cam is B x 4 x 4
    # xyz_cam is B x N x 3
    # padding should be 0 unless you are trying to account for some later cropping
    
    xy_pix = utils.geom.apply_pix_T_cam(pix_T_cam, xyz_cam)

    x = xy_pix[:,:,0]
    y = xy_pix[:,:,1]
    z = xyz_cam[:,:,2]

    # print('x', x.detach().cpu().numpy())
    # print('y', y.detach().cpu().numpy())
    # print('z', z.detach().cpu().numpy())

    x_valid = ((x-padding)>-0.5).byte() & ((x+padding)<float(W-0.5)).byte()
    y_valid = ((y-padding)>-0.5).byte() & ((y+padding)<float(H-0.5)).byte()
    z_valid = ((z>0.0)).byte()

    inbounds = x_valid & y_valid & z_valid
    return inbounds.bool()

def get_point_correspondence_from_flow(xyz0, xyz1, flow_f, pix_T_cam, H, W, flow_valid=None):
    # flow_f is the forward flow, from frame0 to frame1
    # xyz0 and xyz1 are pointclouds, in cam coords
    # we want to get a new xyz1, with points that correspond to xyz0
    B, N, D = list(xyz0.shape)

    # discard depths that are beyond this distance, since they are probably fake
    max_dist = 200.0
    
    # now sample the 2d flow vectors at the xyz0 locations
    # ah wait!:
    # it's important here to only use positions in front of the camera
    xy = apply_pix_T_cam(pix_T_cam, xyz0)
    z0 = xyz0[:, :, 2] # B x N
    x0 = xy[:, :, 0] # B x N
    y0 = xy[:, :, 1] # B x N
    uv = utils.samp.bilinear_sample2d(flow_f, x0, y0) # B x 2 x N

    frustum0_valid = get_image_inbounds(pix_T_cam, xyz0, H, W)

    # next we want to use these to sample into the depth of the next frame 
    # depth0, valid0 = create_depth_image(pix_T_cam, xyz0, H, W)
    depth1, valid1 = create_depth_image(pix_T_cam, xyz1, H, W)
    # valid0 = valid0 * (depth0 < max_dist).float()
    valid1 = valid1 * (depth1 < max_dist).float()
    
    u = uv[:, 0] # B x N
    v = uv[:, 1] # B x N
    x1 = x0 + u
    y1 = y0 + v

    # round to the nearest pixel, since the depth image has holes
    # x0 = torch.clamp(torch.round(x0), 0, W-1).long()
    # y0 = torch.clamp(torch.round(y0), 0, H-1).long()
    x1 = torch.clamp(torch.round(x1), 0, W-1).long()
    y1 = torch.clamp(torch.round(y1), 0, H-1).long()
    z1 = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    valid = torch.zeros(B, N, dtype=torch.float32, device=torch.device('cuda'))
    # since we rounded and clamped, we can index directly, instead of bilinear sampling

    for b in range(B):
        # depth0_b = depth0[b] # 1 x H x W
        # valid0_b = valid0[b]
        # valid0_b_ = valid0_b[0, y0[b], x0[b]] # N
        # z0_b_ = depth0_b[0, y1[b], x1[b]] # N
        
        depth1_b = depth1[b] # 1 x H x W
        valid1_b = valid1[b]
        valid1_b_ = valid1_b[0, y1[b], x1[b]] # N
        z1_b_ = depth1_b[0, y1[b], x1[b]] # N
        
        z1[b] = z1_b_
        # valid[b] = valid0_b_ * valid1_b_ * frustum0_valid[b]
        valid[b] = valid1_b_ * frustum0_valid[b]

        if flow_valid is not None:
            validf_b = flow_valid[b]
            validf_b_ = valid1_b[0, y1[b], x1[b]] # N
            valid[b] = valid[b] * validf_b_

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz1 = pixels2camera(x1,y1,z1,fx,fy,x0,y0)
    # xyz1 is B x N x 3
    # valid is B x N
    return xyz1, valid
