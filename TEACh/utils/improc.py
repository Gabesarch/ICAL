# import tensorflow as tf
import torch
import torchvision.transforms
import cv2
import os
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
from matplotlib import cm
# import hyperparams as hyp
import matplotlib
import imageio
from itertools import combinations
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
import utils.geom
import utils.py
import utils.basic
import utils.dist
from collections import defaultdict, deque

import io
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import datetime

import torch.nn.functional as F 

import ipdb
st = ipdb.set_trace

EPS = 1e-6

# color conversion libs, for flow vis
from skimage.color import (
    rgb2lab, rgb2yuv, rgb2ycbcr, lab2rgb, yuv2rgb, ycbcr2rgb,
    rgb2hsv, hsv2rgb, rgb2xyz, xyz2rgb, rgb2hed, hed2rgb)

def _convert(input_, type_):
    return {
        'float': input_.float(),
        'double': input_.double(),
    }.get(type_, input_)


def _generic_transform_sk_4d(transform, in_type='', out_type=''):
    def apply_transform(input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        if to_squeeze:
            output = output.squeeze(0)
        output = _convert(output, out_type)
        return output.to(device)
    return apply_transform


def _generic_transform_sk_3d(transform, in_type='', out_type=''):
    def apply_transform_individual(input_):
        device = input_.device
        input_ = input_.cpu()
        input_ = _convert(input_, in_type)

        input_ = input_.permute(1, 2, 0).detach().numpy()
        transformed = transform(input_)
        output = torch.from_numpy(transformed).float().permute(2, 0, 1)
        output = _convert(output, out_type)
        return output.to(device)

    def apply_transform(input_):
        to_stack = []
        for image in input_:
            to_stack.append(apply_transform_individual(image))
        return torch.stack(to_stack)
    return apply_transform


# # --- Cie*LAB ---
rgb_to_lab = _generic_transform_sk_4d(rgb2lab)
lab_to_rgb = _generic_transform_sk_3d(lab2rgb, in_type='double', out_type='float')
# # --- YUV ---
# rgb_to_yuv = _generic_transform_sk_4d(rgb2yuv)
# yuv_to_rgb = _generic_transform_sk_4d(yuv2rgb)
# # --- YCbCr ---
# rgb_to_ycbcr = _generic_transform_sk_4d(rgb2ycbcr)
# ycbcr_to_rgb = _generic_transform_sk_4d(ycbcr2rgb, in_type='double', out_type='float')
# # --- HSV ---
# rgb_to_hsv = _generic_transform_sk_3d(rgb2hsv)
hsv_to_rgb = _generic_transform_sk_3d(hsv2rgb)
# # --- XYZ ---
# rgb_to_xyz = _generic_transform_sk_4d(rgb2xyz)
# xyz_to_rgb = _generic_transform_sk_3d(xyz2rgb, in_type='double', out_type='float')
# # --- HED ---
# rgb_to_hed = _generic_transform_sk_4d(rgb2hed)
# hed_to_rgb = _generic_transform_sk_3d(hed2rgb, in_type='double', out_type='float')

'''end color conversion in torch'''

def rgb2lab(rgb):
    # rgb is in -0.5,0.5
    rgb = rgb + 0.5 # put it into [0,1] for my tool
    lab = rgb_to_lab(rgb) # this is in -100,100
    lab = lab / 100.0 # this is in -1,1
    return lab
def lab2rgb(lab):
    # lab is in -1,1
    lab = lab * 100.0 # this is in -100,100
    rgb = lab_to_rgb(lab) # this is in [0,1]
    rgb = rgb - 0.5 # this is in -0.5,0.5
    return rgb

def preprocess_color_tf(x):
    import tensorflow as tf
    return tf.cast(x,tf.float32) * 1./255 - 0.5

def preprocess_color(x):
    return x.float() * 1./255 - 0.5

def pca_embed(emb, keep, valid=None):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + EPS
    #emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

    if valid:
        valid = valid.cpu().detach().numpy().reshape((H*W))

    emb_reduced = list()

    B, H, W, C = np.shape(emb)
    for img in emb:
        if np.isnan(img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        pixels_kd = np.reshape(img, (H*W, C))
        
        if valid:
            pixels_kd_pca = pixels_kd[valid]
        else:
            pixels_kd_pca = pixels_kd

        P = PCA(keep)
        P.fit(pixels_kd_pca)

        if valid:
            pixels3d = P.transform(pixels_kd)*valid
        else:
            pixels3d = P.transform(pixels_kd)

        out_img = np.reshape(pixels3d, [H,W,keep]).astype(np.float32)
        if np.isnan(out_img).any():
            emb_reduced.append(np.zeros([H, W, keep]))
            continue

        emb_reduced.append(out_img)

    emb_reduced = np.stack(emb_reduced, axis=0).astype(np.float32)

    return torch.from_numpy(emb_reduced).permute(0, 3, 1, 2)

def pca_embed_together(emb, keep):
    ## emb -- [S,H/2,W/2,C]
    ## keep is the number of principal components to keep
    ## Helper function for reduce_emb.
    emb = emb + EPS
    #emb is B x C x H x W
    emb = emb.permute(0, 2, 3, 1).cpu().detach().numpy() #this is B x H x W x C

    B, H, W, C = np.shape(emb)
    if np.isnan(emb).any():
        out_img = torch.zeros(B, keep, H, W)

    pixelskd = np.reshape(emb, (B*H*W, C))
    P = PCA(keep)
    P.fit(pixelskd)
    pixels3d = P.transform(pixelskd)
    out_img = np.reshape(pixels3d, [B,H,W,keep]).astype(np.float32)
    if np.isnan(out_img).any():
        out_img = torch.zeros(B, keep, H, W)
    return torch.from_numpy(out_img).permute(0, 3, 1, 2)

def reduce_emb(emb, valid=None, inbound=None, together=False):
    ## emb -- [S,C,H/2,W/2], inbound -- [S,1,H/2,W/2]
    ## Reduce number of chans to 3 with PCA. For vis.
    # S,H,W,C = emb.shape.as_list()
    S, C, H, W = list(emb.size())
    keep = 3

    if together:
        reduced_emb = pca_embed_together(emb, keep)
    else:
        reduced_emb = pca_embed(emb, keep, valid) #not im

    reduced_emb = utils.basic.normalize(reduced_emb) - 0.5
    if inbound is not None:
        emb_inbound = emb*inbound
    else:
        emb_inbound = None

    return reduced_emb, emb_inbound

def get_feat_pca(feat, valid=None):
    B, C, D, W = list(feat.size())
    # feat is B x C x D x W. If 3D input, average it through Height dimension before passing into this function.

    pca, _ = reduce_emb(feat, valid=valid,inbound=None, together=True)
    # pca is B x 3 x W x D
    return pca

def convert_occ_to_height(occ, reduce_axis=3):
    B, C, D, H, W = list(occ.shape)
    assert(C==1)
    # note that height increases DOWNWARD in the tensor
    # (like pixel/camera coordinates)
    
    G = list(occ.shape)[reduce_axis]
    values = torch.linspace(float(G), 1.0, steps=G).type(torch.FloatTensor).cuda()
    if reduce_axis==2:
        # frontal view
        values = values.view(1, 1, G, 1, 1)
    elif reduce_axis==3:
        # top view
        values = values.view(1, 1, 1, G, 1)
    elif reduce_axis==4:
        # lateral view
        values = values.view(1, 1, 1, 1, G)
    else:
        assert(False) # you have to reduce one of the spatial dims (2-4)
    values = torch.max(occ*values, dim=reduce_axis)[0]/float(G)
    # values = values.view([B, C, D, W])
    return values

def gif_and_tile(ims, just_gif=False):
    S = len(ims) 
    # ims is S X B X H X W X C
    # i want a gif in the left, and the tiled frames on the right
    # for the gif tool, this means making a B x S x H x W tensor
    # where the leftmost part is sequential and the rest is tiled
    gif = torch.stack(ims, dim=1)
    if just_gif:
        return gif
    til = torch.cat(ims, dim=2)
    til = til.unsqueeze(dim=1).repeat(1, S, 1, 1, 1)
    im = torch.cat([gif, til], dim=3)
    return im

def back2color(i, blacken_zeros=False):
    if blacken_zeros:
        const = torch.tensor([-0.5])
        i = torch.where(i==0.0, const.cuda() if i.is_cuda else const, i)
        return back2color(i)
    else:
        return ((i+0.5)*255).type(torch.ByteTensor)

def colorize(d):
    # this does not work properly yet
    
    # # d is C x H x W or H x W
    # if d.ndim==3:
    #     d = d.squeeze(dim=0)
    # else:
    #     assert(d.ndim==2)

    if d.ndim==2:
        d = d.unsqueeze(dim=0)
    else:
        assert(d.ndim==3)
    # copy to the three chans
    d = d.repeat(3, 1, 1)
    return d
    
    # d = d.cpu().detach().numpy()
    # # move channels out to last dim
    # # d = np.transpose(d, [0, 2, 3, 1])
    # # d = np.transpose(d, [1, 2, 0])
    # print(d.shape)
    # d = cm.inferno(d)[:, :, 1:] # delete the alpha channel
    # # move channels into dim0
    # d = np.transpose(d, [2, 0, 1])
    # print_stats(d, 'colorize_out')
    # d = torch.from_numpy(d)
    # return d

def seq2color(im, norm=True, colormap='coolwarm'):
    B, S, H, W = list(im.shape)
    # S is sequential

    # prep a mask of the valid pixels, so we can blacken the invalids later
    mask = torch.max(im, dim=1, keepdim=True)[0]

    # turn the S dim into an explicit sequence
    coeffs = np.linspace(1.0, float(S), S).astype(np.float32)/float(S)
    
    # # increase the spacing from the center
    # coeffs[:int(S/2)] -= 2.0
    # coeffs[int(S/2)+1:] += 2.0
    
    coeffs = torch.from_numpy(coeffs).float().cuda()
    coeffs = coeffs.reshape(1, S, 1, 1).repeat(B, 1, H, W)
    # scale each channel by the right coeff
    im = im * coeffs
    # now im is in [1, S], except for the invalid parts which are 0
    # keep the highest valid coeff at each pixel
    im = torch.max(im, dim=1, keepdim=True)[0]

    # # note the range here is -2 to S+2, since we shifted away from the center
    # rgb = colorize(d, vmin=0.0-2.0, vmax=float(S+2.0), vals=255, cmap=colormap)

    out = []
    for b in range(B):
        im_ = im[b]
        # move channels out to last dim_
        im_ = im_.detach().cpu().numpy()
        im_ = np.squeeze(im_)
        # im_ is H x W
        if colormap=='coolwarm':
            im_ = cm.coolwarm(im_)[:, :, :3]
        elif colormap=='PiYG':
            im_ = cm.PiYG(im_)[:, :, :3]
        elif colormap=='winter':
            im_ = cm.winter(im_)[:, :, :3]
        else:
            assert(False) # invalid colormap
        # move channels into dim 0
        im_ = np.transpose(im_, [2, 0, 1])
        im_ = torch.from_numpy(im_).float().cuda()
        out.append(im_)
    out = torch.stack(out, dim=0)
    
    # blacken the invalid pixels, instead of using the 0-color
    out = out*mask
    # out = out*255.0

    # put it in [-0.5, 0.5]
    out = out - 0.5
    
    return out

def oned2inferno(d, norm=True):
    # convert a 1chan input to a 3chan image output

    # if it's just B x H x W, add a C dim
    if d.ndim==3:
        d = d.unsqueeze(dim=1)
    # d should be B x C x H x W, where C=1
    B, C, H, W = list(d.shape)
    assert(C==1)

    if norm:
        d = utils.basic.normalize(d)
        
    rgb = torch.zeros(B, 3, H, W)
    for b in list(range(B)):
        rgb[b] = colorize(d[b])

    rgb = (255.0*rgb).type(torch.ByteTensor)

    # rgb = tf.cast(255.0*rgb, tf.uint8)
    # rgb = tf.reshape(rgb, [-1, hyp.H, hyp.W, 3])
    # rgb = tf.expand_dims(rgb, axis=0)
    return rgb

def xy2mask(xy, H, W, norm=False):
    # xy is B x N x 2, in either pixel coords or normalized coordinates (depending on norm)
    # proto is B x H x W x 1, showing how big to make the mask
    # returns a mask the same size as proto, with a 1 at each specified xy
    B = list(xy.shape)[0]
    if norm:
        # convert to pixel coords
        x, y = torch.unbind(xy, axis=2)
        x = x*float(W)
        y = y*float(H)
        xy = torch.stack(xy, axis=2)
        
    mask = torch.zeros([B, 1, H, W], dtype=torch.float32, device=torch.device('cuda'))
    for b in list(range(B)):
        mask[b] = xy2mask_single(xy[b], H, W)
    return mask

def xy2mask_single(xy, H, W):
    # xy is N x 2
    x, y = torch.unbind(xy, axis=1)
    x = x.long()
    y = y.long()

    x = torch.clamp(x, 0, W-1)
    y = torch.clamp(y, 0, H-1)
    
    inds = utils.basic.sub2ind(H, W, y, x)

    valid = (inds > 0).byte() & (inds < H*W).byte()
    inds = inds[torch.where(valid)]

    mask = torch.zeros(H*W, dtype=torch.float32, device=torch.device('cuda'))
    mask[inds] = 1.0
    mask = torch.reshape(mask, [1,H,W])
    return mask

def xy2heatmap(xy, sigma, grid_xs, grid_ys, norm=False):
    # xy is B x N x 2, containing float x and y coordinates of N things
    # grid_xs and grid_ys are B x N x Y x X

    B, N, Y, X = list(grid_xs.shape)
    
    mu_x = xy[:,:,0]
    mu_y = xy[:,:,1]

    x_valid = (mu_x>-0.5).byte() & (mu_x<float(X-0.5)).byte() & (mu_x!=0.0).byte()
    y_valid = (mu_y>-0.5).byte() & (mu_y<float(Y-0.5)).byte() & (mu_y!=0.0).byte()
    not_valid = 1-(x_valid & y_valid)
    
    mu_x[not_valid] = -10000
    mu_y[not_valid] = -10000
    
    mu_x = mu_x.reshape(B, N, 1, 1).repeat(1, 1, Y, X)
    mu_y = mu_y.reshape(B, N, 1, 1).repeat(1, 1, Y, X)
    
    sigma_sq = sigma*sigma
    sq_diff_x = (grid_xs - mu_x)**2
    sq_diff_y = (grid_ys - mu_y)**2

    term1 = 1./2.*np.pi*sigma_sq
    term2 = torch.exp(-(sq_diff_x+sq_diff_y)/(2.*sigma_sq))
    gauss = term1*term2

    if norm:
        # normalize so each gaussian peaks at 1
        gauss_ = gauss.reshape(B*N, Y, X)
        gauss_ = utils.basic.normalize(gauss_)
        gauss = gauss_.reshape(B, N, Y, X)
    return gauss
    
def xy2heatmaps(xy, Y, X, sigma=30.0):
    # xy is B x N x 2

    B, N, D = list(xy.shape)
    assert(D==2)
    
    grid_y, grid_x = utils.basic.meshgrid2d(B, Y, X)
    # grid_x and grid_y are B x Y x X
    grid_xs = grid_x.unsqueeze(1).repeat(1, N, 1, 1)
    grid_ys = grid_y.unsqueeze(1).repeat(1, N, 1, 1)
    heat = xy2heatmap(xy, sigma, grid_xs, grid_ys, norm=True)
    return heat

def draw_circles_at_xy(xy, Y, X, sigma=12.5):
    B, N, D = list(xy.shape)
    assert(D==2)
    prior = xy2heatmaps(xy, Y, X, sigma=sigma)
    # prior is B x N x Y x X
    prior = (prior > 0.5).float()
    return prior


def draw_rect_on_image(rgb_torch, box, scale,negative= False):
    C, H, W = list(rgb_torch.shape)
    assert(C==3)
    rgb_torch = back2color(rgb_torch)

    box = np.array([int(i) for i in box])

    rgb = rgb_torch.cpu().numpy()

    rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    start_point = box*scale
    end_point = start_point + hyp.max.searchRegion*scale 
    
    if negative:
        # red
        color = (0, 255, 0)     
    else:
        # blue
        color = (255, 0, 0) 

    thickness = 0

    rgb = rgb.astype(np.uint8)
    rgb = cv2.rectangle(rgb, tuple(start_point), tuple(end_point), color, thickness) 


    out = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
    out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
    out = torch.unsqueeze(out, dim=0)
    out = preprocess_color(out)
    out = torch.reshape(out, [1, C, H, W])
    return out

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    '''
    Taken from DINO
    # Copyright (c) Facebook, Inc. and its affiliates.
    # 
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    # 
    #     http://www.apache.org/licenses/LICENSE-2.0
    # 
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    Misc functions.

    Mostly copy-paste from torchvision references or other public repos like DETR:
    https://github.com/facebookresearch/detr/blob/master/util/misc.py
    '''
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        elapsed_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                'elapsed: {elapsed}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                'elapsed: {elapsed}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                elapsed_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        elapsed=elapsed_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        elapsed=elapsed_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class Summ_writer(object):
    def __init__(self, writer, global_step, log_freq=10, fps=8, just_gif=False):
        self.writer = writer
        self.global_step = global_step
        self.log_freq = log_freq
        self.fps = fps
        self.just_gif = just_gif
        self.maxwidth = 1800
        # st()

        self.save_this = (self.global_step % self.log_freq == 0)

    def summ_text(self, name, text):
        if self.save_this:
            self.writer.add_text(name, text, global_step=self.global_step)
    
    def summ_depth(self, name, depth, blacken_zeros=False, cmap='YlGnBu_r'):
        if self.save_this:
            depth[depth==100.] = np.max(depth[depth<100.])
            figure = plt.figure(figsize = (20,14))
            plt.clf()
            plt.imshow(depth, cmap=cmap)
            plt.axis('off')
            # figure.axes.get_xaxis().set_visible(False)
            # figure.axes.get_yaxis().set_visible(False)
            # figure is matplotlib figure
            self.writer.add_figure(name, figure, global_step=self.global_step) 
            plt.close('all')
        

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W
        
        assert tensor.dtype in {torch.uint8,torch.float32}
        shape = list(tensor.shape)
        # assert len(shape) in {4,5}
        # assert shape[4] in {1,3}

        # if len(shape) == 4:
        #     tensor = tensor.unsqueeze(dim=0)

        if tensor.dtype == torch.float32:
            tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        #tensor = tensor.data.numpy()
        #tensor = np.transpose(tensor, axes=[0, 1, 4, 2, 3])

        # tensor = tensor.permute(0, 1, 4, 2, 3) #move the color channel to dim=2

        # tensor = tensor.transpose(2, 4).transpose(3, 4)

        video_to_write = tensor[0:1] #only keep the first if batch > 1 

        self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)
        return video_to_write

    def summ_rotating_pc(self, name, zyx_pc, rgb_pc, blacken_zeros=False):
        my_dpi = 100
        fig = plt.figure(figsize = (480/my_dpi, 480/my_dpi), dpi=my_dpi)
        fig.add_subplot(111, projection='3d')
        fig.tight_layout(pad=0)
        ax = plt.gca()

        pc_x = zyx_pc[:,2].cpu().numpy()
        pc_y = zyx_pc[:,0].cpu().numpy()
        pc_z = -zyx_pc[:,1].cpu().numpy()
        pc_c = rgb_pc.cpu().numpy() + 0.5
        ax.scatter(pc_x, pc_y, pc_z, c = pc_c, s = 0.1)

        # collect views
        views = []
        for angle in range(0,360,20):
            ax.view_init(30, angle)
            buf = io.BytesIO()
            fig.savefig(buf, format='raw')
            buf.seek(0)
            data = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8), newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            data = data[:,:,:3]

            print("angle", angle, data.shape)

            if angle > 0:
                views.append(data)

        views = np.stack(views, axis=0)
        rotate_seq = torch.from_numpy(views).permute(0,3,1,2).unsqueeze(0).cuda()
        return self.summ_gif(name, rotate_seq)

    def summ_rgbs(self, name, ims, blacken_zeros=False, only_return=False):
        if self.save_this:

            ims = gif_and_tile(ims, just_gif=self.just_gif)
            vis = ims

            B, S, C, H, W = list(vis.shape)

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis, blacken_zeros)

    def summ_rgb(self, name, ims, blacken_zeros=False, only_return=False):
        if self.save_this:
            assert ims.dtype in {torch.uint8,torch.float32}

            if ims.dtype == torch.float32:
                ims = back2color(ims, blacken_zeros)

            #ims is B x C x H x W
            vis = ims[0:1] # just the first one
            B, C, H, W = list(vis.shape)

            if int(W) > self.maxwidth:
                vis = vis[:,:,:,:self.maxwidth]

            if only_return:
                return vis
            else:
                return self.summ_gif(name, vis.unsqueeze(1), blacken_zeros)

    def summ_occs(self, name, occs, reduce_axes=[3]):
        if self.save_this:
            B, C, D, H, W = list(occs[0].shape)
            for reduce_axis in reduce_axes:
                heights = [convert_occ_to_height(occ, reduce_axis=reduce_axis) for occ in occs]
                self.summ_oneds(name=('%s_ax%d' % (name, reduce_axis)), ims=heights, norm=False)
            
    def summ_occ(self, name, occ, reduce_axes=[3], only_return=False):
        if self.save_this:
            B, C, D, H, W = list(occ.shape)
            for reduce_axis in reduce_axes:
                height = convert_occ_to_height(occ, reduce_axis=reduce_axis)
                # if only_return:
                #     return height
                if reduce_axis == reduce_axes[-1]:
                    return self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, only_return=only_return)
                else:
                    self.summ_oned(name=('%s_ax%d' % (name, reduce_axis)), im=height, norm=False, only_return=only_return)

    def summ_oneds(self, name, ims, bev=False, logvis=False, max_along_y=False, maxval=50.0, norm=True, only_return=False):
        if self.save_this:

            if bev: 
                B, C, H, _, W = list(ims[0].shape)
                if max_along_y:
                    ims = [torch.max(im, dim=3)[0] for im in ims]
                else:
                    ims = [torch.mean(im, dim=3) for im in ims]

            if len(ims) != 1: # sequence
                im = gif_and_tile(ims, just_gif=self.just_gif)
            else:
                im = torch.stack(ims, dim=1) # single frame

            B, S, C, H, W = list(im.shape)
            
            if logvis and maxval:
                maxval = np.log(maxval)
                im = torch.log(torch.clamp(im, 0)+1.0)
                im = torch.clamp(im, 0, maxval)
                im = im/maxval
            elif maxval:
                im = torch.clamp(im, 0, maxval)
                
            if norm:
                # normalize before oned2inferno,
                # so that the ranges are similar within B across S
                im = utils.basic.normalize(im)
                
            im = im.view(B*S, C, H, W)
            vis = oned2inferno(im, norm=norm)
            vis = vis.view(B, S, 3, H, W)
            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]

            if only_return:
                return vis
            else:
                self.summ_gif(name, vis)

    def rgba2rgb( self, rgba, background=(255,255,255) ):
        row, col, ch = rgba.shape

        if ch == 3:
            return rgba

        assert ch == 4, 'RGBA image has 4 channels.'

        rgb = np.zeros( (row, col, 3), dtype='float32' )
        r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

        a = np.asarray( a, dtype='float32' ) / 255.0

        R, G, B = background

        rgb[:,:,0] = r * a + (1.0 - a) * R
        rgb[:,:,1] = g * a + (1.0 - a) * G
        rgb[:,:,2] = b * a + (1.0 - a) * B

        return np.asarray( rgb, dtype='uint8' )

    def summ_oned(self, name, im, bev=False, logvis=False, maxval=0, max_along_y=False, norm=True, only_return=False, heatmap=False, overlay_image=None):
        if self.save_this:

            if bev: 
                B, C, H, _, W = list(im.shape)
                if max_along_y:
                    im = torch.max(im, dim=3)[0]
                else:
                    im = torch.mean(im, dim=3)
            else:
                B, C, H, W = list(im.shape)
                
            im = im[0:1] # just the first one
            assert(C==1)
            
            if logvis and maxval:
                maxval = np.log(maxval)
                im = torch.log(im)
                im = torch.clamp(im, 0, maxval)
                im = im/maxval
                norm = False
            elif maxval:
                im = torch.clamp(im, 0, maxval)/maxval
                norm = False

            if heatmap:
                im = utils.basic.normalize(im)
                colormap = plt.get_cmap('jet')
                vis = self.rgba2rgb((colormap(im.cpu().squeeze().numpy()) * 255).astype(np.uint8))
                if overlay_image is not None:
                    overlay_image = overlay_image.permute(1,2,0).cpu().numpy()
                    vis = cv2.addWeighted(vis, 0.7, overlay_image, 0.3, 0)
                    # plt.figure()
                    # plt.imshow(vis)
                    # plt.savefig('images/test.png')
                    # st()

                vis = torch.from_numpy(vis).permute(2,0,1).unsqueeze(0)
                # plt.figure()
                # plt.imshow(vis.cpu().numpy())
                # plt.savefig('images/test.png')
            else:
                vis = oned2inferno(im, norm=norm)
            
            # vis = vis.view(B, 3, H, W)
            if W > self.maxwidth:
                vis = vis[...,:self.maxwidth]
            # self.writer.add_images(name, vis, global_step=self.global_step, dataformats='NCHW')
            return self.summ_rgb(name, vis, blacken_zeros=False, only_return=only_return)
            # writer.add_images(name + "_R", vis[:,0:1], global_step=global_step, dataformats='NCHW')
            # writer.add_images(name + "_G", vis[:,1:2], global_step=global_step, dataformats='NCHW')
            # writer.add_images(name + "_B", vis[:,2:3], global_step=global_step, dataformats='NCHW')

    def summ_unps(self, name, unps, occs):
        if self.save_this:
            unps = torch.stack(unps, dim=1)
            occs = torch.stack(occs, dim=1)
            st()
            B, S, C, D, H, W = list(unps.shape)
            occs = occs.repeat(1, 1, C, 1, 1, 1)
            unps = utils.basic.reduce_masked_mean(unps, occs, dim=4)
            unps = torch.unbind(unps, dim=1) #should be S x B x W x D x C
            # unps = [unp.transpose(1, 2) for unp in unps] #rotate 90 degree counter-clockwise
            # return self.summ_rgbs(name=name, ims=unps, blacken_zeros=True) 
            return self.summ_rgbs(name=name, ims=unps, blacken_zeros=False) 


    def summ_unp(self, name, unp, occ):
        if self.save_this:
            B, C, D, H, W = list(unp.shape)
            occ = occ.repeat(1, C, 1, 1, 1)
            unp = utils.basic.reduce_masked_mean(unp, occ, dim=3)
            # unp = [unp.transpose(1, 2) for unp in unp] #rotate 90 degree counter-clockwise
            self.summ_rgb(name=name, ims=unp, blacken_zeros=True) 

    def summ_feats(self, name, feats, valids=None, pca=False, only_return=False):
        if self.save_this:
            if valids is not None:
                valids = torch.stack(valids, dim=1)
            
            feats  = torch.stack(feats, dim=1)
            # feats leads with B x S x C

            if feats.ndim==6:
                # feats is B x S x C x D x H x W
                if valids is None:
                    feats = torch.mean(feats, dim=4)
                else: 
                    valids = valids.repeat(1, 1, feats.size()[2], 1, 1, 1)
                    feats = utils.basic.reduce_masked_mean(feats, valids, dim=4) 

            B, S, C, D, W = list(feats.size())

            if not pca:
                # feats leads with B x S x C
                feats = torch.mean(torch.abs(feats), dim=2, keepdims=True)
                # feats leads with B x S x 1
                
                # feats is B x S x D x W
                feats = torch.unbind(feats, dim=1)
                # feats is a len=S list, each element of shape B x W x D
                # # make "forward" point up, and make "right" point right
                # feats = [feat.transpose(1, 2) for feat in feats]
                return self.summ_oneds(name=name, ims=feats, norm=True, only_return=only_return)

            else:
                __p = lambda x: utils.basic.pack_seqdim(x, B)
                __u = lambda x: utils.basic.unpack_seqdim(x, B)

                feats_  = __p(feats)
                
                if valids is None:
                    feats_pca_ = get_feat_pca(feats_)
                else:
                    valids_ = __p(valids)
                    feats_pca_ = get_feat_pca(feats_, valids)

                feats_pca = __u(feats_pca_)

                return self.summ_rgbs(name=name, ims=torch.unbind(feats_pca, dim=1), only_return=only_return)

    def summ_feat(self, name, feat, valid=None, pca=False, only_return=False):
        if self.save_this:
            if feat.ndim==5: # B x C x D x H x W
                if valid is None:
                    feat = torch.mean(feat, dim=3)
                else:
                    valid = valid.repeat(1, feat.size()[1], 1, 1, 1)
                    feat = utils.basic.reduce_masked_mean(feat, valid, dim=3)
                    
            B, C, D, W = list(feat.shape)

            if not pca:
                feat = torch.mean(torch.abs(feat), dim=1, keepdims=True)
                # feat is B x 1 x D x W
                return self.summ_oned(name=name, im=feat, norm=True, only_return=only_return)
            else:
                feat_pca = get_feat_pca(feat, valid)
                return self.summ_rgb(name, feat_pca, only_return=only_return)

    def summ_scalar(self, name, value):
        if utils.dist.is_main_process():
            if (not (isinstance(value, int) or isinstance(value, float) or isinstance(value, np.float32))) and ('torch' in value.type()):
                value = value.detach().cpu().numpy()
            if not np.isnan(value):
                if (self.log_freq == 1):
                    self.writer.add_scalar(name, value, global_step=self.global_step)
                elif np.mod(self.global_step, 10)==0:
                    self.writer.add_scalar(name, value, global_step=self.global_step)

    def summ_box(self, name, rgbR, boxes_camR, scores, tids, pix_T_cam, only_return=False):
        B, C, H, W = list(rgbR.shape)
        corners_camR = utils.geom.transform_boxes_to_corners(boxes_camR)
        return self.summ_box_by_corners(name, rgbR, corners_camR, scores, tids, pix_T_cam, only_return=only_return)

    def summ_figure(self, name, figure, blacken_zeros=False):
        # figure is matplotlib figure
        # if self.save_this or self.override: 
        if self.save_this or True:
            self.writer.add_figure(name, figure, global_step=self.global_step) 

    def summ_boxlist2d(self, name, rgb, boxlist, scores=None, tids=None, only_return=False):
        B, C, H, W = list(rgb.shape)
        boxlist_vis = self.draw_boxlist2d_on_image(rgb, boxlist, scores=scores, tids=tids)
        if not only_return:
            self.summ_rgb(name, boxlist_vis)
        return boxlist_vis

    def summ_box_by_corners(self, name, rgbR, corners, scores, tids, pix_T_cam, only_return=False, frame_id=None):
        # rgb is B x H x W x C
        # corners is B x N x 8 x 3
        # scores is B x N
        # tids is B x N
        # pix_T_cam is B x 4 x 4

        B, C, H, W = list(rgbR.shape)
        boxes_vis = self.draw_corners_on_image(rgbR,
                                               corners,
                                               torch.mean(corners, dim=2),
                                               scores,
                                               tids,
                                               pix_T_cam,
                                               frame_id=frame_id)
        if not only_return:
            self.summ_rgb(name, boxes_vis)
        return boxes_vis

    def summ_box_by_corners_aithor(self, name, rgbR, corners, scores, tids, pix_T_cam, only_return=False, frame_id=None):
        # rgb is B x H x W x C
        # corners is B x N x 8 x 3
        # scores is B x N
        # tids is B x N
        # pix_T_cam is B x 4 x 4

        B, C, H, W = list(rgbR.shape)

        # first we need to get rid of invalid gt boxes
        # gt_boxes = trim_gt_boxes(gt_boxes)
        assert(C==3)
        B2, N, D, E = list(corners.shape)
        assert(B2==B)
        assert(D==8) # 8 corners
        assert(E==3) # 3D

        rgbR = back2color(rgbR)

        corners_cam_ = torch.reshape(corners, [B, N*8, 3])
        centers_cam_ = torch.reshape(torch.mean(corners, dim=2), [B, N*1, 3])
        corners_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
        centers_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, centers_cam_)
        corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
        centers_pix = torch.reshape(centers_pix_, [B, N, 1, 2])

        # need this in aithor
        corners_pix[:,:,:,1] = H - corners_pix[:,:,:,1]
        centers_pix[:,:,:,1] = H - centers_pix[:,:,:,1]

        boxes_vis = self.draw_boxes_on_image_py(rgbR[0].detach().cpu().numpy(),
                                          corners_pix[0].detach().cpu().numpy(),
                                          centers_pix[0].detach().cpu().numpy(),
                                          scores[0].detach().cpu().numpy(),
                                          tids[0].detach().cpu().numpy(),
                                          frame_id=frame_id)
        boxes_vis = torch.from_numpy(boxes_vis).type(torch.ByteTensor).permute(2, 0, 1)
        boxes_vis = torch.unsqueeze(boxes_vis, dim=0)
        boxes_vis = preprocess_color(boxes_vis)
        boxes_vis = torch.reshape(boxes_vis, [1, C, H, W])
        if not only_return:
            self.summ_rgb(name, boxes_vis)
        return boxes_vis
    
    def summ_lrtlist(self, name, rgbR, lrtlist, scorelist, tidlist, pix_T_cam, only_return=False, frame_id=None):
        # rgb is B x H x W x C
        # lrtlist is B x N x 19
        # scorelist is B x N
        # tidlist is B x N
        # pix_T_cam is B x 4 x 4
        if self.save_this:


            B, C, H, W = list(rgbR.shape)
            B, N, D = list(lrtlist.shape)

            xyzlist_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist)
            # this is B x N x 8 x 3

            clist_cam = utils.geom.get_clist_from_lrtlist(lrtlist)

            boxes_vis = self.draw_corners_on_image(rgbR,
                                                   xyzlist_cam,
                                                   clist_cam, 
                                                   scorelist,
                                                   tidlist,
                                                   pix_T_cam,
                                                   frame_id=frame_id)
            return self.summ_rgb(name, boxes_vis, only_return=only_return)
    
    def summ_lrtlist_bev(self, name, occ_memR, lrtlist, scorelist, tidlist, vox_util, lrt=None, already_mem=False, only_return=False, frame_id=None, boxlist=None):
        if self.save_this:
            # rgb is B x C x Z x Y x X
            # lrtlist is B x N x 19
            # scorelist is B x N
            # tidlist is B x N

            # print('occ_memR', occ_memR.shape)
            # print('lrtlist', lrtlist.shape)
            # print('scorelist', scorelist.shape)
            # print('tidlist', tidlist.shape)
            # if lrt is not None:
            #     print('lrt', lrt.shape)

            B, _, Z, Y, X = list(occ_memR.shape)
            B, N, D = list(lrtlist.shape)

            corners_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist)
            centers_cam = utils.geom.get_clist_from_lrtlist(lrtlist)
            if lrt is None:
                if not already_mem:
                    corners_mem = vox_util.Ref2Mem(corners_cam.reshape(B, N*8, 3), Z, Y, X).reshape(B, N, 8, 3)
                    # this is B x N x 8 x 3
                    centers_mem = vox_util.Ref2Mem(centers_cam, Z, Y, X).reshape(B, N, 1, 3)
                    # this is B x N x 1 x 3
                else:
                    corners_mem = corners_cam.clone().reshape(B, N, 8, 3)
                    centers_mem = centers_cam.clone().reshape(B, N, 1, 3)
            else:
                # use the lrt to know where to voxelize
                corners_mem = vox_util.Ref2Zoom(corners_cam.reshape(B, N*8, 3), lrt, Z, Y, X).reshape(B, N, 8, 3)
                centers_mem = vox_util.Ref2Zoom(centers_cam, lrt, Z, Y, X).reshape(B, N, 1, 3)

            # for i in range(centers_cam.shape[1]):
            #     center = vox_util.voxelize_xyz(centers_cam[:,i:i+1].float(), X,Y,Z)
            #     center_inds = torch.where(center.squeeze())
            
            # centers_mem = centers_mem[:,:,:,[2,1,0]]
            # corners_mem = corners_mem[:,:,:,[2,1,0]]

            # rgb = utils.basic.reduce_masked_mean(unp_memR, occ_memR.repeat(1, C, 1, 1, 1), dim=3)
            rgb_vis = self.summ_occ('', occ_memR, only_return=True)
            # utils.py.print_stats('rgb_vis', rgb_vis.cpu().numpy())
            # print('rgb', rgb.shape)
            # rgb_vis = back2color(rgb)
            # this is in [0, 255]

            # print('rgb_vis', rgb_vis.shape)

            if False:
                # alt method
                box_mem = torch.cat([centers_mem, corners_mem], dim=2).reshape(B, N*9, 3)
                box_vox = vox_util.voxelize_xyz(box_mem, Z, Y, X, already_mem=True)
                box_vis = self.summ_occ('', box_vox, reduce_axes=[3], only_return=True)

                box_vis = convert_occ_to_height(box_vox, reduce_axis=3)
                box_vis = utils.basic.normalize(box_vis)
                box_vis = oned2inferno(box_vis, norm=False)
                # this is in [0, 255]

                # replace black with occ vis
                box_vis[box_vis==0] = (rgb_vis[box_vis==0].float()*0.5).byte() # darken the bkg a bit
                box_vis = preprocess_color(box_vis)
                return self.summ_rgb(('%s' % (name)), box_vis, only_return=only_return)#, only_return=only_return)

            # take the xz part
            centers_mem = torch.stack([centers_mem[:,:,:,0], centers_mem[:,:,:,2]], dim=3)
            corners_mem = torch.stack([corners_mem[:,:,:,0], corners_mem[:,:,:,2]], dim=3)

            out = self.draw_boxes_on_image_py(rgb_vis[0].detach().cpu().numpy(),
                                              corners_mem[0].detach().cpu().numpy(),
                                              centers_mem[0].detach().cpu().numpy(),
                                              scorelist[0].detach().cpu().numpy(),
                                              tidlist[0].detach().cpu().numpy(),
                                              frame_id=frame_id)
            # utils.py.print_stats('py out', out)
            out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
            out = torch.unsqueeze(out, dim=0)
            out = preprocess_color(out)
            return self.summ_rgb(name, out, only_return=only_return)
            # box_vis = torch.reshape(out, [1, 3, Z, X]).byte()
            # out = torch.reshape(out, [1, C, X, Z])
            # out = out.permute(0, 1, 3, 2)

            # box_vis = preprocess_color(out)
            # utils.py.print_stats('box_vis', box_vis.cpu().numpy())

            # if not only_return:
            #     self.summ_rgb(name, box_vis)
            # return box_vis
            #     self.summ_rgb(name, box_vis)

    def draw_corners_on_image(self, rgb, corners_cam, centers_cam, scores, tids, pix_T_cam, frame_id=None):
        # first we need to get rid of invalid gt boxes
        # gt_boxes = trim_gt_boxes(gt_boxes)
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D, E = list(corners_cam.shape)
        assert(B2==B)
        assert(D==8) # 8 corners
        assert(E==3) # 3D

        rgb = back2color(rgb)

        corners_cam_ = torch.reshape(corners_cam, [B, N*8, 3])
        centers_cam_ = torch.reshape(centers_cam, [B, N*1, 3])
        corners_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, corners_cam_)
        centers_pix_ = utils.geom.apply_pix_T_cam(pix_T_cam, centers_cam_)
        corners_pix = torch.reshape(corners_pix_, [B, N, 8, 2])
        centers_pix = torch.reshape(centers_pix_, [B, N, 1, 2])

        out = self.draw_boxes_on_image_py(rgb[0].detach().cpu().numpy(),
                                          corners_pix[0].detach().cpu().numpy(),
                                          centers_pix[0].detach().cpu().numpy(),
                                          scores[0].detach().cpu().numpy(),
                                          tids[0].detach().cpu().numpy(),
                                          frame_id=frame_id)
        out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
        out = torch.unsqueeze(out, dim=0)
        out = preprocess_color(out)
        out = torch.reshape(out, [1, C, H, W])
        return out
    
    def draw_boxes_on_image_py(self, rgb, corners_pix, centers_pix, scores, tids, boxes=None, thickness=1, frame_id=None):
        # all inputs are numpy tensors
        # rgb is H x W x 3
        # corners_pix is N x 8 x 2, in xy order
        # centers_pix is N x 1 x 2, in xy order
        # scores is N
        # tids is N
        # boxes is N x 9 < this is only here to print some rotation info

        # cv2.cvtColor seems to cause an Illegal instruction error on compute-0-38; no idea why
        
        rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 

        H, W, C = rgb.shape
        assert(C==3)
        N, D, E = corners_pix.shape
        assert(D==8)
        assert(E==2)

        if boxes is not None:
            rx = boxes[:,6]
            ry = boxes[:,7]
            rz = boxes[:,8]
        else:
            rx = 0
            ry = 0
            rz = 0

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        if frame_id is not None:
            color = np.array(color_map[0])*255.0
            color[:] = 255
            # color[-1] = 255
            # print(np.array(color_map[0])*255.0)
            # color = np.array([255, 255, 255], np.uint8)
            # print(color)
            # ???
            # print('putting frame id', frame_id)
            cv2.putText(rgb,
                        '%02d' % frame_id,
                        (5, 20), # from left, from top
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, # font scale (float)
                        color, 
                        1) # font thickness (int)
        # else:
        #     print('frame_id is none')
            
        # draw
        for ind, corners in enumerate(corners_pix):
            # corners is 8 x 2
            if not np.isclose(scores[ind], 0.0):

                # print('ind', ind)
                # print('score = %.2f' % scores[ind])
                color_id = int(tids[ind]) % 20
                color = color_map[color_id]
                color = np.array(color)*255.0
                # color = (0,191,255)
                # color = (255,191,0)
                # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])
                
                cv2.circle(rgb,(int(centers_pix[ind,0,0]),int(centers_pix[ind,0,1])),1,color,-1)

                if True:
                    # recently the scores are not informative
                    if scores[ind] < 1.0 and scores[ind] > 0.0:
                        # print('for example, putting this one at', np.min(corners[:,0]), np.min(corners[:,1]))
                        cv2.putText(rgb,
                                    '%.3f' % (scores[ind]), 
                                    # '%.2f match' % (scores[ind]), 
                                    # '%.2f IOU' % (scores[ind]), 
                                    # '%d (%.2f)' % (tids[ind], scores[ind]), 
                                    (np.min(corners[:,0]), np.min(corners[:,1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, # font scale (float)
                                    color,
                                    2) # font thickness (int)

                for c in corners:

                    # rgb[pt1[0], pt1[1], :] = 255
                    # rgb[pt2[0], pt2[1], :] = 255
                    # rgb[np.clip(int(c[0]), 0, W), int(c[1]), :] = 255

                    c0 = np.clip(int(c[0]), 0,  W-1)
                    c1 = np.clip(int(c[1]), 0,  H-1)
                    rgb[c1, c0, :] = 255

                # we want to distinguish between in-plane edges and out-of-plane ones
                # so let's recall how the corners are ordered:

                # (new clockwise ordering)
                xs = np.array([1/2., 1/2., -1/2., -1/2., 1/2., 1/2., -1/2., -1/2.])
                ys = np.array([1/2., 1/2., 1/2., 1/2., -1/2., -1/2., -1/2., -1/2.])
                zs = np.array([1/2., -1/2., -1/2., 1/2., 1/2., -1/2., -1/2., 1/2.])

                for ii in list(range(0,2)):
                    cv2.circle(rgb,(int(corners_pix[ind,ii,0]),int(corners_pix[ind,ii,1])),2,color,-1)
                for ii in list(range(2,4)):
                    cv2.circle(rgb,(int(corners_pix[ind,ii,0]),int(corners_pix[ind,ii,1])),1,color,-1)

                xs = np.reshape(xs, [8, 1])
                ys = np.reshape(ys, [8, 1])
                zs = np.reshape(zs, [8, 1])
                offsets = np.concatenate([xs, ys, zs], axis=1)

                corner_inds = list(range(8))
                combos = list(combinations(corner_inds, 2))

                for combo in combos:
                    pt1 = offsets[combo[0]]
                    pt2 = offsets[combo[1]]
                    # draw this if it is an in-plane edge
                    eqs = pt1==pt2
                    if np.sum(eqs)==2:
                        i, j = combo
                        pt1 = (int(corners[i, 0]), int(corners[i, 1]))
                        pt2 = (int(corners[j, 0]), int(corners[j, 1]))
                        retval, pt1, pt2 = cv2.clipLine((0, 0, W, H), pt1, pt2)
                        if retval:
                            cv2.line(rgb, pt1, pt2, color, thickness, cv2.LINE_AA)

                        # rgb[pt1[0], pt1[1], :] = 255
                        # rgb[pt2[0], pt2[1], :] = 255
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # utils.basic.print_stats_py('rgb_uint8', rgb)
        # imageio.imwrite('boxes_rgb.png', rgb)
        return rgb

    def draw_boxlist2d_on_image(self, rgb, boxlist, scores=None, tids=None):
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D = list(boxlist.shape)
        assert(B2==B)
        assert(D==4) # ymin, xmin, ymax, xmax

        rgb = back2color(rgb)
        if scores is None:
            scores = torch.ones(B2, N).float()
        if tids is None:
            tids = torch.zeros(B2, N).long()
        out = self.draw_boxlist2d_on_image_py(
            rgb[0].cpu().numpy(),
            boxlist[0].cpu().numpy(),
            scores[0].cpu().numpy(),
            tids[0].cpu().numpy())
        out = torch.from_numpy(out).type(torch.ByteTensor).permute(2, 0, 1)
        out = torch.unsqueeze(out, dim=0)
        out = preprocess_color(out)
        out = torch.reshape(out, [1, C, H, W])
        return out
    
    def draw_boxlist2d_on_image_py(self, rgb, boxlist, scores, tids, thickness=1):
        # all inputs are numpy tensors
        # rgb is H x W x 3
        # boxlist is N x 4
        # scores is N
        # tids is N

        rgb = np.transpose(rgb, [1, 2, 0]) # put channels last
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        H, W, C = rgb.shape
        assert(C==3)
        N, D = boxlist.shape
        assert(D==4)

        color_map = matplotlib.cm.get_cmap('tab20')
        color_map = color_map.colors

        # draw
        for ind, box in enumerate(boxlist):
            # box is 4
            if not np.isclose(scores[ind], 0.0):
                # box = utils.geom.scale_box2d(box, H, W)
                ymin, xmin, ymax, xmax = box
                ymin, ymax = ymin*H, ymax*H
                xmin, xmax = xmin*W, xmax*W
                
                # print 'score = %.2f' % scores[ind]
                color_id = tids[ind] % 20
                color = color_map[color_id]
                color = np.array(color)*255.0
                # print 'tid = %d; score = %.3f' % (tids[ind], scores[ind])

                if False:
                    # in this model, the scores are usually 1.0 anyway,
                    # so this is not helpful
                    cv2.putText(rgb,
                                '%d (%.2f)' % (tids[ind], scores[ind]), 
                                (int(xmin), int(ymin)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, # font size
                                color),
                    #1) # font weight

                xmin = np.clip(int(xmin), 0,  W-1)
                xmax = np.clip(int(xmax), 0,  W-1)
                ymin = np.clip(int(ymin), 0,  H-1)
                ymax = np.clip(int(ymax), 0,  H-1)

                cv2.line(rgb, (xmin, ymin), (xmin, ymax), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmin, ymin), (xmax, ymin), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmax, ymin), (xmax, ymax), color, thickness, cv2.LINE_AA)
                cv2.line(rgb, (xmax, ymax), (xmin, ymax), color, thickness, cv2.LINE_AA)
                
        rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return rgb
    
    def summ_histogram(self, name, data):
        if self.save_this:
            data = data.flatten() 
            self.writer.add_histogram(name, data, global_step=self.global_step)

    def flow2color(self, flow, clip=50.0):
        """
        :param flow: Optical flow tensor.
        :return: RGB image normalized between 0 and 1.
        """

        # flow is B x C x H x W

        B, C, H, W = list(flow.size())
        
        abs_image = torch.abs(flow)
        flow_mean = abs_image.mean(dim=[1,2,3])
        flow_std = abs_image.std(dim=[1,2,3])

        if clip:
            flow = torch.clamp(flow, -clip, clip)/clip
        else:
            # Apply some kind of normalization. Divide by the perceived maximum (mean + std)
            flow = flow / (flow_mean + flow_std + 1e-10)[:, None, None, None].repeat(1, C, H, W)

        radius = torch.sqrt(torch.sum(flow**2, dim=1, keepdim=True)) #B x 1 x H x W
        radius_clipped = torch.clamp(radius, 0.0, 1.0)

        angle = torch.atan2(flow[:, 1:], flow[:, 0:1]) / np.pi #B x 1 x H x W

        hue = torch.clamp((angle + 1.0) / 2.0, 0.0, 1.0)
        saturation = torch.ones_like(hue) * 0.75
        value = radius_clipped
        hsv = torch.cat([hue, saturation, value], dim=1) #B x 3 x H x W

        #flow = tf.image.hsv_to_rgb(hsv)
        flow = hsv_to_rgb(hsv)
        flow = (flow*255.0).type(torch.ByteTensor)
        return flow

    def summ_flow(self, name, im, clip=0.0):
        # flow is B x C x D x W
        if self.save_this:
            return self.summ_rgb(name, self.flow2color(im, clip=clip))
        else:
            return None

    def summ_3D_flow(self, name, flow, clip=0.0):
        if self.save_this:
            self.summ_histogram('%s_flow_x' % name, flow[:,0])
            self.summ_histogram('%s_flow_y' % name, flow[:,1])
            self.summ_histogram('%s_flow_z' % name, flow[:,2])

            # flow is B x 3 x D x H x W; inside the 3 it's XYZ
            # D->z, H->y, W->x
            flow_xz = torch.cat([flow[:, 0:1], flow[:, 2:]], dim=1) # grab x, z
            flow_xy = torch.cat([flow[:, 0:1], flow[:, 1:2]], dim=1) # grab x, y
            flow_yz = torch.cat([flow[:, 1:2], flow[:, 2:]], dim=1) # grab y, z
            # these are B x 2 x D x H x W

            flow_xz = torch.mean(flow_xz, dim=3) # reduce over H (y)
            flow_xy = torch.mean(flow_xy, dim=2) # reduce over D (z)
            flow_yz = torch.mean(flow_yz, dim=4) # reduce over W (x)

            return self.summ_flow('%s_flow_xz' % name, flow_xz, clip=clip) # rot90 for interp
            # self.summ_flow('%s_flow_xy' % name, flow_xy, clip=clip)
            # self.summ_flow('%s_flow_yz' % name, flow_yz, clip=clip) # not as interpretable
            
            # flow_mag = torch.mean(torch.sum(torch.sqrt(EPS+flow**2), dim=1, keepdim=True), dim=3)
            # self.summ_oned('%s_flow_mag' % name, flow_mag)
        else:
            return None

    def summ_traj_on_occ(self, name, traj, occ_mem, vox_util, traj_g=None, show_bkg=True, already_mem=False, sigma=2, only_return=False):
        # traj is B x S x 3
        B, C, Z, Y, X = list(occ_mem.shape)
        B2, S, D = list(traj.shape)
        assert(D==3)
        assert(B==B2)
        
        if self.save_this:
            if already_mem:
                traj_mem = traj
                if traj_g is not None:
                    traj_g_mem = traj_g
            else:
                traj_mem = vox_util.Ref2Mem(traj, Z, Y, X, assert_cube=False)
                if traj_g is not None:
                    traj_g_mem = vox_util.Ref2Mem(traj_g, Z, Y, X, assert_cube=False)

            if show_bkg:
                height_mem = convert_occ_to_height(occ_mem, reduce_axis=3)
                # this is B x C x Z x X

                occ_vis = utils.basic.normalize(height_mem)
                occ_vis = oned2inferno(occ_vis, norm=False)
                # print(vis.shape)
            else:
                occ_vis = torch.zeros(B, 3, Z, X).cpu().byte()

            x, y, z = torch.unbind(traj_mem, dim=2)
            xz = torch.stack([x,z], dim=2)
            heats = draw_circles_at_xy(xz, Z, X, sigma=sigma)
            # this is B x S x 1 x Z x X
            heats = torch.squeeze(heats, dim=2)
            heat = seq2color(heats)
            # make black 0
            heat = back2color(heat)

            # print(heat.shape)
            # vis[heat > 0] = heat

            if traj_g is not None:
                x, y, z = torch.unbind(traj_g_mem, dim=2)
                xz = torch.stack([x,z], dim=2)
                heats = draw_circles_at_xy(xz, Z, X, sigma=sigma)
                # this is B x S x 1 x Z x X
                heats = torch.squeeze(heats, dim=2)
                heat_g = seq2color(heats, colormap='winter')
                # make black 0
                heat_g = back2color(heat_g)
                # replace black with g vis
                heat[heat==0] = (heat_g[heat==0].float()).byte()
            
            # replace black with occ vis
            # heat[heat==0] = (occ_vis[heat==0].float()*0.5).byte() # darken the bkg a bit
            # heat[heat==0] = (occ_vis[heat==0].float()*0.5).byte() # darken the bkg a bit
            heat[heat==0] = occ_vis[heat==0]
            heat = preprocess_color(heat)
            
            return self.summ_rgb(('%s' % (name)), heat, only_return=only_return)

    def summ_seg(self, name, seg):
        if not self.save_this:
            return

        B,H,W = seg.shape

        label_colors = [
            (0, 0, 0),         # None
            (70, 70, 70),      # Buildings
            (190, 153, 153),   # Fences
            (72, 0, 90),       # Other
            (220, 20, 60),     # Pedestrians
            (153, 153, 153),   # Poles
            (157, 234, 50),    # RoadLines
            (128, 64, 128),    # Roads
            (244, 35, 232),    # Sidewalks
            (107, 142, 35),    # Vegetation
            (0, 0, 255),      # Vehicles
            (102, 102, 156),  # Walls
            (220, 220, 0)     # TrafficSigns
        ]

        r = torch.zeros_like(seg,dtype=torch.uint8)
        g = torch.zeros_like(seg,dtype=torch.uint8)
        b = torch.zeros_like(seg,dtype=torch.uint8)
        
        for label in range(0,len(label_colors)):
            idx = (seg == label)
            r[idx] = label_colors[label][0]
            g[idx] = label_colors[label][1]
            b[idx] = label_colors[label][2]
            
        rgb = torch.stack([r,g,b],axis=1)
        return self.summ_rgb(name,rgb)
        

    def summ_soft_seg(self, name, seg, bev=False, only_return=False):
        if not self.save_this:
            return

        if bev:
            B,N,D,H,W = seg.shape
            seg = torch.mean(seg, dim=3)
        B,N,H,W = seg.shape
            
        # the values along N should sum to 1

        label_colors = [
            (0, 0, 0),         # None
            (70, 70, 70),      # Buildings
            (190, 153, 153),   # Fences
            (72, 0, 90),       # Other
            (220, 20, 60),     # Pedestrians
            (153, 153, 153),   # Poles
            (157, 234, 50),    # RoadLines
            (128, 64, 128),    # Roads
            (244, 35, 232),    # Sidewalks
            (107, 142, 35),    # Vegetation
            (0, 0, 255),      # Vehicles
            (102, 102, 156),  # Walls
            (220, 220, 0)     # TrafficSigns
        ]

        N2 = len(label_colors)
        if N >= N2:
            print('dropping a few objects, because we do not have enough colors')
            seg = seg[:,:N2]
            N = N2
        # assert(N <= N2) # else we need more colors

        color_map = torch.zeros([B, 3, N, H, W], dtype=torch.float32).cuda()
        seg_ = seg.unsqueeze(1)
        # this is B x 1 x N x H x W

        for label in range(0,N):
            color_map[:,0,label] = label_colors[label][0]
            color_map[:,1,label] = label_colors[label][1]
            color_map[:,2,label] = label_colors[label][2]

        out = torch.sum(color_map * seg_, dim=2)
        out = out.type(torch.ByteTensor)
        return self.summ_rgb(name, out, only_return=only_return)

if __name__ == "__main__":
    logdir = './runs/my_test'
    writer = SummaryWriter(logdir = logdir)

    summ_writer = Summ_writer(writer, 0, 'my_test')

    '''test summ_rgbs'''
    # rand_input = torch.rand(1, 2, 128, 384, 3) - 0.5 #rand from -0.5 to 0.5
    # summ_rgbs(name = 'rgb', ims = torch.unbind(rand_input, dim=1), writer=writer, global_step=0)
    # rand_input = torch.rand(1, 2, 128, 384, 3) - 0.5 #rand from -0.5 to 0.5
    # summ_rgbs(name = 'rgb', ims = torch.unbind(rand_input, dim=1), writer=writer, global_step=1)

    '''test summ_occs'''
    # rand_input = torch.randint(low=0, high = 2, size=(1, 2, 32, 32, 32, 1)).type(torch.FloatTensor) #random 0 or 1
    # summ_occs(name='occs', occs=torch.unbind(rand_input, dim=1), writer=writer, global_step=0)
    # rand_input = torch.randint(low=0, high = 2, size=(1, 2, 32, 32, 32, 1)).type(torch.FloatTensor) #random 0 or 1
    # summ_occs(name='occs', occs=torch.unbind(rand_input, dim=1), writer=writer, global_step=1)

    '''test summ_unps'''
    # for global_step in [0, 1]:
    #     rand_occs = torch.randint(low=0, high = 2, size=(1, 2, 128, 128, 32, 1)).type(torch.FloatTensor) #random 0 or 1
    #     rand_unps = torch.rand(1, 2, 128, 128, 32, 3) - 0.5
    #     summ_unps(name='unps', unps=torch.unbind(rand_unps, dim=1), occs=torch.unbind(rand_occs, dim=1), writer=writer, global_step=global_step)

    '''test summ_feats'''
    # for global_step in [0, 1]:
    #     rand_feats = torch.rand(1, 2, 128, 128, 32, 3) - 0.5
    #     summ_feats(name='feats', feats=torch.unbind(rand_feats, dim=1), writer=writer, global_step=global_step)

    '''test summ_flow'''
    # rand_feats = torch.rand(2, 2, 128, 128) - 0.5
    # summ_writer.summ_flow('flow', rand_feats)

    '''test summ_flow'''
    rand_feats = torch.rand(2, 3, 128, 32, 128)
    summ_writer.summ_3D_flow(rand_feats)


    writer.close()

def convert_boxlist2d_to_masklist(boxlist2d, H, W):
    B, N, C = list(boxlist2d.shape)
    assert(C==4) # 2d boxes
    boxlist2d = utils.geom.unnormalize_boxlist2d(boxlist2d, H, W)
    yminlist, xminlist, ymaxlist, xmaxlist = torch.unbind(boxlist2d, dim=2)
    
    yminlist = yminlist.long()
    xminlist = xminlist.long()
    ymaxlist = ymaxlist.long()
    xmaxlist = xmaxlist.long()

    yminlist = yminlist.clamp(0, H-1)
    ymaxlist = ymaxlist.clamp(0, H-1)
    
    xminlist = xminlist.clamp(0, W-1)
    xmaxlist = xmaxlist.clamp(0, W-1)
    
    masklist = torch.zeros([B, N, H, W]).float().cuda()
    for b in list(range(B)):
        for n in list(range(N)):
            ymin = yminlist[b, n]
            xmin = xminlist[b, n]
            ymax = ymaxlist[b, n]
            xmax = xmaxlist[b, n]
            masklist[b,n,ymin:ymax,xmin:xmax] = 1.0
    return masklist

def get_masks_and_bbox(panoptic_seg):
  num_objs = torch.max(panoptic_seg)
  w, h = panoptic_seg.shape
  pred_masks = []
  pred_boxes = []

  for i in range(num_objs + 1):
    mask = torch.zeros(w, h)
    mask[panoptic_seg==i] = 1

    # getting bbox from mask
    i, j = torch.where(mask)
    box = torch.Tensor([min(j), min(i), max(j), max(i)])

    pred_masks.append(mask)
    pred_boxes.append(box)

  pred_masks = torch.stack(pred_masks).cuda()
  pred_boxes = torch.stack(pred_boxes).cuda()
  return pred_masks, pred_boxes

def get_masks(panoptic_seg):
    num_objs = torch.max(panoptic_seg)
    w, h = panoptic_seg.shape
    pred_masks = torch.stack([(panoptic_seg==i).float() for i in range(num_objs+1)]).cuda()

    return pred_masks

def darken_rgb_outside_mask(rgb, mask, factor=0.5):
    mask = mask*factor + (1.0 - factor) # max_val=1, min_val=factor
    # return (rgb+0.5)*mask - 0.5
    return rgb*mask

def dilate2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = F.conv2d(im, weights2d, padding=1).clamp(0, 1)
    return im

def dilate3d(im, times=1, device='cuda'):
    weights3d = torch.ones(1, 1, 3, 3, 3, device=device)
    for time in range(times):
        im = F.conv3d(im, weights3d, padding=1).clamp(0, 1)
    return im

def erode2d(im, times=1, device='cuda'):
    weights2d = torch.ones(1, 1, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv2d(1.0 - im, weights2d, padding=1).clamp(0, 1)
    return im

def erode3d(im, times=1, device='cuda'):
    weights3d = torch.ones(1, 1, 3, 3, 3, device=device)
    for time in range(times):
        im = 1.0 - F.conv3d(1.0 - im, weights3d, padding=1).clamp(0, 1)
    return im
