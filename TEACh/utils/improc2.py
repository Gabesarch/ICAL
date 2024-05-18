import tensorflow as tf
import torch
import torchvision.transforms
import cv2
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import numpy as np
from matplotlib import cm
# import hyperparams as hyp
import matplotlib
import imageio
from itertools import combinations
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
# import utils.geom
# import utils.py
# import utils.basic

import io
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

EPS = 1e-6

# color conversion libs, for flow vis

def draw_line_single(img, whwh, color, thickness=1):
    rgb = np.copy(img)
    cv2.line(rgb, (whwh[1], whwh[0]), (whwh[3], whwh[2]), color, thickness, cv2.LINE_AA)
    return rgb

def draw_box_single(img, box, color, thickness=1):
    hmin, wmin, hmax, wmax = box
    rgb = np.copy(img)
    cv2.line(rgb, (wmin, hmin), (wmin, hmax), color, thickness, cv2.LINE_AA)
    cv2.line(rgb, (wmin, hmin), (wmax, hmin), color, thickness, cv2.LINE_AA)
    cv2.line(rgb, (wmax, hmin), (wmax, hmax), color, thickness, cv2.LINE_AA)
    cv2.line(rgb, (wmax, hmax), (wmin, hmax), color, thickness, cv2.LINE_AA)
    return rgb

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
        self.override = False

    def summ_figure(self, name, figure, blacken_zeros=False):
        # figure is matplotlib figure
        if self.save_this or self.override: 
            self.writer.add_figure(name, figure, global_step=self.global_step) 
    
    def summ_text(self, name, text):
        if self.save_this or self.override: 
            # text_np = np.array(text)
            # text_tensor = tf.constant(text_np)
            # summary_op = tf.summary.text(name, text_tensor)
            # summary = tf.Session.run(summary_op)
            self.writer.add_text(name, text, global_step=self.global_step)

    def summ_rgb(self, name, image):
        if self.save_this or self.override:
            self.writer.add_image(name, image, global_step=self.global_step) 

    def summ_gif(self, name, tensor, blacken_zeros=False):
        # tensor should be in B x S x C x H x W
        
        assert tensor.dtype in {torch.uint8,torch.float32}
        shape = list(tensor.shape)
        # assert len(shape) in {4,5}
        # assert shape[4] in {1,3}

        # if len(shape) == 4:
        #     tensor = tensor.unsqueeze(dim=0)

        # if tensor.dtype == torch.float32:
        #     tensor = back2color(tensor, blacken_zeros=blacken_zeros)

        #tensor = tensor.data.numpy()
        #tensor = np.transpose(tensor, axes=[0, 1, 4, 2, 3])

        # tensor = tensor.permute(0, 1, 4, 2, 3) #move the color channel to dim=2

        # tensor = tensor.transpose(2, 4).transpose(3, 4)

        video_to_write = tensor[0:1] #only keep the first if batch > 1 

        self.writer.add_video(name, video_to_write, fps=self.fps, global_step=self.global_step)
        return video_to_write 
        
    def summ_boxlist2d(self, name, rgb, boxlist, scores=None, tids=None, only_return=False):
        B, C, H, W = list(rgb.shape)
        if self.save_this or self.override:
            boxlist_vis = self.draw_boxlist2d_on_image(rgb, boxlist, scores=scores, tids=tids)
            boxlist_vis = boxlist_vis[0] # first image
            if only_return:
                return boxlist_vis
            else:
                self.writer.add_image(name, boxlist_vis, global_step=self.global_step) 
                return boxlist_vis
        else:
            return None
        # return boxlist_vis

    def summ_rgbs(self, name, images):
        if self.save_this or self.override:
            self.writer.add_images(name, images, global_step=self.global_step)     

    def summ_scalar(self, name, value):
        if (not (isinstance(value, int) or isinstance(value, float) or isinstance(value, np.float32))) and ('torch' in value.type()):
            value = value.detach().cpu().numpy()
        if not np.isnan(value):
            self.writer.add_scalar(name, value, global_step=self.global_step)
            # if (self.log_freq == 1):
            #     self.writer.add_scalar(name, value, global_step=self.global_step)
            # elif np.mod(self.global_step, 10)==0:
            #     self.writer.add_scalar(name, value, global_step=self.global_step)
    
    def draw_boxlist2d_on_image(self, rgb, boxlist, scores=None, tids=None):
        B, C, H, W = list(rgb.shape)
        assert(C==3)
        B2, N, D = list(boxlist.shape)
        assert(B2==B)
        assert(D==4) # ymin, xmin, ymax, xmax

        rgb = self.back2color(rgb)
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
        out = self.preprocess_color(out)
        out = torch.reshape(out, [1, C, H, W])
        return out

    def back2color(self, i, blacken_zeros=False):
        if blacken_zeros:
            const = torch.tensor([-0.5])
            i = torch.where(i==0.0, const.cuda() if i.is_cuda else const, i)
            return back2color(i)
        else:
            # return ((i+0.5)*255).type(torch.ByteTensor)
            return (i*255).type(torch.ByteTensor)

    def preprocess_color(self, x):
        return x.float() * 1./255

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
                # ymin, ymax = ymin*H, ymax*H
                # xmin, xmax = xmin*W, xmax*W
                
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

    
