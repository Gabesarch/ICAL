#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp

import attr
# import numba
import numpy as np

# from habitat_sim.bindings import cuda_enabled
# from habitat_sim.registry import registry
# from habitat_sim.sensor import SensorType
from utils.noise_models.sensor_noise_model import SensorNoiseModel

# if cuda_enabled:
#     from habitat_sim._ext.habitat_sim_bindings import RedwoodNoiseModelGPUImpl

# torch = None
import torch
import ipdb
st = ipdb.set_trace
import time


# Read about the noise model here: http://www.alexteichman.com/octo/clams/
# Original source code: http://redwood-data.org/indoor/data/simdepth.py
# @numba.jit(nopython=True)s
def _undistort(x, y, z, model):
    i2 = int((z + 1) / 2)
    i1 = int(i2 - 1)
    a = (z - (i1 * 2.0 + 1.0)) / 2.0
    x = x // 8
    y = y // 6
    f = (1 - a) * model[y, x, min(max(i1, 0), 4)] + a * model[y, x, min(i2, 4)]

    if f < 1e-5:
        return 0
    else:
        return z / f


# @numba.jit(nopython=True, parallel=True)
def _simulate(gt_depth, model, noise_multiplier):
    noisy_depth = np.empty_like(gt_depth)

    H, W = gt_depth.shape
    ymax, xmax = H - 1, W - 1

    rand_nums = np.random.randn(H, W, 3).astype(np.float32)
    for j in range(H):
        for i in range(W):
            y = int(
                min(max(j + rand_nums[j, i, 0] * 0.25 * noise_multiplier, 0.0), ymax)
                + 0.5
            )
            x = int(
                min(max(i + rand_nums[j, i, 1] * 0.25 * noise_multiplier, 0.0), xmax)
                + 0.5
            )

            # Downsample
            d = gt_depth[y - y % 2, x - x % 2]
            # If the depth is greater than 10, the sensor will just return 0
            if d >= 10.0:
                noisy_depth[j, i] = 0.0
            else:
                # Distort
                # The noise model was originally made for a 640x480 sensor,
                # so re-map our arbitrarily sized sensor to that size!
                undistorted_d = _undistort(
                    int(x / xmax * 639.0 + 0.5), int(y / ymax * 479.0 + 0.5), d, model
                )

                if undistorted_d == 0.0:
                    noisy_depth[j, i] = 0.0
                else:
                    denom = round(
                        (
                            35.130 / undistorted_d
                            + rand_nums[j, i, 2] * 0.027778 * noise_multiplier
                        )
                        * 8.0
                    )
                    if denom <= 1e-5:
                        noisy_depth[j, i] = 0.0
                    else:
                        noisy_depth[j, i] = 35.130 * 8.0 / denom

    return noisy_depth


def _simulate_torch(gt_depth, model, noise_multiplier):
    gt_depth = torch.from_numpy(gt_depth).cuda()
    noisy_depth = torch.empty_like(gt_depth).cuda()

    H, W = gt_depth.shape
    ymax, xmax = H - 1, W - 1

    rand_nums = torch.randn(H, W, 3).cuda()
    for j in range(H):
        for i in range(W):
            y = int(
                min(max(j + rand_nums[j, i, 0] * 0.25 * noise_multiplier, 0.0), ymax)
                + 0.5
            )
            x = int(
                min(max(i + rand_nums[j, i, 1] * 0.25 * noise_multiplier, 0.0), xmax)
                + 0.5
            )

            # Downsample
            d = gt_depth[y - y % 2, x - x % 2]
            # If the depth is greater than 10, the sensor will just return 0
            if d >= 10.0:
                noisy_depth[j, i] = 0.0
            else:
                # Distort
                # The noise model was originally made for a 640x480 sensor,
                # so re-map our arbitrarily sized sensor to that size!
                undistorted_d = _undistort(
                    int(x / xmax * 639.0 + 0.5), int(y / ymax * 479.0 + 0.5), d, model
                )

                if undistorted_d == 0.0:
                    noisy_depth[j, i] = 0.0
                else:
                    denom = torch.round(
                        (
                            35.130 / undistorted_d
                            + rand_nums[j, i, 2] * 0.027778 * noise_multiplier
                        )
                        * 8.0
                    )
                    if denom <= 1e-5:
                        noisy_depth[j, i] = 0.0
                    else:
                        noisy_depth[j, i] = 35.130 * 8.0 / denom

    return noisy_depth


# def _simulate_torch(gt_depth, model, noise_multiplier):
#     H, W = gt_depth.shape
#     ymax, xmax = H - 1, W - 1
    
#     gt_depth = torch.from_numpy(gt_depth).cuda()
#     noisy_depth = torch.empty_like(gt_depth).cuda()
    
#     rand_nums = torch.randn(H, W, 3).cuda() #.astype(np.float32)
#     H_t = torch.arange(H).unsqueeze(1).repeat(1, W).cuda()
#     W_t = torch.arange(W).unsqueeze(0).repeat(H, 1).cuda()
#     ys = H_t + rand_nums[:, :, 0] * 0.25 * noise_multiplier
#     ys = torch.clip(ys, min=0.0, max=ymax) + 0.5
#     ys = ys.long() #torch.round(ys)
#     xs = W_t + rand_nums[:, :, 1] * 0.25 * noise_multiplier
#     xs = torch.clip(xs, min=0.0, max=ymax) + 0.5
#     xs = xs.long() #torch.round(xs)
#     rand_num_mult = rand_nums[:, :, 2] * 0.027778 * noise_multiplier
#     # ys_mod = ys % 2
#     # d = gt_depth[y - y % 2, x - x % 2]

#     ys_2 = ys % 2
#     xs_2 = xs % 2
#     ys_2 = ys_2.flatten()
#     xs



#     # st()
#     # for j in range(H):
#     #     for i in range(W):
#     #         y = int(
#     #             min(max(j + rand_nums[j, i, 0] * 0.25 * noise_multiplier, 0.0), ymax)
#     #             + 0.5
#     #         )
#     #         x = int(
#     #             min(max(i + rand_nums[j, i, 1] * 0.25 * noise_multiplier, 0.0), xmax)
#     #             + 0.5
#     #         )
#     for j in range(H):
#         for i in range(W):

#             y = ys[j, i]
#             x = xs[j, i]

#             # Downsample
#             noisy_depth[j,i] = gt_depth[y - y % 2, x - x % 2]
#     st()
#     xs = xs / xmax * 639.0 + 0.5
#     ys = ys / ymax * 479.0 + 0.5
#     xs = xs.long()
#     ys = ys.long()

#     z = noisy_depth.flatten()
#     x = xs.flatten()
#     y = ys.flatten()

#     i2 = (z + 1) / 2
#     i2 = i2.long()
#     i1 = i2 - 1
#     i1 = i1.long()
#     a = (z - (i1 * 2.0 + 1.0)) / 2.0
#     x = x // 8
#     y = y // 6
#     i1 = torch.clip(i1, min=0, max=4)
#     i2 = torch.clip(i2, max=4)
#     f = (1 - a) * model[y, x, i1] + a * model[y, x, i2]
#     undistorted_d = z/f
#     undistorted_d[f<1e-5] = 0.0
#     st()

#             # # If the depth is greater than 10, the sensor will just return 0
#             # if d >= 10.0:
#             #     noisy_depth[j, i] = 0.0
#             # else:
#             #     # Distort
#             #     # The noise model was originally made for a 640x480 sensor,
#             #     # so re-map our arbitrarily sized sensor to that size!
#             #     undistorted_d = _undistort(
#             #         int(x / xmax * 639.0 + 0.5), int(y / ymax * 479.0 + 0.5), d, model
#             #     )

#             #     if undistorted_d == 0.0:
#             #         noisy_depth[j, i] = 0.0
#             #     else:
#             #         denom = (35.130 / undistorted_d + rand_num_mult[j, i]) * 8.0
#             #         denom = torch.round(denom).long()
#             #         if denom <= 1e-5:
#             #             noisy_depth[j, i] = 0.0
#             #         else:
#             #             noisy_depth[j, i] = 35.130 * 8.0 / denom

#     return noisy_depth


# @attr.s(auto_attribs=True)
class RedwoodNoiseModelCPUImpl():

    def __init__(self, model, noise_multiplier=1.0):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.model = self.model.reshape(self.model.shape[0], -1, 4)

    def simulate(self, gt_depth):
        return _simulate(gt_depth, self.model, self.noise_multiplier)


class RedwoodNoiseModelGPUImpl():

    def __init__(self, model, noise_multiplier=1.0):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.model = self.model.reshape(self.model.shape[0], -1, 4)
        self.model = torch.from_numpy(self.model).cuda()

    def simulate(self, gt_depth):
        return _simulate_torch(gt_depth, self.model, self.noise_multiplier)


# @registry.register_noise_model
# @attr.s(auto_attribs=True, kw_only=True)
class RedwoodDepthNoiseModel(SensorNoiseModel):

    def __init__(self, noise_multiplier=1.0):
        self.noise_multiplier = noise_multiplier
        dist = np.load(
            osp.join("utils/noise_models/data", "redwood-depth-dist-model.npy")
        )

        # if cuda_enabled:
        #     self._impl = RedwoodNoiseModelGPUImpl(
        #         dist, self.gpu_device_id, self.noise_multiplier
        #     )
        # else:
        self._impl_cpu = RedwoodNoiseModelCPUImpl(dist, self.noise_multiplier)
        self._impl_gpu = RedwoodNoiseModelGPUImpl(dist, self.noise_multiplier)

    # @staticmethod
    # def is_valid_sensor_type(sensor_type: SensorType) -> bool:
    #     return sensor_type == SensorType.DEPTH

    def simulate(self, gt_depth):
        # global torch
        # if cuda_enabled:
        #     if isinstance(gt_depth, np.ndarray):
        #         return self._impl.simulate_from_cpu(gt_depth)
        #     else:
        #         if torch is None:
        #             import torch
        #         noisy_depth = torch.empty_like(gt_depth)
        #         rows, cols = gt_depth.size()
        #         self._impl.simulate_from_gpu(
        #             gt_depth.data_ptr(), rows, cols, noisy_depth.data_ptr()
        #         )
        #         return noisy_depth
        # else:
        start = time.time()
        noisy_depth = self._impl_gpu.simulate(gt_depth)
        end = time.time()
        print(end-start)
        start = time.time()
        noisy_depth = self._impl_cpu.simulate(gt_depth)
        end = time.time()
        print(end-start)
        
        st()
        return noisy_depth

    def apply(self, gt_depth):
        r"""Alias of `simulate()` to conform to base-class and expected API
        """
        return self.simulate(gt_depth)
