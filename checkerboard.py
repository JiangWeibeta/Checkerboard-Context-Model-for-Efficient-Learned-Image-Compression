import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
import compressai.models as models
from compressai.models.utils import conv, deconv, update_registered_buffers
import time

from Gcm import Gcm

class Checkerboard(models.JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, M=192, height=256, width=256, batch_size=8, **kwargs):
        super(Checkerboard, self).__init__(N=N, M=M)
        self.gcm = Gcm(N, M)
        self.height = int(height)
        self.width = int(width)
        self.batch_size = int(batch_size)
        self.pseudo_y_half = torch.zeros([int(batch_size), M * 2, int(height) // 16, int(width) // 16], requires_grad=False).cuda()
        self.mask = torch.zeros([batch_size, M, height // 16, width // 16], requires_grad=False).cuda()
        self.remask = torch.zeros([batch_size, M, height // 16, width // 16], requires_grad=False).cuda()
        self.context_mask = torch.zeros([batch_size, M * 2, height // 16, width // 16], requires_grad=False).cuda()
        self.init_mask(height, width)

    def init_mask(self, height, width):
        # self.register_buffer('mask', torch.zeros([self.batch_size, self.M, height // 16, width // 16]).clone())
        # self.register_buffer('remask', torch.zeros([self.batch_size, self.M, height // 16, width // 16]).clone())
        for i in range(height // 16):
            for j in range(width // 16):
                if (i + j) % 2:
                    self.mask[:, :, i, j] = 1
                else:
                    self.remask[:, :, i, j] = 1
                    self.context_mask[:, :, i, j] = 1

    def quantize(self, y):
        pass


    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        hyper_info = self.h_s(z_hat)

        y_anchor = y_hat * self.mask
        masked_context = self.gcm(y_anchor)
        masked_context = self.context_mask * masked_context

        gaussian_params_anchor = self.entropy_parameters(
            torch.cat((hyper_info, self.pseudo_y_half), dim=1)
        )
        gaussian_params_nonanchor = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )

        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        scales_nonanchor, means_nonanchor = gaussian_params_nonanchor.chunk(2, 1)

        scales_anchor = scales_anchor * self.mask
        means_anchor = means_anchor * self.mask
        scales_nonanchor = scales_nonanchor * self.remask
        means_nonanchor = means_nonanchor * self.remask

        scales = scales_anchor + scales_nonanchor
        means = means_anchor + means_nonanchor

        _, y_likelihoods = self.gaussian_conditional(y, scales, means=means)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_info = self.h_s(z_hat)

        y_anchor = y * self.mask

        gaussian_params_anchor = self.entropy_parameters(
            torch.cat((hyper_info, self.pseudo_y_half), dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        y_strings_anchor = self.gaussian_conditional.compress(y_anchor, indexes_anchor, means_anchor)
        y_anchor_quantized = self.gaussian_conditional.decompress(y_strings_anchor, indexes_anchor, means=means_anchor)

        masked_context = self.gcm(y_anchor_quantized)
        masked_context = self.context_mask * masked_context

        gaussian_params_nonanchor = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )

        scales_nonanchor, means_nonanchor = gaussian_params_nonanchor.chunk(2, 1)

        y_nonanchor = y * self.remask

        indexes_nonanchor = self.gaussian_conditional.build_indexes(scales_nonanchor)
        y_strings_nonanchor = self.gaussian_conditional.compress(y_nonanchor, indexes_nonanchor, means_nonanchor)
        y_nonanchor_quantized = self.gaussian_conditional.decompress(y_strings_nonanchor, indexes_nonanchor, means=means_nonanchor)
        x_hat = self.g_s(y_anchor_quantized + y_nonanchor_quantized)

        return {"strings": [y_strings_anchor, y_strings_nonanchor, z_strings], "shape": z.size()[-2:]}, hyper_info

    def decompress(self, strings, shape, hyper_info):

        start_time = time.clock()

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)

        # hyper_info = self.h_s(z_hat)

        gaussian_params_anchor = self.entropy_parameters(
            torch.cat((hyper_info, self.pseudo_y_half), dim=1)
        )

        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        y_anchor_quantized = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_anchor)

        masked_context = self.gcm(y_anchor_quantized)

        masked_context = self.context_mask * masked_context

        gaussian_params_nonanchor = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )

        scales_nonanchor, means_nonanchor = gaussian_params_nonanchor.chunk(2, 1)

        indexes_nonanchor = self.gaussian_conditional.build_indexes(scales_nonanchor)
        y_nonanchor_quantized = self.gaussian_conditional.decompress(strings[1], indexes_nonanchor, means=means_nonanchor)

        x_hat = self.g_s(y_anchor_quantized + y_nonanchor_quantized)

        end_time = time.clock()

        cost_time = end_time - start_time

        return x_hat, cost_time

    def update_entropy_bottleneck(self):
        self.entropy_bottleneck.update()

    def compress_slice_concatenate(self):
        pass

    def decompress_slice_concatenate(self):
        pass


if __name__ == "__main__":
    # y_hat = torch.ones([1, 192, 16, 16])
    # batch_size, channel, height, width = y_hat.shape
    # mask = torch.zeros([batch_size, channel, height, width])
    # mask
    # for i in range(height):
    #     for j in range(width):
    #         if (i + j) % 2:
    #             mask[:, :, i, j] = 1
    # y_half = y_hat * mask
    # print(y_half)
    test = Checkerboard()
    test.cuda()
    x = torch.zeros([16, 3, 256, 256]).cuda()
    print(test(x))
