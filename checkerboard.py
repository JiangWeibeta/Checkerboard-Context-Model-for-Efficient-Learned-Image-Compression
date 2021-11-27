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

from Gcm import Gcm


class Checkerboard(models.JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=128, M=192, height=256, width=256, batch_size=8, **kwargs):
        super(Checkerboard, self).__init__(N=N, M=M)
        self.gcm = Gcm(N, M)
        self.N = N
        self.M = M
        self.height = int(height)
        self.width = int(width)
        self.batch_size = int(batch_size)
        self.mask = None
        self.init_mask(height, width)
        self.pseudo_y_half = torch.zeros([int(batch_size), M * 2, int(height) // 16, int(width) // 16])

    def init_mask(self, height, width):
        self.mask = torch.zeros([self.batch_size, self.M, height // 16, width // 16])
        for i in range(height // 16):
            for j in range(width // 16):
                if (i + j) % 2:
                    self.mask[:, :, i, j] = 1

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        hyper_info = self.h_s(z_hat)
        if not self.training:
            _, _, image_height, image_width = x.shape
            self.init_mask(image_height, image_width)
        y_half = y_hat * self.mask.cuda()
        masked_context = self.gcm(y_half)
        gaussian_params = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        y_hat = torch.round(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_info = self.h_s(z_hat)
        y_half = y_hat * self.mask
        masked_context = self.gcm(y_half)
        gaussian_params = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_info = self.hs(z_hat)
        pseudo_means, pseudo_variances = self.gep(self.pseudo_y_half, hyper_info)
        pseudo_indexes = self.gaussian_conditional.build_indexes(pseudo_variances)
        pseudo_y_hat = self.gaussian_conditional.decompress(strings[0], pseudo_indexes, means=pseudo_means)
        y_half = pseudo_y_hat * self.mask
        means, variances = self.gep(y_half, hyper_info)
        indexes = self.gaussian_conditional.build_indexes(variances)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means)
        x_hat = self.gs(y_hat).clamp_(0, 1)
        return x_hat


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
