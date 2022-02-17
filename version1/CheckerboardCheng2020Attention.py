import numpy as np
import os
import torch
import time
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math
import torch.nn.init as init
import logging
import compressai.models as models

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from compressai.models.utils import conv, deconv, update_registered_buffers


from Gcm import Gcm

class CheckerboardCheng2020Attention(models.Cheng2020Attention):
    def __init__(self, N=192, height=256, width=256, batch_size=8, **kwargs):
        super(CheckerboardCheng2020Attention, self).__init__(N=192)
        self.N = N
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.zeros = torch.zeros([int(batch_size), N * 2, int(height) // 16, int(width) // 16], requires_grad=False).cuda()
        self.mask = torch.zeros([batch_size, N, height // 16, width // 16], requires_grad=False).cuda()
        self.remask = torch.zeros([batch_size, N, height // 16, width // 16], requires_grad=False).cuda()
        self.context_mask = torch.zeros([batch_size, N * 2, height // 16, width // 16], requires_grad=False).cuda()
        for i in range(height // 16):
            for j in range(width // 16):
                if (i + j) % 2:
                    self.mask[:, :, i, j] = 1
                else:
                    self.remask[:, :, i, j] = 1
                    self.context_mask[:, :, i, j] = 1

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
            torch.cat((hyper_info, self.zeros), dim=1)
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
            torch.cat((hyper_info, self.zeros), dim=1)
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
        y_nonanchor_quantized = self.gaussian_conditional.decompress(y_strings_nonanchor, indexes_nonanchor,
                                                                     means=means_nonanchor)
        x_hat = self.g_s(y_anchor_quantized + y_nonanchor_quantized)

        return {"strings": [y_strings_anchor, y_strings_nonanchor, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):

        start_time = time.clock()

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)

        hyper_info = self.h_s(z_hat)

        gaussian_params_anchor = self.entropy_parameters(
            torch.cat((hyper_info, self.zeros), dim=1)
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
        y_nonanchor_quantized = self.gaussian_conditional.decompress(strings[1], indexes_nonanchor,
                                                                     means=means_nonanchor)

        x_hat = self.g_s(y_anchor_quantized + y_nonanchor_quantized)

        end_time = time.clock()

        cost_time = end_time - start_time

        return x_hat, cost_time

    # Slice y into four parts A, B, C and D. Encode and Decode them separately.
    # Workflow is described in Figure 1 and Figure 2 in Supplementary Material.

    def compress_slice_concatenate(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_info = self.h_s(z_hat)

        gaussian_params_anchor = self.entropy_parameters(
            torch.cat((hyper_info, self.zeros), dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        y_b = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        y_c = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_b = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_c = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_b = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_c = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        y_b[:, :, :, :] = y[:, :, 0::2, 1::2]
        y_c[:, :, :, :] = y[:, :, 1::2, 0::2]
        scales_b[:, :, :, :] = scales_anchor[:, :, 0::2, 1::2]
        scales_c[:, :, :, :] = scales_anchor[:, :, 1::2, 0::2]
        means_b[:, :, :, :] = means_anchor[:, :, 0::2, 1::2]
        means_c[:, :, :, :] = means_anchor[:, :, 1::2, 0::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)

        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)
        y_anchor_quantized = torch.zeros([self.batch_size, self.M, self.height // 16, self.width // 16]).cuda()
        y_anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        y_anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        masked_context = self.gcm(y_anchor_quantized)
        masked_context = self.context_mask * masked_context

        gaussian_params_nonanchor = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )
        scales_nonanchor, means_nonanchor = gaussian_params_nonanchor.chunk(2, 1)

        y_a = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        y_d = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_a = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_d = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_a = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_d = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        y_a[:, :, :, :] = y[:, :, 0::2, 0::2]
        y_d[:, :, :, :] = y[:, :, 1::2, 1::2]
        scales_a[:, :, :, :] = scales_nonanchor[:, :, 0::2, 0::2]
        scales_d[:, :, :, :] = scales_nonanchor[:, :, 1::2, 1::2]
        means_a[:, :, :, :] = means_nonanchor[:, :, 0::2, 0::2]
        means_d[:, :, :, :] = means_nonanchor[:, :, 1::2, 1::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means_a)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means_d)

        return {"strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
                "shape": z.size()[-2:]}

    def decompress_slice_concatenate(self, strings, shape):
        start_time = time.clock()

        y_quantized = torch.zeros([self.batch_size, self.M, self.height // 16, self.width // 16]).cuda()

        z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        hyper_info = self.h_s(z_hat)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat((hyper_info, self.zeros), dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_c = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_b = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_c = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_b[:, :, :, :] = scales_anchor[:, :, 0::2, 1::2]
        scales_c[:, :, :, :] = scales_anchor[:, :, 1::2, 0::2]
        means_b[:, :, :, :] = means_anchor[:, :, 0::2, 1::2]
        means_c[:, :, :, :] = means_anchor[:, :, 1::2, 0::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        y_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        y_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        masked_context = self.gcm(y_quantized)
        masked_context = self.context_mask * masked_context
        gaussian_params_nonanchor = self.entropy_parameters(
            torch.cat((hyper_info, masked_context), dim=1)
        )
        scales_nonanchor, means_nonanchor = gaussian_params_nonanchor.chunk(2, 1)

        scales_a = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_d = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_a = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        means_d = torch.zeros([self.batch_size, self.M, self.height // 32, self.width // 32]).cuda()
        scales_a[:, :, :, :] = scales_nonanchor[:, :, 0::2, 0::2]
        scales_d[:, :, :, :] = scales_nonanchor[:, :, 1::2, 1::2]
        means_a[:, :, :, :] = means_nonanchor[:, :, 0::2, 0::2]
        means_d[:, :, :, :] = means_nonanchor[:, :, 1::2, 1::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        y_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        y_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        x_hat = self.g_s(y_quantized)

        end_time = time.clock()

        cost_time = end_time - start_time

        return x_hat, cost_time


