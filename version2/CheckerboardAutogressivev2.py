#
# Created by Wei Jiang on 2022/02/15.
#

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models import JointAutoregressiveHierarchicalPriors
from layers import CheckerboardContext


class CheckerboardAutogressivev2(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N, M, **kwargs)

        self.context_prediction = CheckerboardContext(M, M * 2, 5, 1, 2)

    def forward(self, x):
        """
        anchor :
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
        non-anchor (use anchor as context):
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
        """
        batch_size, channel, x_height, x_width = x.shape
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        params = self.h_s(z_hat)

        anchor = torch.zeros_like(y_hat).to(x.device)
        non_anchor = torch.zeros_like(y_hat).to(x.device)

        anchor[:, :, 0::2, 1::2] = y_hat[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y_hat[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y_hat[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y_hat[:, :, 1::2, 1::2]

        # print(anchor)
        # print(non_anchor)

        # compress anchor
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        # compress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_hat = torch.zeros([batch_size, self.M, x_height // 16, x_width // 16]).to(x.device)
        means_hat = torch.zeros([batch_size, self.M, x_height // 16, x_width // 16]).to(x.device)

        scales_hat[:, :, 0::2, 1::2] = scales_anchor[:, :, 0::2, 1::2]
        scales_hat[:, :, 1::2, 0::2] = scales_anchor[:, :, 1::2, 0::2]
        scales_hat[:, :, 0::2, 0::2] = scales_non_anchor[:, :, 0::2, 0::2]
        scales_hat[:, :, 1::2, 1::2] = scales_non_anchor[:, :, 1::2, 1::2]
        means_hat[:, :, 0::2, 1::2] = means_anchor[:, :, 0::2, 1::2]
        means_hat[:, :, 1::2, 0::2] = means_anchor[:, :, 1::2, 0::2]
        means_hat[:, :, 0::2, 0::2] = means_non_anchor[:, :, 0::2, 0::2]
        means_hat[:, :, 1::2, 1::2] = means_non_anchor[:, :, 1::2, 1::2]

        # print(scales_hat - scales_anchor)
        # print(scales_hat - scales_non_anchor)

        _, y_likelihoods = self.gaussian_conditional(y_hat, scales=scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    def compress(self, x):
        """
        if y[i, :, j, k] == 0 
        then bpp = 0
        """
        batch_size, channel, x_height, x_width = x.shape
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]

        # compress anchor
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_strings = self.gaussian_conditional.compress(anchor, indexes_anchor, means_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor)

        # compress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(non_anchor, indexes_non_anchor, means_non_anchor)

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        """
        if y[i, :, j, k] == 0 
        then bpp = 0
        """
        start_time = time.process_time()

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        params = self.h_s(z_hat)

        batch_size, channel, z_height, z_width = z_hat.shape

        # decompress anchor
        ctx_params_anchor = torch.zeros([batch_size, 2 * self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_anchor)

        # decompress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor)
        non_anchor_quantized = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor, means=means_non_anchor)

        y_hat = anchor_quantized + non_anchor_quantized
        x_hat = self.g_s(y_hat)

        end_time = time.process_time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def compress_slice_concatenate(self, x):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        batch_size, channel, x_height, x_width = x.shape

        y = self.g_a(x)

        y_a = y[:, :, 0::2, 0::2]
        y_d = y[:, :, 1::2, 1::2]
        y_b = y[:, :, 0::2, 1::2]
        y_c = y[:, :, 1::2, 0::2]

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        anchor = torch.zeros_like(y).to(x.device)
        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, x_height // 16, x_width // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)

        anchor_quantized = torch.zeros_like(y).to(x.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means_d)

        return {
            "strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress_slice_concatenate(self, strings, shape):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        start_time = time.process_time()

        z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        params = self.h_s(z_hat)

        batch_size, channel, z_height, z_width = z_hat.shape
        ctx_params_anchor = torch.zeros([batch_size, self.M * 2, z_height * 4, z_width * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        anchor_quantized = torch.zeros([batch_size, self.M, z_height * 4, z_width * 4]).to(z_hat.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        # Add non_anchor_quantized
        anchor_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        x_hat = self.g_s(anchor_quantized)

        end_time = time.process_time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }


if __name__ == "__main__":
    x = torch.randn([1, 3, 64, 64])
    model = CheckerboardJointAutogressivev2()
    model.update(force=True)
    out_c = model.compress_slice_concatenate(x)
    out_d = model.decompress_slice_concatenate(out_c["strings"], out_c["shape"])
