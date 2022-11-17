import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ops import ste_round
from compressai.models import JointAutoregressiveHierarchicalPriors, Cheng2020Anchor
from compressai.ans import BufferedRansEncoder, RansDecoder
from layers import CheckerboardContext


class Cheng2020AnchorwithCheckerboard(Cheng2020Anchor):
    """
    share entropy_parameters model for anchor and non-anchor
    Note the receptive field of entropy parameters module is 1x1
    """
    def __init__(self, N=192, **kwargs):
        super().__init__(N, **kwargs)
        self.context_prediction = CheckerboardContext(
            in_channels = N, out_channels = N * 2, kernel_size=5, stride=1, padding=2
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        hyper_params = self.h_s(z_hat)
        ctx_params = self.context_prediction(y_hat)
        # mask anchor
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def validate(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        # mask non-anchor
        gaussian_params_anchor[:, :, 0::2, 0::2] = 0
        gaussian_params_anchor[:, :, 1::2, 1::2] = 0
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        ctx_params = self.context_prediction(ste_round(y - means_anchor) + means_anchor)
        # mask anchor
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat = ste_round(y - means_hat) + means_hat
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        torch.backends.cudnn.deterministic = True
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        y = self.g_a(x)
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([y.size(0), y.size(1) * 2, y.size(2), y.size(3)], device=y.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.compress_anchor(y, scales_anchor, means_anchor, symbols_list, indexes_list)

        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.compress_nonanchor(y, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True

        torch.cuda.synchronize()
        start_time = time.process_time()
        
        y_strings = strings[0][0]
        z_strings = strings[1]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros([z_hat.size(0), self.M * 2, z_hat.size(2) * 4, z_hat.size(3) * 4], device=z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.decompress_anchor(scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)

        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.decompress_nonanchor(scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)

        y_hat = anchor_hat + nonanchor_hat
        x_hat = self.g_s(y_hat)

        torch.cuda.synchronize()
        end_time = time.process_time()
        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

    def ckbd_anchor_sequeeze(self, y):
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y):
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor):
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor):
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor

    def compress_anchor(self, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
        # squeeze anchor to avoid non-anchor symbols
        anchor_squeeze = self.ckbd_anchor_sequeeze(anchor)
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
        return anchor_hat

    def compress_nonanchor(self, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
        nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(nonanchor)
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
        return nonanchor_hat

    def decompress_anchor(self, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) + means_anchor_squeeze
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat)
        return anchor_hat

    def decompress_nonanchor(self, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) + means_nonanchor_squeeze
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
