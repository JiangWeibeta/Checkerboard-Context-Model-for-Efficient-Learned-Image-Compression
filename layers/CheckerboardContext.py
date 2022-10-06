#
# Created by Wei Jiang on 2022/02/15.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class CheckerboardContext(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out


if __name__ == '__main__':
    ckbd = CheckerboardContext(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, bias=False)
    # print(ckbd.mask)
    anchor = torch.zeros([1, 3, 8, 8])
    anchor[:, :, 0::2, 1::2] = 1
    anchor[:, :, 1::2, 0::2] = 1
    # print(anchor)
    print(ckbd(anchor))

"""
ckbd(anchor):

          [-3.9174e-01,  0.0000e+00, -5.6143e-01,  0.0000e+00, -5.6143e-01,
            0.0000e+00, -4.6364e-01,  0.0000e+00],
          [ 0.0000e+00, -2.6317e-01,  0.0000e+00, -3.3227e-01,  0.0000e+00,
           -3.3227e-01,  0.0000e+00, -1.2223e-03],
          [-3.3980e-01,  0.0000e+00, -3.0401e-01,  0.0000e+00, -3.0401e-01,
            0.0000e+00, -1.9141e-01,  0.0000e+00],
          [ 0.0000e+00, -2.3491e-01,  0.0000e+00, -3.0401e-01,  0.0000e+00,
           -3.0401e-01,  0.0000e+00,  1.3273e-01],
          [-3.3980e-01,  0.0000e+00, -3.0401e-01,  0.0000e+00, -3.0401e-01,
            0.0000e+00, -1.9141e-01,  0.0000e+00],
          [ 0.0000e+00, -2.3491e-01,  0.0000e+00, -3.0401e-01,  0.0000e+00,
           -3.0401e-01,  0.0000e+00,  1.3273e-01],
          [-2.6121e-01,  0.0000e+00, -1.8591e-01,  0.0000e+00, -1.8591e-01,
            0.0000e+00, -7.3309e-02,  0.0000e+00],
          [ 0.0000e+00,  5.6478e-02,  0.0000e+00,  1.2801e-01,  0.0000e+00,
            1.2801e-01,  0.0000e+00,  3.8838e-01]],

when training and testing, bias = True
"""
