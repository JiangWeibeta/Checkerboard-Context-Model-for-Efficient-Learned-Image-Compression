import torch
import torch.nn as nn

import MaskedConv

torch.set_printoptions(profile="full")

class Gcm(nn.Module):
    def __init__(self, N=128, M=192):
        super(Gcm, self).__init__()
        self.mask_conv = MaskedConv.MaskedConv2d(M, M * 2, 5, stride=1, padding=2, bias=False)

    def forward(self, y_half):
        y_half = self.mask_conv(y_half)
        return y_half


if __name__ == "__main__":
    y_half = torch.ones([1, 128, 16, 16])
    mask = torch.zeros([1, 128, 16, 16])
    for i in range(16):
        for j in range(16):
            if (i + j) % 2:
                mask[:, :, i, j] = 1
    # print(mask)
    y_half *= mask
    # print(y_half)
    gcm = Gcm()
    y_half = gcm(y_half)
    print(y_half.shape)
    # print(y_half)
