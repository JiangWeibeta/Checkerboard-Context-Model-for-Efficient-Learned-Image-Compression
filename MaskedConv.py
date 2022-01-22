import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kernel_height, kernel_width = self.mask.size()
        self.mask.fill_(0)
        for i in range(kernel_height):
            for j in range(kernel_width):
                if (i + j) % 2:
                    self.mask[:, :, i, j] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        # print(self.weight.data)
        return super(MaskedConv2d, self).forward(x)


   
if __name__ == "__main__":
    M = 128
    maskedconv = MaskedConv2d(M, M * 2, 5, stride=1, padding=2)
    print(maskedconv.mask)
    # print(maskedconv.weight.data * maskedconv.mask)
