import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResAttBlock, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        # 3 Convolution Layer, 1x1, 3x3, 1x1
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Residual Convolution kernel_size=1
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        # Output Convolution of F_out
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.adaptive_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax2d = nn.Softmax2d()
        self.act = nn.GELU()

    def forward(self, x):
        out_1 = self.conv_block(x)

        if x.shape != out_1.shape:
            out_res = self.residual_conv(x)
            out_c = self.sigmoid(self.adaptive_pool(out_1)) * out_res
            out_s = self.softmax2d(out_c) * out_res
            output = out_1 + out_c + out_s
        else:
            out_c = self.sigmoid(self.adaptive_pool(out_1)) * x
            out_s = self.softmax2d(out_c) * x
            output = out_1 + out_c + out_s
        return self.conv_out(output)