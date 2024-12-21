import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Flatten, Sequential, Module

from training.encoder.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=2, use_pixelshuffle=True):
        super().__init__()

        if use_pixelshuffle:
            self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        else:
            self.up = nn.Upsample(scale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TriPlaneEncoder(Module):

    def __init__(self, opts):

        super(TriPlaneEncoder, self).__init__()
        self.opts = opts
        blocks = get_blocks(num_layers=50)
        unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(9, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = Sequential(*modules)

        if self.opts.use_pixelshuffle:
            self.up1 = Up(1024, 512, upscale_factor=1, use_pixelshuffle=True)
            self.up2 = Up(384, 384, use_pixelshuffle=True)
            self.up3 = Up(224, 256, use_pixelshuffle=True)
            self.up4 = Up(128, 96, use_pixelshuffle=True)
            self.up5 = nn.PixelShuffle(upscale_factor=2)
            self.final_head_two = nn.Sequential(
                nn.Conv2d(120, 96, kernel_size=3, padding=1),
                nn.PReLU(96),
                nn.Conv2d(96, 96, kernel_size=3, padding=1),
                nn.PReLU(96),
                nn.Conv2d(96, 96, kernel_size=1),
            )
        else:
            self.up1 = Up(1024, 512, upscale_factor=1, use_pixelshuffle=False)
            self.up2 = Up(768, 256, use_pixelshuffle=False)
            self.up3 = Up(384, 128, use_pixelshuffle=False)
            self.up4 = Up(192, 64, use_pixelshuffle=False)
            self.up5 = nn.Upsample(scale_factor=2)
            self.final_head_two = nn.Sequential(
                nn.Conv2d(352, 96 * 3, kernel_size=3, padding=1),
                nn.PReLU(96 * 3),
                # nn.Conv2d(96, 96, kernel_size=3, padding=1),
                # nn.PReLU(96),
                nn.Conv2d(96 * 3, 96 * 3, kernel_size=1),
            )

    def forward(self, x, t_planes):
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x

        tri_plane = self.up1(x, c3)
        tri_plane = self.up2(tri_plane, c2)
        tri_plane = self.up3(tri_plane, c1)
        tri_plane = self.up4(tri_plane, c0)
        tri_plane = self.up5(tri_plane)
        tri_plane = torch.cat([t_planes.reshape(-1, 96 * 3, 256, 256), tri_plane], dim=1)
        tri_plane2 = self.final_head_two(tri_plane).reshape(-1, 3, 96, 256, 256)
        t_planes = t_planes + tri_plane2
        return t_planes


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Test TriPlaneEncoder")
#     parser.add_argument("--use_pixelshuffle", action="store_true", help="Use PixelShuffle instead of Upsample")
#     opts = parser.parse_args()

#     # def count_parameters(model):
#     #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

#     model = TriPlaneEncoder(opts)
#     # print(count_parameters(model))

#     # from training.encoder import encoders as psp_encoders
#     # model = psp_encoders.GradualStyleEncoder(50, "ir_se", n_styles=14, )
#     # print(count_parameters(model))

#     x_input = torch.cat([torch.randn(3, 256, 256), torch.randn(3, 256, 256)], dim=0).unsqueeze(0)
#     print(x_input.shape)
#     t_planes = torch.randn(1, 3, 96, 256, 256)
#     y = model(x_input, t_planes)
#     print(y.shape)
