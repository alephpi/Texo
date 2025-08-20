"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBNAct, LightConvBNAct

__all__ = ["HGNetv2"]
class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
        )
        self.stem2a = ConvBNAct(
            mid_channels,
            mid_channels // 2,
            kernel_size=2,
            stride=1,
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2,
            mid_channels,
            kernel_size=2,
            stride=1,
        )
        self.stem3 = ConvBNAct(
            mid_channels * 2,
            mid_channels,
            kernel_size=3,
            stride=2,
        )
        self.stem4 = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        # we don't use PaddleOCR2Pytorch implementation is since it will create a wrapper for pooling layer
        # which we don't want as we want to have a network structure as closer as possible to PPFormulaNet
        # PaddleOCR2Pytorch
        # self.pool = PaddingSameAsPaddleMaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class HG_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        layer_num, # NOTE in paddleOCR, this is 6 by default
        kernel_size=3,
        residual=False, # NOTE same as the argument "identity" in paddleOCR but we keep the name here since it is indeed a residual connectioin
        light_block=True,
        # drop_path=0.0,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_channels if i == 0 else mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_channels if i == 0 else mid_channels,
                        mid_channels,
                        kernel_size=kernel_size,
                        stride=1,
                    )
                )

        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            total_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
        )
        self.aggregation_excitation_conv = ConvBNAct(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            stride=1,
        )
        # self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.residual:
            # x = self.drop_path(x) + identity
            x = x + identity
        return x


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        block_num,
        layer_num,
        downsample=True,
        light_block=True,
        kernel_size=3,
        # drop_path=0.0,
    ):
        super().__init__()
        self.use_downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
            )

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    # drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.use_downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


# NOTE we follow the D-FINE structure which only model the backbone of PPHGNetV2 in HGNetV2
# we seperate the task specific head into the task specific model
# i.e. last_conv layers etc. in FormulaNet
class HGNetv2(nn.Module):
    """
    HGNetV2 backbone
    """

    def __init__(self, stem_channels, stage_config):
        super().__init__()

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
        )

        # stages
        self.stages = nn.ModuleList()
        for k in stage_config:
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                )
            )

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x