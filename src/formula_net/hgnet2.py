"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    ConvBNAct,
    FrozenBatchNorm2d,
    LightConvBNAct,
    PaddingSameAsPaddleMaxPool2d,
)
from .utils import register

__all__ = ["HGNetv2"]

def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

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
            padding="same",
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2,
            mid_channels,
            kernel_size=2,
            stride=1,
            padding="same",
        )
        self.stem3 = ConvBNAct(
            mid_channels * 2,
            mid_channels,
            kernel_size=3,
            stride=2,  # NOTE in paddleOCR, this stride can be 1 if it is text recognition task
        )
        self.stem4 = ConvBNAct(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )
        # TODO check whether this implementation is equivalent to paddleOCR
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        # Paddle Detection
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True, padding="SAME")
        # PaddleOCR2Pytorch
        # self.pool = PaddingSameAsPaddleMaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.stem1(x)
        # x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        # x2 = F.pad(x2, (0, 1, 0, 1))
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
        aggregation_squeeze_conv = ConvBNAct(
            total_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
        )
        aggregation_excitation_conv = ConvBNAct(
            out_channels // 2,
            out_channels,
            kernel_size=1,
            stride=1,
        )
        self.aggregation = nn.Sequential(
            aggregation_squeeze_conv,
            aggregation_excitation_conv,
        )

        # self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            # x = self.drop_path(x) + identity
            x += identity
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
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
            )
        else:
            self.downsample = nn.Identity()

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
        x = self.downsample(x)
        x = self.blocks(x)
        return x


# NOTE we follow the D-FINE structure which only model the backbone of PPHGNetV2 in HGNetV2
# we seperate the task specific head into the task specific model
# i.e. last_conv layers etc. in FormulaNet
@register()
class HGNetv2(nn.Module):
    """
    HGNetV2 backbone
    """



    def __init__(
        self,
        stem_channels,
        stage_config,
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
    ):
        super().__init__()

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
        )

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
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

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x