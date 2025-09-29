"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .layer import ConvBNAct, LightConvBNAct

__all__ = ["HGNetv2", "HGNetv2Config"]
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
        in_channels:int,
        mid_channels:int,
        out_channels:int,
        num_blocks:int,
        num_layers:int,
        kernel_size:int=3,
        downsample:bool=True,
        light_block:bool=True,
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
        for i in range(num_blocks):
            blocks_list.append(
                HG_Block(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    num_layers,
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

# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula
class HGNetv2Config(PretrainedConfig):
    model_type = "my_hgnetv2" # do not confuse with the transformers hgnetv2

    def __init__(
        self,
        stem_channels: List[int]=[3, 32, 48],
        stage_config: Dict[str, Tuple[int,int,int,int,int,int,bool,bool]]={
            # in_channels, mid_channels, out_channels, num_blocks, num_layers, kernel_size, downsample, light_block
            "stage1": (48, 48, 128, 1, 6, 3, False, False),
            "stage2": (128, 96, 512, 1, 6, 3, True, False),
            "stage3": (512, 192, 1024, 3, 6, 5, True, True),
            "stage4": (1024, 384, 2048, 1, 6, 5, True, True),
        },
        hidden_size:int=384,
        pretrained:str|Path="",
        freeze:bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stem_channels = stem_channels
        self.stage_config = stage_config
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self.freeze = freeze

# NOTE we follow the D-FINE structure which only model the backbone of PPHGNetV2 in HGNetV2
# we seperate the task specific head into the task specific model
# i.e. last_conv layers etc. in FormulaNet
class HGNetv2(PreTrainedModel):
    """
    HGNetV2 backbone
    """
    config_class = HGNetv2Config
    base_model_prefix = "my_hgnetv2"
    main_input_name = "pixel_values"

    def __init__(self, config:HGNetv2Config):
        super().__init__(config)
        self.stem = StemBlock(*config.stem_channels)
        self.stages = nn.ModuleList(HG_Stage(*config.stage_config[k]) for k in config.stage_config)

        if config.pretrained:
            logging.log(logging.INFO, f"load pretrained model from {config.pretrained}")
            state_dict = torch.load(config.pretrained)
            self.load_state_dict(state_dict)

        if config.freeze:
            logging.log(logging.INFO, "freeze model weight")
            self._freeze_norm(self)
            self._freeze_parameters(self)


    def forward(self, pixel_values, **kwargs):
        x = self.stem(pixel_values)
        for stage in self.stages:
            x = stage(x)
        # (B,C,H,W) -> (B,C,H*W) -> (B,H*W,C)
        out = x.flatten(2).transpose(1,2)
        return BaseModelOutput(last_hidden_state=out)

    def _freeze_norm(self, m: nn.Module):
        from .layer import FrozenBatchNorm2d
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
