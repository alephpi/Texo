import torch
import torch.nn as nn

from .hgnet2 import HGNetv2


# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula
class FormulaNet(nn.Module):
    """FormulaNet is a combination of HGNetv2 backbone and a feature extractor head.
    """
    def __init__(
            self, 
            in_channels=3,
            class_num=1024,
            class_expand=2048,
            dropout=0.0
        ):

        super().__init__()
        self.in_channels = in_channels
        stem_channels = [3, 32, 48]
        stage_config = {
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [48, 48, 128, 1, False, False, 3, 6],
            "stage2": [128, 96, 512, 1, True, False, 3, 6],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        }
        out_channels = stage_config["stage4"][2]  # the out_channels of stage4
        self.backbone = HGNetv2(
            stem_channels, 
            stage_config, 
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=class_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(class_expand, class_num)

    def forward(self, x):
        out = self.backbone(x)
        ...