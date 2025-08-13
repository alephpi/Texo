from pathlib import Path

import torch
import torch.nn as nn

from .hgnet2 import HGNetv2

# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula

ENCODER_CONFIG = {
    "stem_channels": [3, 32, 48],
    "stage_config": {
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [48, 48, 128, 1, False, False, 3, 6],
            "stage2": [128, 96, 512, 1, True, False, 3, 6],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        },
    "output_hidden_dim": 384,
}

PATH = Path(__file__).resolve().parent
BACKBONE_PRETRAINED_PATH = PATH / "./formulanet_encoder_hgnetv2.pt"

class FormulaNetEncoder(nn.Module):
    def __init__(self, config, pretrained_backbone=True, freeze_backbone=True):
        super().__init__()
        stem_channels = config["stem_channels"]
        stage_config = config["stage_config"]
        output_hidden_dim = config["output_hidden_dim"]
        backbone_output_dim = config["stage_config"]["stage4"][2]
        self.backbone = HGNetv2(stem_channels, stage_config)
        self.projection = nn.Linear(backbone_output_dim, output_hidden_dim)

        self.freeze_backbone = freeze_backbone
        if pretrained_backbone:
            backbone_state_dict = torch.load(BACKBONE_PRETRAINED_PATH)
            self.backbone.load_state_dict(backbone_state_dict)

        if freeze_backbone:
            self._freeze_norm(self.backbone)
            self._freeze_parameters(self.backbone)

    def forward(self, pixel_values):
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values)
        else:
            out = self.backbone(pixel_values)
        out = self.projection(out)
        # (B,C,H,W) -> (B,C,H*W) -> (B,H*W,C)
        out = out.flatten(2).transpose(1, 2)

        return {"last_hidden_state": out}

    def _freeze_norm(self, m: nn.Module):
        from .layers import FrozenBatchNorm2d
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