# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .hgnet2 import HGNetv2


class HGNetFormula(nn.Module):
    def __init__(self, stem_channels, stage_config, hidden_size, pretrained_backbone, freeze_backbone=True):
        super().__init__()
        backbone_output_dim = stage_config["stage4"][2]
        self.backbone = HGNetv2(stem_channels, stage_config)
        self.projection = nn.Linear(backbone_output_dim, hidden_size)

        self.freeze_backbone = freeze_backbone
        if pretrained_backbone:
            backbone_state_dict = torch.load(pretrained_backbone)
            self.backbone.load_state_dict(backbone_state_dict)

            if freeze_backbone:
                self._freeze_norm(self.backbone)
                self._freeze_parameters(self.backbone)

    def forward(self, pixel_values):
        out = self.backbone(pixel_values)
        out = self.projection(out)
        # (B,C,H,W) -> (B,C,H*W) -> (B,H*W,C)
        out = out.flatten(2).transpose(1, 2)

        return out

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

class HGNetFormulaConfig(PretrainedConfig):
    model_type = "hgnet_formula"

    def __init__(
        self,
        stem_channels,
        stage_config,
        hidden_size,
        pretrained_backbone,
        freeze_backbone,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stem_channels = stem_channels
        self.stage_config = stage_config
        self.hidden_size = hidden_size
        self.pretrained_backbone = pretrained_backbone
        self.freeze_backbone = freeze_backbone

class HGNetFormulaHF(PreTrainedModel):
    config_class = HGNetFormulaConfig
    base_model_prefix = "hgnet_formula"
    main_input_name = "pixel_values"

    def __init__(self, config: HGNetFormulaConfig):
        super().__init__(config)
        backbone_output_dim = config.stage_config["stage4"][2]

        self.backbone = HGNetv2(config.stem_channels, config.stage_config)
        self.projection = nn.Linear(backbone_output_dim, config.hidden_size)

        if config.pretrained_backbone:
            logging.log(logging.INFO, f"load pretrained backbone from {config.pretrained_backbone}")
            backbone_state_dict = torch.load(config.pretrained_backbone)
            self.backbone.load_state_dict(backbone_state_dict)

            if config.freeze_backbone:
                logging.log(logging.INFO, "freeze backbone weight")
                self._freeze_norm(self.backbone)
                self._freeze_parameters(self.backbone)


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

    def forward(self, pixel_values, **kwargs):
        out = self.backbone(pixel_values)
        out = out.flatten(2).transpose(1, 2)
        out = self.projection(out)
        # (B,C,H,W) -> (B,C,H*W) -> (B,H*W,C)

        return BaseModelOutput(last_hidden_state=out)


if __name__ == "__main__":

    import torchinfo

    PATH = Path(__file__).resolve().parent
    pretrained_backbone = PATH / "./formulanet_encoder_hgnetv2.pt"
    ENCODER_CONFIG = {
        "stem_channels": [3, 32, 48],
        "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        "hidden_dim": 384,
        "pretrained_backbone": pretrained_backbone,
        "freeze_backbone": True,
    }

    encoder = HGNetFormula(**ENCODER_CONFIG)
    torchinfo.summary(encoder)

    encoder_HF = HGNetFormulaHF(HGNetFormulaConfig(**ENCODER_CONFIG))
    torchinfo.summary(encoder_HF)
