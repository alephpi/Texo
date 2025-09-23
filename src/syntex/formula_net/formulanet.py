# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .hgnet2 import HG_Stage, StemBlock


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
        pretrained_backbone:str|Path="",
        freeze_backbone:bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stem_channels = stem_channels
        self.stage_config = stage_config
        self.hidden_size = hidden_size
        self.pretrained_backbone = pretrained_backbone
        self.freeze_backbone = freeze_backbone

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

        if config.pretrained_backbone:
            logging.log(logging.INFO, f"load pretrained backbone from {config.pretrained_backbone}")
            state_dict = torch.load(config.pretrained_backbone)
            self.load_state_dict(state_dict)

            if config.freeze_backbone:
                logging.log(logging.INFO, "freeze backbone weight")
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

AutoConfig.register("my_hgnetv2", HGNetv2Config)
AutoModel.register(HGNetv2Config, HGNetv2)


if __name__ == "__main__":
    import torch
    import torchinfo
    from transformers import VisionEncoderDecoderModel
    from transformers.models.mbart.modeling_mbart import MBartConfig, MBartForCausalLM

    from .formulanet import HGNetv2, HGNetv2Config

    # parallel to PPFormulaNet config
    ENCODER_CONFIG = HGNetv2Config(
        stem_channels=[3, 32, 48],
        stage_config= {
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
            "stage1": (48, 48, 128, 1, 6, 3, False, False),
            "stage2": (128, 96, 512, 1, 6, 3, True, False),
            "stage3": (512, 192, 1024, 3, 6, 5, True, True),
            "stage4": (1024, 384, 2048, 1, 6, 5, True, True),
        },
        hidden_size= 2048,
        pretrained_backbone="",
        freeze_backbone= False,
    )


    # see paddleocr/ppocr/modeling/heads/rec_ppformulanet_head.py and paddleocr/configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml
    DECODER_CONFIG = MBartConfig(
        vocab_size=50000,
        max_position_embeddings=1024+3,
        d_model=384,
        decoder_layers=2,
        decoder_attention_heads=16,
        decoder_ffn_dim=1536,
        decoder_start_token_id=0,
        layer_norm_eps=1e-05,
        is_decoder=True,
        scale_embedding=True,
        tie_word_embeddings=False,
    )

    encoder = HGNetv2(ENCODER_CONFIG)
    decoder = MBartForCausalLM(DECODER_CONFIG)
    input_data = {
        "pixel_values": torch.randn(16, 3, 384, 384, dtype=torch.float32),
        "decoder_input_ids":torch.randint(0, 50000, (16, 256), dtype=torch.int32), 
        "decoder_attention_mask": torch.ones(16, 256, dtype=torch.int32),
    }
    model = VisionEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder,
    )
    
    torchinfo.summary(model, input_data=input_data, depth=4)