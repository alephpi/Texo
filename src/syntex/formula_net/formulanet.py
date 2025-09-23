import torch
import torchinfo

# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula
from transformers import (
    MBartConfig,
    MBartForCausalLM,
    PreTrainedTokenizerFast,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from .hgnet2 import HGNetv2, HGNetv2Config


class FormulaNet(VisionEncoderDecoderModel):
    def __init__(self, config):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.pop("tokenizer"))
        self.config = VisionEncoderDecoderConfig(**config)
        super().__init__(self.config)

if __name__ == "__main__":
    config = {
        "model": {
            "tokenizer": "./data/tokenizer",
            "encoder": {
                "model_type": "my_hgnetv2",
                "stem_channels": [3, 32, 48],
                "stage_config": {
                    "stage1": [48, 48, 128, 1, 6, 3, False, False],
                    "stage2": [128, 96, 512, 1, 6, 3, True, False],
                    "stage3": [512, 192, 1024, 3, 6, 5, True, True],
                    "stage4": [1024, 384, 2048, 1, 6, 5, True, True],
                },
                "hidden_size": 2048,
                "pretrained_backbone": False,
                "freeze_backbone": True,
            },
            "decoder": {
                "model_type": "mbart",
                "vocab_size": 687,
                "max_position_embeddings": 1024,
                "d_model": 384,
                "decoder_layers": 2,
                "decoder_attention_heads": 16,
                "decoder_ffn_dim": 1536,
                "bos_token_id": 2,
                "eos_token_id": 3,
                "pad_token_id": 0,
                "forced_eos_token_id": 3,
                "layer_norm_eps": 1e-5,
                "is_decoder": True,
                "scale_embedding": True,
                "tie_word_embeddings": False,
            },
    }
}

    model = FormulaNet(config["model"])

    input_data = {
        "pixel_values": torch.randn(16, 3, 384, 384, dtype=torch.float32),
        "decoder_input_ids":torch.randint(0, config["model"]["decoder"]["vocab_size"], (16, 256), dtype=torch.int32), 
        "decoder_attention_mask": torch.ones(16, 256, dtype=torch.int32),
    }
    torchinfo.summary(model, input_data=input_data, depth=4)
