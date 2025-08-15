import torch
import torch.nn as nn
from lightning import LightningModule
from transformers import MBartConfig, MBartForCausalLM, VisionEncoderDecoderModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .encoder import HGNetFormulaConfig, HGNetFormulaHF

tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained("tokenizer_config.json")

MODEL_CONFIG = {
    "encoder_config" : {
        "stem_channels": [3, 32, 48],
        "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        "output_hidden_dim": 384,
        "pretrained_backbone": "formulanet_encoder_hgnetv2.pt",
        "freeeze_backbone": True,
    },
    "decoder_config": {
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": 1024,
        "d_model": 384,
        "decoder_layers": 2,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 1536,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "bos_token_id": tokenizer.bos_token.id,
        "eos_token_id": tokenizer.eos_token.id,
        "pad_token_id": tokenizer.pad_token.id,
        "forced_eos_token_id": tokenizer.eos_token.id,
        "layer_norm_eps": 1e-05,
        "is_decoder": True,
    }
}


class FormulaNetLit(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.save_hyperparameters()

        self.model = VisionEncoderDecoderModel(
            encoder=HGNetFormulaHF(HGNetFormulaConfig(**MODEL_CONFIG["encoder_config"])),
            decoder=MBartForCausalLM(MBartConfig(**MODEL_CONFIG["decoder_config"])),
        )

    def forward(self, pixel_values, labels, attention_mask):
        outputs =  self.model.forward(
            pixel_values=pixel_values, 
            labels=labels,
            decoder_attention_mask=attention_mask,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False
            )
        return outputs
    
    def model_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = outputs.loss.mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return
