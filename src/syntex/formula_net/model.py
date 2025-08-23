import torch
from lightning import LightningModule
from transformers import (
    AutoConfig,
    AutoModel,
    MBartConfig,
    MBartForCausalLM,
    VisionEncoderDecoderModel,
)
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .encoder import HGNetFormulaConfig, HGNetFormulaHF
from .scores import compute_bleu, compute_edit_distance

AutoConfig.register("hgnet_formula", HGNetFormulaConfig)
AutoModel.register(HGNetFormulaConfig, HGNetFormulaHF)

class FormulaNetLit(LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config

        self.save_hyperparameters()

        self.model = VisionEncoderDecoderModel(
            encoder=HGNetFormulaHF(HGNetFormulaConfig(**self.model_config["encoder"])),
            decoder=MBartForCausalLM(MBartConfig(**self.model_config["decoder"])),
        )
        # transformers.VisionEncoderDecoderModel is not smart enough to initialize from the encoder and decoder
        # where we have to initialize the model.config manually as the following.
        self.model.config.decoder_start_token_id = self.model.decoder.config.bos_token_id
        self.model.config.pad_token_id = self.model.decoder.config.pad_token_id
        self.model.config.eos_token_id = self.model.decoder.config.eos_token_id

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.training_config["optimizer"])
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(self.optimizer, **self.training_config["lr_scheduler"])
        self.tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(self.model_config["tokenizer_path"])
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values, decoder_input_ids, decoder_attention_mask, labels, **kwargs):
        # here we don't do token shift for labels, since the MBartForCausalLM behavior is different from RobertaForCausalLM or GPT2LMHeadModel
        # that's why we don't simply input labels to the forward function as many of the HF tutorials do, 
        # since the VisionEncoderDecoderModel assumes the decoder's behavior is RobertaForCausalLM-like,
        # and cannot handle the sophistication here.
        # Man what can I say, if UniMERNet/ppFormulaNet didn't use it, I wouldn't touch it at all.
        # BTW we have another token-shift bug for VisionEncoderDecoderModel in >4.49,
        # both of them cost me one day...
        outputs =  self.model(
            pixel_values=pixel_values, 
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False
            )
        return outputs.loss

    def generate(self, pixel_values, **kwargs):
        outputs = self.model.generate(pixel_values, **kwargs)
        return outputs

    def model_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        return outputs
    
    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.model_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)

        # labels = batch["labels"]
        # labels[labels == -100] = self.model.config.pad_token_id
        # ref_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # max_length = labels.shape[-1] # in validation, since we know how long the ground truth is, we truncate to it to save computation.

        # outputs = self.generate(batch["pixel_values"], num_beams=1, do_sample=False, max_length=max_length)
        # pred_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # bleu = compute_bleu(pred_str, ref_str)
        # edit_distance = compute_edit_distance(pred_str, ref_str)

        # outputs_beam_search = self.generate(batch["pixel_values"], num_beams=4, do_sample=False, max_length=max_length)
        # pred_str_beam_search = self.tokenizer.batch_decode(outputs_beam_search, skip_special_tokens=True)
        # bleu_beam_search = compute_bleu(pred_str_beam_search, ref_str)
        # edit_distance_beam_search = compute_edit_distance(pred_str_beam_search, ref_str)

        # # log metrics
        # self.log("BLEU", bleu, on_step=False, on_epoch=True, logger=True)
        # self.log("edit_distance", edit_distance, on_step=False, on_epoch=True, logger=True)
        # self.log("BLEU_beam_search", bleu_beam_search, on_step=False, on_epoch=True, logger=True)
        # self.log("edit_distance_beam_search", edit_distance_beam_search, on_step=False, on_epoch=True, logger=True)

        return

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

if __name__ == '__main__':
    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained("./data/tokenizer")

    MODEL_CONFIG = {
        "tokenizer": {
            "path": "./data/tokenizer"
        },
        "encoder" : {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                    # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                    "stage1": [48, 48, 128, 1, False, False, 3, 6],
                    "stage2": [128, 96, 512, 1, True, False, 3, 6],
                    "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                    "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
                },
            "hidden_size": 384,
            "pretrained_backbone": "./src/syntex/formula_net/formulanet_encoder_hgnetv2.pt",
            "freeze_backbone": True,
        },
        "decoder": {
            "vocab_size": tokenizer.vocab_size,
            "max_position_embeddings": 1024,
            "d_model": 384,
            "decoder_layers": 2,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 1536,
            "decoder_start_token_id": tokenizer.bos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "forced_eos_token_id": tokenizer.eos_token_id,
            "layer_norm_eps": 1e-05,
            "is_decoder": True,
        }
    }

    TRAINING_CONFIG = {
        "max_steps": 3e5,
        "optimizer": {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.05,


        },
        "lr_scheduler":{
            "min_lr": 1e-8,
            "num_warmup_steps": 5e3,
            "num_training_steps": 3e5,
        }
    }

    _ = FormulaNetLit(MODEL_CONFIG, TRAINING_CONFIG)
    print(_)