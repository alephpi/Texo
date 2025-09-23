from pathlib import Path

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch.optim.optimizer import Optimizer
from transformers import (
    PreTrainedTokenizerFast,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from .config import OmegaConf
from .scores import compute_bleu, compute_edit_distance


class FormulaNetLit(LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config

        self.save_hyperparameters()

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_config.pop("tokenizer"))
        self.model = VisionEncoderDecoderModel(VisionEncoderDecoderConfig(**model_config))
        # transformers.VisionEncoderDecoderModel is not smart enough to initialize from the encoder and decoder
        # where we have to initialize the model.config manually as the following.
        self.model.config.decoder_start_token_id = self.model.decoder.config.bos_token_id
        self.model.config.pad_token_id = self.model.decoder.config.pad_token_id
        self.model.config.eos_token_id = self.model.decoder.config.eos_token_id
        # print(self.model.decoder.config)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.training_config["optimizer"])
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(self.optimizer, **self.training_config["lr_scheduler"])

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
        self.log("seq_len", batch["labels"].size(-1), on_step=True, on_epoch=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if not self.trainer.fit_loop.epoch_loop.done:
            loss = self.model_step(batch, batch_idx)
            self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        
        else:
            self.score_step(batch, batch_idx)

    def score_step(self, batch, batch_idx):
        labels = batch["labels"]
        labels[labels == -100] = self.model.config.pad_token_id
        ref_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        max_length = labels.shape[-1] # in validation, since we know how long the ground truth is, we truncate to it to save computation.

        outputs = self.generate(batch["pixel_values"], num_beams=1, do_sample=False, max_length=max_length)
        pred_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        bleu = compute_bleu(pred_str, ref_str)
        edit_distance = compute_edit_distance(pred_str, ref_str)

        self.log("BLEU", bleu, on_step=False, on_epoch=True, logger=True)
        self.log("edit_distance", edit_distance, on_step=False, on_epoch=True, logger=True)
        return
    
    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    #     # log gradient norms
    #     norms = grad_norm(self.model.decoder, norm_type=2)
    #     self.log_dict(norms)
    
    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        if self.current_epoch > 1:
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
            # if do in on_before_optimizer_step, it will log gradient norms before the clip
        norms = grad_norm(self.model.decoder, norm_type=2)
        self.log_dict(norms)
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

    import pprint

    import torchinfo
    config = OmegaConf.load("./config/train.yaml")
    OmegaConf.resolve(config.model)
    config.model.pop("tokenizer")
    model = VisionEncoderDecoderModel(VisionEncoderDecoderConfig(**OmegaConf.to_container(config.model)))

    input_data = {
        "pixel_values": torch.randn(16, 3, 384, 384, dtype=torch.float32),
        "decoder_input_ids":torch.randint(0, config.model.decoder.vocab_size, (16, 256), dtype=torch.int32), 
        "decoder_attention_mask": torch.ones(16, 256, dtype=torch.int32),
    }
    torchinfo.summary(model, input_data=input_data, depth=4, col_names=["input_size", "output_size", "num_params", "params_percent"])
    pprint.pprint(model.config)
