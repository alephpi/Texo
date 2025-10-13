import torch
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from ..utils.config import hydra, OmegaConf


class FormulaNet(VisionEncoderDecoderModel):
    def __init__(self, config):
        super().__init__(VisionEncoderDecoderConfig(**config))
        if ckpt_path := config.get("pretrained"):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=True)
            # transformers.VisionEncoderDecoderModel is not smart enough to
            # initialize the model.config manually as the following.
            self.config.decoder_start_token_id = self.decoder.config.bos_token_id
            self.config.pad_token_id = self.decoder.config.pad_token_id
            self.config.eos_token_id = self.decoder.config.eos_token_id

@hydra.main(version_base="1.3.2",config_path="../../../config", config_name="train.yaml")
def main(cfg):
    import torchinfo
    OmegaConf.set_struct(cfg, False)
    model = FormulaNet(cfg.model)
    batch_size = cfg.data.train_batch_size
    dummy_input = {
        "pixel_values": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32),
        "decoder_input_ids":torch.randint(0, model.decoder.config.vocab_size, (batch_size, 256), dtype=torch.int32), 
        "decoder_attention_mask": torch.ones(batch_size, 256, dtype=torch.int32),
    }
 
    torchinfo.summary(model, input_data=dummy_input, depth=4, col_names=["input_size", "output_size", "num_params", "params_percent"])

if __name__ == '__main__':
    main()