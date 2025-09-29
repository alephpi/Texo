from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

from ..utils.config import OmegaConf


class FormulaNet(VisionEncoderDecoderModel):
    def __init__(self, config):
        super().__init__(VisionEncoderDecoderConfig(**config))
        if config.pretrained:
            state_dict = torch.load(config.pretrained, map_location=self.device)
            self.load_state_dict(state_dict, strict=True)
        # transformers.VisionEncoderDecoderModel is not smart enough to
        # initialize the model.config manually as the following.
        self.config.decoder_start_token_id = self.decoder.config.bos_token_id
        self.config.pad_token_id = self.decoder.config.pad_token_id
        self.config.eos_token_id = self.decoder.config.eos_token_id

def load_model(config_path="/home/mao/workspace/SynTeX/config/model/FormulaNet-S.yaml"):
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config.model)
    model = FormulaNet(OmegaConf.to_container(config.model))

    return model


if __name__ == '__main__':
    import pprint

    import torch
    import torchinfo

    model = load_model()
    dummy_input = {
        "pixel_values": torch.randn(16, 3, 384, 384, dtype=torch.float32),
        "decoder_input_ids":torch.randint(0, model.decoder.config.vocab_size, (16, 256), dtype=torch.int32), 
        "decoder_attention_mask": torch.ones(16, 256, dtype=torch.int32),
    }
 
    torchinfo.summary(model, input_data=dummy_input, depth=4, col_names=["input_size", "output_size", "num_params", "params_percent"])
    pprint.pprint(model.config)
 