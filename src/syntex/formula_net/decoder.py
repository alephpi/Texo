# parallel to PPFormulaNet config
# see paddleocr/ppocr/modeling/heads/rec_ppformulanet_head.py and paddleocr/configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml
from transformers.models.mbart.modeling_mbart import MBartConfig, MBartForCausalLM

if __name__ == '__main__':
    import torchinfo

    default_decoder_config = MBartConfig(
        vocab_size=1000,
        max_position_embeddings=1024,
        d_model=384,
        decoder_layers=2,
        decoder_attention_heads=16,
        decoder_ffn_dim=1536,
        decoder_start_token_id=0,
        layer_norm_eps=1e-05,
        is_decoder=True,
        tie_word_embeddings=False,
    )

    decoder = MBartForCausalLM(default_decoder_config)
    torchinfo.summary(decoder)
