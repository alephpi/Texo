# parallel to PPFormulaNet config
# see paddleocr/ppocr/modeling/heads/rec_ppformulanet_head.py and paddleocr/configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml
from transformers.models.mbart.modeling_mbart import MBartConfig, MBartForCausalLM

decoder_config = MBartConfig(
    vocab_size=50000,
    max_position_embeddings=1029,
    d_model=384,
    decoder_layers=2,
    decoder_attention_heads=16,
    decoder_ffn_dim=1536,
    decoder_start_token_id=0,
    layer_norm_eps=1e-05,
    is_decoder=True,
)

decoder = MBartForCausalLM(decoder_config)

if __name__ == '__main__':
    s = 0
    for p in decoder.parameters():
        s += p.numel()
    print(decoder)
    print(s / 1e6)