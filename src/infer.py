import torch

from syntex.formula_net.dataset import MERDataset
from syntex.formula_net.model import FormulaNetLit
from syntex.processor import EvalMERImageProcessor, TextProcessor

MODEL_CONFIG = {
    "tokenizer_path": "./data/tokenizer",
    "encoder": {
        "stem_channels": [3, 32, 48],
        "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
        "hidden_size": 384,
        "pretrained_backbone": None,
        "freeze_backbone": True,
    },
    "decoder": {
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

def load_model(ckpt_path):
    model = FormulaNetLit.load_from_checkpoint(ckpt_path, model_config = MODEL_CONFIG)
    model.eval()
    return model

def load_plain_model():
    model = FormulaNetLit(model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG)
    model.eval()
    return model

def inference(model: FormulaNetLit, input_data: torch.Tensor, device):
    model.to(device)
    input_data = input_data.to(device)
    output_ids = model.generate(pixel_values=input_data, max_length=256, num_beams=1, 
                                # bad_words_ids=[[2]]
                                )
    output_str = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_ids, output_str

def main():
    # 配置
    # checkpoint_path = "outputs/step=2000-val_loss=4.9977e-02-BLEU=0.0000-edit_distance=0.0000.ckpt"  # 预训练模型 checkpoint 路径
    # checkpoint_path = "outputs/step=95000-val_loss=0.0000-BLEU=0.0000-edit_distance=0.0000.ckpt"  # 预训练模型 checkpoint 路径
    checkpoint_path = "outputs/step=152000-val_loss=1.0339-BLEU=0.5015-edit_distance=0.4442.ckpt"  # 预训练模型 checkpoint 路径

    # 检查是否有可用的GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path)
    # model = load_plain_model()
    text_file = "data/dataset/UniMER-Test/cpe.txt"  # 输入图片路径
    image_path = "data/dataset/UniMER-Test/cpe"
    dataset = MERDataset(image_dir=image_path, text_path=text_file, image_processor=EvalMERImageProcessor(), text_processor=TextProcessor())
    data = dataset[144]
    output_ids, output_str = inference(model, data['pixel_values'].unsqueeze(0), device)
    print(f"{output_ids}\n{output_str}\n{data['text']}")

if __name__ == "__main__":
    main()
