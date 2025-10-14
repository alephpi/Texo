import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from optimum.onnxruntime import ORTModelForVision2Seq
from texo.data.processor import EvalMERImageProcessor
from texo.utils.config import *

def validate_encoder(torch_model, onnx_model, inputs):
    with torch.no_grad():
        torch_encoder_output = torch_model.encoder(inputs).last_hidden_state
        onnx_encoder_output = onnx_model.encoder(inputs).last_hidden_state

    diff = torch.abs(torch_encoder_output - onnx_encoder_output)
    print(f"最大差异: {diff.max().item():.6e}")

def validate_generate(torch_model, onnx_model, inputs):
    with torch.no_grad():
        torch_output = torch_model.generate(inputs)
        onnx_output = onnx_model.generate(inputs)
    diff = torch.abs(torch_output-onnx_output)
    print(f"最大差异: {diff.max().item():.6e}")

def main():
    torch_model = VisionEncoderDecoderModel.from_pretrained("./model")
    onnx_model = ORTModelForVision2Seq.from_pretrained('./model/trio_onnx')

    image_path = "./TechnoSelection/test_img/单行公式2.png"
    image = Image.open(image_path)
    image_processor = EvalMERImageProcessor(image_size={'width':384, 'height':384})
    processed_image = image_processor(image).unsqueeze(0)
    

    # 验证encoder输出
    encoder_valid = validate_encoder(torch_model, onnx_model, processed_image)
    
    # 验证生成任务
    generate_valid = validate_generate(torch_model, onnx_model, processed_image)

if __name__ == '__main__':
    main()