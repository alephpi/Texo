from pathlib import Path
from tqdm import tqdm
import torch
from optimum.onnxruntime import ORTModelForVision2Seq
import onnxruntime as ort

# 设置为只显示 Error 级别，忽略 Warning
ort.set_default_logger_severity(3)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal


from texo.data.dataset import MERDatasetHF
from texo.data.processor import EvalMERImageProcessor, TextProcessor
from texo.utils.scores import compute_bleu, compute_edit_distance
from texo.utils.config import *

test_dataset_paths = Path("./data/dataset/hf_datasets/UniMER-Test")
test_dataset_names = []
test_datasets = []
image_processor = EvalMERImageProcessor(image_size={"height":384,"width":384})
text_processor = TextProcessor(config={
    "tokenizer_path": "data/tokenizer",
    "tokenizer_config":{
        "add_special_tokens": True,
        "max_length": 1024,
        "padding": "longest",
        "truncation": True,
        "return_tensors": "pt",
        "return_attention_mask": False,
    }
})

for test_dataset_path in test_dataset_paths.iterdir():
    if test_dataset_path.is_dir():
        test_dataset_names.append(test_dataset_path.name)
        test_datasets.append(MERDatasetHF(
                            dataset_path=test_dataset_path,
                            image_processor=EvalMERImageProcessor(),
                            text_processor=TextProcessor()
                            ))

test_loaders= []
for test_dataset in test_datasets:
    test_loaders.append(torch.utils.data.DataLoader(
                        dataset=test_dataset,
                        batch_size=1,
                        num_workers=10,
                        ))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
onnx_model = ORTModelForVision2Seq.from_pretrained('./model/onnx')
onnx_model.to(device)

for idx, test_loader in enumerate(test_loaders):
    total_bleu = 0
    total_edit_distance = 0
    pbar = tqdm(test_loader, total=len(test_loader), desc=f"Test on {test_dataset_names[idx]}")
    for i, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        ref_str = text_processor.tokenizer.batch_decode(text_processor(batch['text']).input_ids, skip_special_tokens=True)
        outputs = onnx_model.generate(pixel_values)
        pred_str = text_processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        bleu = compute_bleu(pred_str, ref_str)
        edit_distance = compute_edit_distance(pred_str, ref_str)
        total_bleu += bleu
        total_edit_distance += edit_distance
        pbar.set_postfix({
            'BLEU': f'{total_bleu/(i+1):.4f}',
            'Edit-Dist': f'{total_edit_distance/(i+1):.4f}'
        })

        # print(ref_str[0])
        # print(pred_str[0])
        # if idx == 5:
        #     break
    print(test_dataset_names[idx] ,total_bleu/len(test_loader), total_edit_distance/len(test_loader))
