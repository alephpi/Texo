# dataset.py alternative implementation with HF Datasets
import io
import os
from pathlib import Path
from datasets import Dataset
import numpy as np
import torch
from PIL import Image
from ..processor.image_processor import preprocess, BaseMERImageProcessor, TrainMERImageProcessor, EvalMERImageProcessor
from ..processor.text_processor import TextProcessor

class MERDataset:
    def __init__(self, dataset_path: str, image_processor: BaseMERImageProcessor, text_processor: TextProcessor):
        self.image_processor = image_processor
        self.text_processor = text_processor
        
        self.dataset = Dataset.load_from_disk(dataset_path)
        self.dataset = self.dataset.map(self._preprocess, remove_columns=["image"], batched=True)

    def _preprocess(self, samples):
        """
        image preprocessing for every sample in the dataset, can be batched
        """
        samples["pixel_values"] = []
        for img_bytes in samples["image"]:
            img = Image.open(io.BytesIO(img_bytes))
            img = preprocess(img, self.image_processor.image_size)
            img = np.array(img)
            samples["pixel_values"].append(img)

        return samples

    def __len__(self) -> int:
        """让 len(dataset) 可以正常工作"""
        return len(self.dataset)

    def __getitem__(self, index: int):
        """让 dataset[i] 可以正常工作"""
        # 直接调用底层的 HF dataset 的 __getitem__
        # 由于 set_transform 已经设置好了，这里会自动进行预处理
        return self.dataset[index]

class CustomDataCollator:
    def __init__(self, text_processor):
        self.text_processor = text_processor
        self.pad_token_id = int(self.text_processor.tokenizer.pad_token_id)

    def __call__(self, batch):
        # batch 是一个列表，每个元素是 preprocess_function 的输出
        # e.g., [{'pixel_values': tensor, 'text': str}, {'pixel_values': tensor, 'text': str}, ...]
        
        texts = [item["text"] for item in batch]
        images = torch.stack([item["pixel_values"] for item in batch])

        # 批量处理文本
        res = self.text_processor(texts)
        input_ids: torch.Tensor = res["input_ids"] #type: ignore
        attention_mask = res["attention_mask"]

        labels = input_ids.new_zeros(input_ids.shape)
        labels[:, :-1] = input_ids[:, 1:].clone()
        labels[:, -1] = self.pad_token_id
        labels[labels == self.pad_token_id] = -100

        return {
            "pixel_values": images,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": attention_mask,
            "labels": labels,
        }