import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..processor import BaseMERImageProcessor, TextProcessor


class MERDataset(Dataset):
    def __init__(self, image_dir: str, text_path: str, image_processor: BaseMERImageProcessor, text_processor: TextProcessor):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.text_processor = text_processor

        # 读取文本数据
        with open(text_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        self.texts = [text.strip() for text in self.texts]  # 去除换行符和首尾空格

        # 验证图像文件是否存在
        self.padding_digits = len(str(len(self.texts)))
        self.valid_indices = []
        for idx in range(len(self.texts)):
            img_path = Path(image_dir) / f"{idx:0{self.padding_digits}d}.png"
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                print(f"{img_path=} does not exist.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        idx = self.valid_indices[index]

        img_path = Path(self.image_dir) / f"{idx:0{self.padding_digits}d}.png"
        image = Image.open(img_path)
        processed_image = self.image_processor(image)

        text = self.texts[idx]
        # 去除batch维度
        
        return {
            'pixel_values': processed_image,
            'text': text
        }

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        images = torch.stack([item["pixel_values"] for item in batch])

        # batch-process text in collate_fn
        text_encoding = self.text_processor(texts)
        # we remove the bos token `<s>` to construct the labels
        # since the transformers.decoder will automatically shift it to the right as the decoder_input_ids
        labels = text_encoding["input_ids"][:, 1:]
        attention_mask = text_encoding["attention_mask"][:, 1:]
        return {
            "pixel_values": images,
            "labels": labels,
            "attention_mask": attention_mask
        }