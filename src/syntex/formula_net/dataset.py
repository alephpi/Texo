import os
from pathlib import Path
from typing import Optional

import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..processor import (
    BaseMERImageProcessor,
    EvalMERImageProcessor,
    TextProcessor,
    TrainMERImageProcessor,
)


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
        self.padding_digits = 7 # for UniMER-1M
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
        labels = self.text_processor(texts).input_ids
        labels[labels == self.text_processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": images,
            "labels": labels,
        }

class MERDataModule(LightningDataModule):
    def __init__(self, data_config: dict):
        super().__init__()
        self.data_config = data_config
        self.save_hyperparameters()
        self.train_dataset: Optional[MERDataset] = None
        self.val_dataset: Optional[MERDataset] = None
        self.test_dataset: Optional[MERDataset] = None
    
    def setup(self, stage=None):
        self.train_dataset = MERDataset(
                                image_dir=self.data_config["train_image_dir"],
                                text_path=self.data_config["train_text_path"],
                                image_processor=TrainMERImageProcessor(**self.data_config["image_processor"]),
                                text_processor=TextProcessor(self.data_config["text_processor"])
                                )

        self.val_dataset = MERDataset(
                                image_dir=self.data_config["eval_image_dir"],
                                text_path=self.data_config["eval_text_path"],
                                image_processor=EvalMERImageProcessor(**self.data_config["image_processor"]),
                                text_processor=TextProcessor(self.data_config["text_processor"])
                                )
    
    def train_dataloader(self):
        train_loader = DataLoader(
                            dataset=self.train_dataset,
                            batch_size=self.data_config["train_batch_size"],
                            shuffle=True,
                            num_workers=self.data_config["num_workers"],
                            collate_fn=self.train_dataset.collate_fn,
                            persistent_workers=True
                            )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
                            dataset=self.val_dataset,
                            batch_size=self.data_config["val_batch_size"],
                            shuffle=False,
                            num_workers=self.data_config["num_workers"],
                            collate_fn=self.val_dataset.collate_fn,
                            persistent_workers=True
                            )
        return val_loader
