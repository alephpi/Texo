import io
import os
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset

from .processor import (
    BaseMERImageProcessor,
    TextProcessor,
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
        self.labels_length = [0] * len(self.texts)

        # 验证图像文件是否存在
        self.padding_digits = 7 # for UniMER-1M
        self.valid_indices = []
        # self.text_length = [] # store the token numbers of each text label for bucketing
        for idx in range(len(self.texts)):
            img_path = Path(image_dir) / f"{idx:0{self.padding_digits}d}.png"
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
            else:
                ...
                # print(f"{img_path=} does not exist.")

            text = self.texts[idx]
            self.labels_length[idx] = len(text.split(' ')) + 1 # labels 1 longer than text due to eos token
        self.pad_token_id = int(self.text_processor.tokenizer.pad_token_id)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int):
        idx = self.valid_indices[index]

        img_path = Path(self.image_dir) / f"{idx:0{self.padding_digits}d}.png"
        image = Image.open(img_path)
        processed_image = self.image_processor(image)
        text = self.texts[idx]

        return {
            'pixel_values': processed_image,
            'text': text,
        }

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        images = torch.stack([item["pixel_values"] for item in batch])

        # batch-process text in collate_fn
        res = self.text_processor(texts)
        input_ids: torch.Tensor = res["input_ids"] #type: ignore
        attention_mask = res["attention_mask"]
        # NOTE Unlike many seq2seq models tutorials, we don't apply token shift for labels
        # i.e. we keep the first token as the start token, since MBart decoder compute loss without token shift, see https://github.com/huggingface/transformers/issues/10480
        # also by careful debugging, we found the output.logits of MBartForCausalLM is aligned with the decoder_input_ids, not the shifted label
        # shift to left
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

class MERDatasetHF(Dataset):
    def __init__(self, dataset_path, image_processor: BaseMERImageProcessor, text_processor: TextProcessor, filtered=False):
        self.dataset = HFDataset.load_from_disk(dataset_path)
        self.image_processor = image_processor
        self.text_processor = text_processor

        self.pad_token_id = int(self.text_processor.tokenizer.pad_token_id)
        self.unk_token_id = int(self.text_processor.tokenizer.unk_token_id)
        # filter out items with <unk> token from training data
        if filtered:
            filtered_dataset = []
            for item in self.dataset:
                input_ids = self.text_processor.tokenizer(
                    item["text"], 
                    add_special_tokens=False,
                )["input_ids"]
                if self.unk_token_id not in input_ids:
                    filtered_dataset.append(item)
            print(f"Dataset filtered, from {len(self.dataset)} items to {len(filtered_dataset)} items.")
            self.dataset = filtered_dataset
        else:
            self.dataset = list(self.dataset) # we can afford loading the whole dataset into memory, otherwise comment it out
        self.labels_length = [len(item['text'].split(' ')) + 1 for item in self.dataset]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image = Image.open(io.BytesIO(self.dataset[index]['image']))
        processed_image = self.image_processor(image)
        text = self.dataset[index]['text']

        return {
            'pixel_values': processed_image,
            'text': text,
        }

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        images = torch.stack([item["pixel_values"] for item in batch])

        # batch-process text in collate_fn
        res = self.text_processor(texts)
        input_ids: torch.Tensor = res["input_ids"] #type: ignore
        attention_mask = res["attention_mask"]
        # labels is input_ids shifted to the left, we don't use the default behavior of transformers as it is somehow complicated
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
