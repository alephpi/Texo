from .dataset import MERDataset, MERDatasetHF
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from ..processor import (
    EvalMERImageProcessor,
    TextProcessor,
    TrainMERImageProcessor,
)
from .sampler import BucketBatchSampler, SortedSampler

from lightning import LightningDataModule

class MERDataModule(LightningDataModule):
    def __init__(self, data_config: dict):
        super().__init__()
        self.data_config = data_config
        self.save_hyperparameters()
        sampling_strategy: str = self.data_config.get("sampling_strategy",'')
        sampling_strategies = {
            "random": RandomSampler,
            "sorted": SortedSampler,
            "sequential": SequentialSampler
        }
        self.sampler = sampling_strategies.get(sampling_strategy, None)

    def setup(self, stage=None):
        self.train_dataset = MERDatasetHF(
                                dataset_path=self.data_config["train_dataset_path"],
                                image_processor=TrainMERImageProcessor(**self.data_config["image_processor"]),
                                text_processor=TextProcessor(self.data_config["text_processor"])
                                )

        self.val_dataset = MERDataset(
                                image_dir=self.data_config["eval_image_dir"],
                                text_path=self.data_config["eval_text_path"],
                                image_processor=EvalMERImageProcessor(**self.data_config["image_processor"]),
                                text_processor=TextProcessor(self.data_config["text_processor"])
                                )

        self.test_dataset = MERDataset(
                                image_dir=self.data_config["test_image_dir"],
                                text_path=self.data_config["test_text_path"],
                                image_processor=EvalMERImageProcessor(**self.data_config["image_processor"]),
                                text_processor=TextProcessor(self.data_config["text_processor"])
                                )

    def train_dataloader(self):
        if self.sampler is None:
            train_loader = DataLoader(
                                dataset=self.train_dataset,
                                batch_size=self.data_config["train_batch_size"],
                                shuffle=True,
                                drop_last=False,
                                num_workers=self.data_config["num_workers"],
                                collate_fn=self.train_dataset.collate_fn,
                                pin_memory=True,
                                persistent_workers=True
                                )
        elif self.sampler == SortedSampler:
            train_loader = DataLoader(
                                dataset=self.train_dataset,
                                batch_sampler=BucketBatchSampler(
                                    sampler=self.sampler(data_source=self.train_dataset.labels_length, sort_key=lambda idx: idx),
                                    batch_size=self.data_config["train_batch_size"],
                                    drop_last=False,
                                    sort_key=lambda idx: self.train_dataset.labels_length[idx]
                                ),
                                num_workers=self.data_config["num_workers"],
                                collate_fn=self.train_dataset.collate_fn,
                                pin_memory=True,
                                persistent_workers=True
                                )
        else:
            train_loader = DataLoader(
                                dataset=self.train_dataset,
                                batch_sampler=BucketBatchSampler(
                                    sampler=self.sampler(data_source=self.train_dataset),
                                    batch_size=self.data_config["train_batch_size"],
                                    drop_last=False,
                                    sort_key=lambda idx: self.train_dataset.labels_length[idx]
                                ),
                                num_workers=self.data_config["num_workers"],
                                collate_fn=self.train_dataset.collate_fn,
                                pin_memory=True,
                                persistent_workers=True
                                )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
                            dataset=self.val_dataset,
                            batch_size=self.data_config["val_batch_size"],
                            sampler=SortedSampler(self.val_dataset.labels_length, sort_key=lambda x: x),
                            num_workers=self.data_config["num_workers"],
                            collate_fn=self.val_dataset.collate_fn,
                            pin_memory=True,
                            persistent_workers=True
                            )
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(
                            dataset=self.test_dataset,
                            batch_size=self.data_config["test_batch_size"],
                            sampler=SortedSampler(self.test_dataset.labels_length, sort_key=lambda x: x),
                            num_workers=self.data_config["num_workers"],
                            collate_fn=self.val_dataset.collate_fn,
                            pin_memory=True,
                            persistent_workers=True
                            )
        return test_loader
