from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from syntex.data.dataset import MERDataset, MERDatasetHF
from syntex.data.processor import (
    EvalMERImageProcessor,
    TextProcessor,
    TrainMERImageProcessor,
)
from syntex.data.sampler import BucketBatchSampler, SortedSampler


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
        if stage == "fit":
            self.train_dataset = MERDatasetHF(
                                    dataset_path=self.data_config["train_dataset_path"],
                                    image_processor=TrainMERImageProcessor(**self.data_config["image_processor"]),
                                    text_processor=TextProcessor(self.data_config["text_processor"]),
                                    filtered=self.data_config["filter_train"]
                                    )

            self.val_dataset = MERDatasetHF(
                                    dataset_path=self.data_config["eval_dataset_path"],
                                    image_processor=EvalMERImageProcessor(**self.data_config["image_processor"]),
                                    text_processor=TextProcessor(self.data_config["text_processor"])
                                    )
        elif stage == "test":
            self.test_dataset_names = []
            self.test_datasets = []
            for test_dataset_path in Path(self.data_config["test_dataset_paths"]).iterdir():
                if test_dataset_path.is_dir():
                    self.test_dataset_names.append(test_dataset_path.name)
                    self.test_datasets.append(MERDatasetHF(
                                        dataset_path=test_dataset_path,
                                        image_processor=EvalMERImageProcessor(**self.data_config["image_processor"]),
                                        text_processor=TextProcessor(self.data_config["text_processor"])
                                        ))

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
        test_loaders: list[DataLoader] = []
        for test_dataset in self.test_datasets:
            test_loaders.append(DataLoader(
                                dataset=test_dataset,
                                batch_size=self.data_config["test_batch_size"],
                                sampler=SortedSampler(test_dataset.labels_length, sort_key=lambda x: x),
                                num_workers=self.data_config["num_workers"],
                                collate_fn=test_dataset.collate_fn,
                                ))
        return test_loaders
