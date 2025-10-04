import argparse
from pathlib import Path
from datasets import Dataset, Features, Value, DatasetDict
from tqdm import tqdm



def convert_to_hf_dataset(image_dir, text_path, padding_digits=7):
    # 读文本
    with open(text_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f]

    samples = []
    for idx, text in tqdm(enumerate(texts), total=len(texts)):
        img_path = Path(image_dir) / f"{idx:0{padding_digits}d}.png"
        if not img_path.exists():
            print(f"Warning: {img_path} does not exist, skipping.")
            continue

        with open(img_path, "rb") as f:
            img_bytes = f.read()
        assert text, f"Text is empty for {img_path}"

        samples.append({
            "image": img_bytes,  # 存 PNG bytes
            "text": text,
        })

    features = Features({
        "image": Value("binary"),
        "text": Value("string"),
    })

    dataset = Dataset.from_list(samples, features=features)
    return dataset


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("command", type=str, choices=["build", "test", "push"])
    args = parser.parse_args()
    parent_dir = Path(__file__).resolve().parent
    train_save_dir = parent_dir / "hf_datasets" / "UniMER-Train"
    test_save_dir = parent_dir / "hf_datasets" / "UniMER-Test"
    if args.command == "build":
        if not train_save_dir.exists():
            train_image_dir = parent_dir / "UniMER-1M_merged/images"
            train_text_path = parent_dir / "UniMER-1M_merged/train_normalized_.txt"
            train_dataset = convert_to_hf_dataset(train_image_dir, train_text_path)
            # --- 顶层 DatasetDict ---
            print(train_dataset)
            train_dataset.save_to_disk(train_save_dir )
            print(f"HuggingFace dataset saved to {train_save_dir}")
        else:
            print(f"HuggingFace dataset already exists at {train_save_dir}")
        if not test_save_dir.exists():
            test_root = parent_dir / "UniMER-Test"
            test_splits = ["spe", "cpe", "sce", "hwe"]
            test_dict = DatasetDict()
            for split in test_splits:
                test_image_dir = test_root / split
                test_text_path = test_root / f"{split}.txt"
                test_dict[split] = convert_to_hf_dataset(test_image_dir, test_text_path)
            test_dataset = DatasetDict(test_dict)
            print(test_dataset)
            test_dataset.save_to_disk(test_save_dir)
            print(f"HuggingFace dataset saved to {test_save_dir}")
        else:
            print(f"HuggingFace dataset already exists at {test_save_dir}")
    elif args.command == "test":
        import time
        train_dataset = Dataset.load_from_disk(train_save_dir)
        print(f"Dataset size: {len(train_dataset)} samples")
        start_time = time.time()
        for idx, sample in tqdm(enumerate(train_dataset), total=len(train_dataset)):
            img_bytes = sample["image"]
        total_time = time.time() - start_time
        print(f"Read {len(train_dataset)} samples in {total_time:.2f}s")
        print(f"Speed: {len(train_dataset)/total_time:.2f} samples/sec")

        test_dataset = DatasetDict.load_from_disk(test_save_dir)
        for name, split in test_dataset.items():
            start_time = time.time()
            for idx, sample in tqdm(enumerate(split), total=len(split)):
                img_bytes = sample["image"]
            total_time = time.time() - start_time
            print(f"Read {len(split)} samples in {total_time:.2f}s")
            print(f"Speed: {len(split)/total_time:.2f} samples/sec")
    elif args.command == "push":
        # push to hf hub
        dataset = Dataset.load_from_disk(train_save_dir)
        dataset.push_to_hub("alephpi/UniMER-Train")
        dataset = DatasetDict.load_from_disk(test_save_dir)
        dataset.push_to_hub("alephpi/UniMER-Test")
    else:
        raise ValueError(f"Unknown command {args.command}")
