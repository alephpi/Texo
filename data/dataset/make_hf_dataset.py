from pathlib import Path
from datasets import Dataset, Features, Value
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
    parent_dir = Path(__file__).resolve().parent
    image_dir = parent_dir / "UniMER-1M_merged/images"
    text_path = parent_dir / "UniMER-1M_merged/train_normalized_.txt"
    save_dir = parent_dir / "hf_dataset"

    if not save_dir.exists():
        dataset = convert_to_hf_dataset(image_dir, text_path)
        print(dataset)
        dataset.save_to_disk(save_dir )
        print(f"HuggingFace dataset saved to {save_dir}")
    else:
        import time
        dataset = Dataset.load_from_disk(save_dir)
        print(f"Dataset size: {len(dataset)} samples")
        start_time = time.time()
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            img_bytes = sample["image"]
        total_time = time.time() - start_time
        print(f"Read {len(dataset)} samples in {total_time:.2f}s")
        print(f"Speed: {len(dataset)/total_time:.2f} samples/sec")
