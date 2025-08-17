# convert HME100k to UniMER-1M format
# adapt from https://github.com/opendatalab/UniMERNet/issues/14

import os
import os.path as osp
import random
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm

random.seed(0)

# convert train/test set
caption_file = osp.join("./HME100K/train_labels.txt")
jpg_dir = osp.join("./HME100K/train_images")

# read lines from caption file
with open(caption_file, "r") as f:
    lines = f.read().splitlines()  # format: image_name(no ext) \t caption

samples = []
for line in lines:
    image_name, caption = line.split("\t")
    image_path = osp.join(jpg_dir, image_name)
    samples.append((image_path, caption))
random.shuffle(samples)

output_txt = osp.join(osp.dirname(caption_file), "train_formulae.txt")
output_png_dir = osp.join(osp.dirname(caption_file), "train_images_png")

if not osp.exists(output_png_dir):
    os.makedirs(output_png_dir)


def process_sample(args):
    i, (image_path, caption) = args
    try:
        img = Image.open(image_path)
        new_img_basename = f"{i:07d}.png"
        img.save(osp.join(output_png_dir, new_img_basename))
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


if __name__ == '__main__':
    n_cpus = os.cpu_count()
    with Pool(n_cpus) as pool:
        results = list(tqdm(pool.imap(process_sample, enumerate(samples)), total=len(samples)))

    with open(output_txt, "w") as f:
        for caption in results:
            if caption is not None:
                f.write(caption + "\n")
