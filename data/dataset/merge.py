# merge two datasets
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

random.seed(0)

image_dirs = [
    "./HME100K/train_images_png",
    "./UniMER-1M/images"
]
equation_txts = [
    "./HME100K/train_formulae.txt",
    "./UniMER-1M/train.txt"
]

output_dir = "./UniMER-1M_merged/images"
os.makedirs(output_dir, exist_ok=True)
output_txt = "./UniMER-1M_merged/train.txt"

image_equation_pairs = []
for image_dir, equations_txt in zip(image_dirs, equation_txts):
    with open(equations_txt, "r") as f:
        equations = f.readlines()
    pbar = tqdm(enumerate(equations), total=len(equations))
    for i, equation in pbar:
        equation = equation.strip()
        if equation == "":
            continue
        image_name = "0" * (7 - len(str(i))) + str(i) + ".png"
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_equation_pairs.append((image_path, equation))

print("total pairs:", len(image_equation_pairs))
random.shuffle(image_equation_pairs)
with open(output_txt, "w") as f:
    for image_path, equation in image_equation_pairs:
        f.write(equation + "\n")

def link_image(i_image_pair):
    i, (image_path, equation) = i_image_pair
    image_name = os.path.basename(image_path)
    new_image_name = "0" * (7 - len(str(i))) + str(i) + ".png"
    new_image_path = os.path.join(output_dir, new_image_name)
    os.link(image_path, new_image_path)


pbar = tqdm(enumerate(image_equation_pairs), total=len(image_equation_pairs))
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(link_image, item) for item in pbar]
    for future in tqdm(as_completed(futures), total=len(futures)):
        pass