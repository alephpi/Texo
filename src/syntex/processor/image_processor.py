# adapted from https://github.com/ParaN3xus/my-unimernet/blob/46f7ee83bbb6a61c83776e0c33b300a17d02d1d2/unimernet/components/processor/image_processor.py
# and https://github.com/opendatalab/UniMERNet/blob/41659a39d683de1de9cdbcc6a941bf9a2efdd55e/unimernet/processors/formula_processor.py

import math
from typing import Dict

import albumentations as alb
import cv2
import numpy as np
import torchvision.transforms.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps

UNIMERNET_MEAN = (0.7931, 0.7931, 0.7931)
UNIMERNET_STD = (0.1738, 0.1738, 0.1738)

def resize(image: Image.Image, image_size: Dict[str, int], random_padding: bool = False):
    """
    Resize an image to `(size["height"], size["width"])` by thumbnailing and padding.
    """
    output_size = (image_size["height"], image_size["width"])

    image = F.resize(image, min(output_size))
    image.thumbnail((output_size[1], output_size[0]))
    delta_height = output_size[0] - image.height
    delta_width = output_size[1] - image.width
    if random_padding:
        pad_width = np.random.randint(low=0, high=delta_width + 1)
        pad_height = np.random.randint(low=0, high=delta_height + 1)
    else:
        pad_width = delta_width // 2
        pad_height = delta_height // 2
    padding = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    return ImageOps.expand(image, padding)

def crop_margin(img: Image.Image) -> Image.Image:
    data = np.array(img.convert("L"))
    data = data.astype(np.uint8)
    max_val = data.max()
    min_val = data.min()
    if max_val == min_val:
        return img
    data = (data - min_val) / (max_val - min_val) * 255
    gray = 255 * (data < 200).astype(np.uint8)

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text pixels)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    return img.crop((a, b, w + a, h + b))


class BaseMERImageProcessor:
    def __init__(self, image_size: Dict[str, int]) -> None:
        super().__init__()
        self.image_size = image_size
        self.transform: alb.Compose
    
    def __call__(self, image):
        return self.process(image)

    def process(self,image: Image.Image):
        raise NotImplementedError

class TrainMERImageProcessor(BaseMERImageProcessor):
    def __init__(self, image_size: Dict[str, int] = {'height': 384, 'width': 384}) -> None:
        super().__init__(image_size)
        from .image_augment import (
            Bitmap,
            Dilation,
            Erosion,
            Fog,
            Frost,
            Rain,
            Shadow,
            Snow,
        )
        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Snow(), Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.Affine(scale=(0.85, 1.0), rotate=(-1, 1), interpolation=3, pborder_mode=0, fill=[255, 255, 255], p=1),
                        alb.GridDistortion(distort_limit=0.1, interpolation=3, p=.5)],
                    p=.15),
                # alb.InvertImg(p=.4), #NOTE no InvertImg in UniMERNet
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                alb.GaussNoise(mean_range=(0, 0), std_range=(0, math.sqrt(10) / 255), p=0.2),
                alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                alb.ImageCompression(quality_range=(95, 100), p=.3),
                alb.ToGray(p=1), #NOTE ToGray always_apply=True for old-version alb in UniMERNet, which means p=1
                alb.Normalize(UNIMERNET_MEAN, UNIMERNET_STD),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    # TODO batch mode preprocessing
    def process(self, image: Image.Image):
        assert isinstance(image, Image.Image), "image must be a PIL.Image.Image"

        image_ = image if image.mode == "RGB" else image.convert("RGB")
        image_ = crop_margin(image)
        image_ = resize(image=image, image_size=self.image_size, random_padding=True)

        return self.transform(image_)['image']

class EvalMERImageProcessor(BaseMERImageProcessor):
    def __init__(self, image_size: Dict[str, int] = {'height': 384, 'width': 384}) -> None:
        super().__init__(image_size)
        self.transform = alb.Compose(
            [
                alb.ToGray(p=1),
                alb.Normalize(UNIMERNET_MEAN, UNIMERNET_STD),
                ToTensorV2(),
            ]
        )

    def process(self, image: Image.Image):
        assert isinstance(image, Image.Image), "image must be a PIL.Image.Image"

        image_ = image if image.mode == "RGB" else image.convert("RGB")
        image_ = crop_margin(image)
        image_ = resize(image=image, image_size=self.image_size, random_padding=False)

        return self.transform(image_)['image']