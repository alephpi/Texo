# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for ViT."""

from typing import Dict, List, Optional, Union

import numpy as np

from torch import TensorType
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import filter_out_non_signature_kwargs, logging
from PIL import Image, ImageOps
import cv2
from torchvision.transforms.functional import resize
from albumentations.pytorch import ToTensorV2
import albumentations as alb
from .image_utils.nougat import Bitmap, Dilation, Erosion
from .image_utils.weathers import Fog, Frost, Snow, Rain, Shadow
import math

logger = logging.get_logger(__name__)


class UniMERNetBaseImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)
        self.size = size
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def resize(
        self,
        image: Image.Image,
        size: Dict[str, int],
        random_padding: False,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])` by thumbnailing and padding.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(
                f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])

        image = resize(image, min(output_size))
        image.thumbnail((output_size[1], output_size[0]))
        delta_width = output_size[1] - image.width
        delta_height = output_size[0] - image.height
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

    @staticmethod
    def crop_margin(img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            logger.warning_once(
                "Passing incorrect img to crop_margin!"
            )
            return image

        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        # Find minimum spanning bounding box
        a, b, w, h = cv2.boundingRect(coords)
        return img.crop((a, b, w + a, h + b))

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        size: Dict[str, int] = None,
        random_padding: bool = False,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
        """
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=False,
            rescale_factor=1,
            do_normalize=True,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=False,
            size=size,
            resample=None,
        )
        # convert to rgb
        images = [convert_to_rgb(image) for image in images]

        # crop margins
        images = [self.crop_margin(image) for image in images]

        # resize
        images = [
            self.resize(image=image, size=size_dict,
                        random_padding=random_padding)
            for image in images
        ]

        return images

    def transform(self):
        return alb.Compose(
            [
                alb.ToGray(),
                alb.Normalize(self.image_mean, self.image_std),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    @staticmethod
    def postprocess(images, return_tensors):
        images = [(image.repeat(3, 1, 1) if image.shape[0]
                   == 1 else image) for image in images]

        images = [image.numpy() for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)


class UniMERNetTrainImageProcessor(UniMERNetBaseImageProcessor):
    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(size=size, image_mean=image_mean, image_std=image_std, **kwargs)

    def transform(self):
        return alb.Compose(
            [
                alb.Compose(
                    [
                        Bitmap(p=0.05),
                        alb.OneOf([Fog(), Frost(), Snow(),
                                   Rain(), Shadow()], p=0.2),
                        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.2),
                        alb.Affine(
                            scale=(0.85, 1.0),
                            rotate=(-1, 1),
                            interpolation=3,
                            border_mode=0,
                            fill=[255, 255, 255],
                            p=1
                        ),
                        alb.GridDistortion(distort_limit=0.1, interpolation=3, p=.5)],
                    p=.15),
                alb.InvertImg(p=.4),
                alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                             b_shift_limit=15, p=0.3),
                alb.GaussNoise(
                    mean_range=(0, 0),
                    std_range=(0, math.sqrt(10) / 255),
                    p=0.2
                ),
                alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
                alb.ImageCompression(quality_range=(95, 100), p=.3),
                alb.ToGray(),
                alb.Normalize(self.image_mean, self.image_std),
                # alb.Sharpen()
                ToTensorV2(),
            ]
        )

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        size: Dict[str, int] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        images = super().preprocess(images=images, size=size, random_padding=False,
                                    image_mean=image_mean, image_std=image_std)

        images = [self.transform()(image=np.array(image))[
            'image'][:1] for image in images]

        return super().postprocess(images, return_tensors)


class UniMERNetEvalImageProcessor(UniMERNetBaseImageProcessor):
    def __init__(
        self,
        size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(size=size, image_mean=image_mean, image_std=image_std, **kwargs)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        size: Dict[str, int] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        images = super().preprocess(images=images, size=size, random_padding=False,
                                    image_mean=image_mean, image_std=image_std)

        images = [self.transform()(image=np.array(image))[
            'image'][:1] for image in images]

        return super().postprocess(images, return_tensors)
