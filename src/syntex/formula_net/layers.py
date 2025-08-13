"""
Adapted from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math

import torch
import torch.nn as nn


# from https://github.com/frotms/PaddleOCR2Pytorch/blob/702c805136d7224884d9c9e032949e35533233b4/pytorchocr/modeling/backbones/rec_pphgnetv2.py#L1065
class PaddingSameAsPaddleMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = torch.nn.MaxPool2d(kernel_size, stride, padding=0, ceil_mode=True)

    def forward(self, x):
        _, _, h, w = x.shape
        pad_h_total = max(0, (math.ceil(h / self.stride) - 1) * self.stride + self.kernel_size - h)
        pad_w_total = max(0, (math.ceil(w / self.stride) - 1) * self.stride + self.kernel_size - w)
        pad_h = pad_h_total // 2
        pad_w = pad_w_total // 2
        x = torch.nn.functional.pad(x, [pad_w, pad_w_total - pad_w, pad_h, pad_h_total - pad_h])
        return self.pool(x)

class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return "{num_features}, eps={eps}".format(**self.__dict__)


def freeze_batch_norm2d(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.BatchNorm2d):
        module = FrozenBatchNorm2d(module.num_features)
    else:
        for name, child in module.named_children():
            _child = freeze_batch_norm2d(child)
            if _child is not child:
                setattr(module, name, _child)
    return module

class ConvBNAct(nn.Module):
    """A combination of Conv, BN and activation layer.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        use_act=True,
    ):
        super().__init__()

        self.use_act = use_act

        # NOTE that the padding implementation is different across codebases
        # paddle padding is different from pytorch padding so pytorchOCR defines PaddingSameAsPaddleMaxPool2d
        # D-FINE just uses F.pad(x, [0,1,0,1]), which is equivalent to it in this particular (kernel_size, stride) setting
        # we follow D-FINE and delete the condition expression here since the `padding` argument of ConvBNAct is never used in D-FINE

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding= (kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if self.use_act:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class LightConvBNAct(nn.Module):
    """A combination of point-wise Conv layer and depth-wise Conv layers.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            use_act=False,
        )
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
