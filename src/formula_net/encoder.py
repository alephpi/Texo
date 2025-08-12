
from .hgnet2 import HGNetv2

# adapted from PPHGNetV2 and PPHGNetV2_B4_Formula

encoder_config = {
    "stem_channels": [3, 32, 48],
    "stage_config": {
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [48, 48, 128, 1, False, False, 3, 6],
            "stage2": [128, 96, 512, 1, True, False, 3, 6],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
        }
}

encoder = HGNetv2(**encoder_config)