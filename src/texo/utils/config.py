import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModel

from ..model.hgnet2 import HGNetv2, HGNetv2Config

# setup global configs
AutoConfig.register("my_hgnetv2", HGNetv2Config)
AutoModel.register(HGNetv2Config, HGNetv2)
OmegaConf.register_new_resolver("eval", eval)
