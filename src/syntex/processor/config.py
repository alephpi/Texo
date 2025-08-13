from typing import Dict, List, Optional, Union
from transformers import PretrainedConfig


class UniMERNetEvalImageProcessorConfig(PretrainedConfig):
    model_type = "unimernet_evalprocessor"

    def __init__(self, size: Optional[Dict[str, int]] = None,
                 image_mean: Optional[Union[float, List[float]]] = None,
                 image_std: Optional[Union[float, List[float]]] = None, **kwargs):
        self.size = size
        self.image_mean = image_mean
        self.image_std = image_std
        super().__init__(**kwargs)


class UniMERNetTrainImageProcessorConfig(UniMERNetEvalImageProcessorConfig):
    model_type = "unimernet_trainprocessor"

    def __init__(self, size: Optional[Dict[str, int]] = None,
                 image_mean: Optional[Union[float, List[float]]] = None,
                 image_std: Optional[Union[float, List[float]]] = None, **kwargs):
        super().__init__(size, image_mean, image_std, **kwargs)
