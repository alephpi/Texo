# This file is used to convert paddle formula net to pytorch formula net.
# However to keep this project environment clean, the conversion is done in another environment
# where you need to properly download paddle package, paddleOCR repo and PPFormulaNet model.

import numpy as np
import paddle
import paddle.nn as pnn
import torch
import torch.nn as tnn

CONVERT_MAP: dict[tuple[pnn.Layer, tnn.Module], dict[str, str]] = {
    (pnn.Conv2D, tnn.Conv2d): {
        "weight": "weight"
    },
    (pnn.BatchNorm2D, tnn.BatchNorm2d): {
        "weight": "weight",
        "bias": "bias",
        "_mean": "running_mean",
        "_variance": "running_var",
    }, #NOTE be careful about torch bn.tracking_running_stats and paddle bn._use_global_stats
    (pnn.ReLU, tnn.ReLU): {},
    (pnn.MaxPool2D, tnn.MaxPool2d): {}

}

def convert_tensor(pp_tensor:paddle.Tensor) -> torch.Tensor:
    pt_tensor = torch.from_numpy(pp_tensor.numpy())
    return pt_tensor

def convert_layer(pp_layer: pnn.layer, pt_layer: tnn.Module):
    conversion_pair = (type(pp_layer), type(pt_layer))
    assert conversion_pair in CONVERT_MAP.keys(), f"{conversion_pair=} is invalid"
    param_map = CONVERT_MAP[(type(pp_layer), type(pt_layer))]
    pp_state_dict = pp_layer.state_dict()
    pt_state_dict = {}
    for pp_key, pt_key in param_map.items():
        pt_state_dict[pt_key] = convert_tensor(pp_state_dict[pp_key])

    return pt_layer.load_state_dict(pt_state_dict, strict=True, assign=False)

def validate_layer_params(pp_layer: pnn.Layer, pt_layer: tnn.Module):
    conversion_pair = (type(pp_layer), type(pt_layer))
    assert conversion_pair in CONVERT_MAP.keys(), f"{conversion_pair=} is invalid"
    param_map = CONVERT_MAP[(type(pp_layer), type(pt_layer))]
    pp_state_dict = pp_layer.state_dict()
    pt_state_dict = pt_layer.state_dict()
    diff = 0
    for pp_key, pt_key in param_map.items():
        pp_param = pp_state_dict[pp_key].numpy()
        pt_param = pt_state_dict[pt_key].numpy(force=True)
        diff = np.abs(pp_param - pt_param).max()
        assert np.allclose(pp_param, pt_param), f"{pp_key=} vs {pt_key=} params mismatch"
    return diff

def validate_block_params(pp_block: pnn.Layer, pt_block: tnn.Module):
    diffs = []
    pp_block_children = list(pp_block.named_children())
    pt_block_children = list(pt_block.named_children())
    assert (l1:=len(pp_block_children)) == (l2:=len(pt_block_children)), f"{pp_block=} vs {pt_block=} layer numbers mismatch: {l1} vs {l2}"
    if pp_block_children == [] and pt_block_children == []:
        diff = validate_layer_params(pp_block, pt_block)
        diffs.append(diff)

    for (pp_layer_name, pp_layer_child), (pt_layer_name, pt_layer_child) in zip(pp_block_children, pt_block_children):
        assert pp_layer_name == pt_layer_name, f"layer name mismatch: {pp_layer_name=} vs {pt_layer_name=}"
        sub_diffs = validate_block_params(pp_layer_child, pt_layer_child)
        diffs.extend(sub_diffs)

    return diffs

def validate_forward(pp_layer: pnn.Layer, pt_layer: tnn.Module, random_array: np.ndarray, addition=None):
    print("validate forward") 
    print(f"input shape={random_array.shape}")

    pp_in = paddle.to_tensor(random_array, dtype=np.float32).cpu()
    pt_in = torch.from_numpy(random_array).cpu()
    pp_out: paddle.Tensor = pp_layer(pp_in)
    pt_out: torch.Tensor = pt_layer(pt_in)

    if addition:
        pt_out = addition(pt_out)

    pp_out_shape, pt_out_shape = tuple(pp_out.shape), tuple(pt_out.shape)
    pp_out_numpy, pt_out_numpy = pp_out.numpy(), pt_out.numpy(force=True)

    assert pp_out_shape == pt_out_shape, f"Output shape mismatch: {pp_out_shape} vs {pt_out_shape}"
    print(f"output shape={pp_out_shape}")
    print(f"max absolute difference in output tensor: {np.abs(pp_out_numpy - pt_out_numpy).max()}")
    return pp_out_numpy - pt_out_numpy

def convert_block(pp_block: pnn.Layer, pt_block: tnn.Module, level, verbose=True):
    pp_block_children = list(pp_block.named_children())
    pt_block_children = list(pt_block.named_children())
    assert (l1:=len(pp_block_children)) == (l2:=len(pt_block_children)), f"{pp_block=} vs {pt_block=} layer numbers mismatch: {l1} vs {l2}"
    if pp_block_children == [] and pt_block_children == []:
        res = convert_layer(pp_block, pt_block)
        if verbose:
            print('    '*max(level-1,0)+str(res))
        return

    for (pp_layer_name, pp_layer_child), (pt_layer_name, pt_layer_child) in zip(pp_block_children, pt_block_children):
        assert pp_layer_name == pt_layer_name, f"layer name mismatch: {pp_layer_name=} vs {pt_layer_name=}"
        if verbose:
            print('    '*level+f"converting {pp_layer_name}")
        convert_block(pp_layer_child, pt_layer_child, level+1, verbose=verbose)
        if verbose:
            print('    '*level+f"{pp_layer_name} converted")

    return

def main():
    import sys
    sys.path.append("/home/mao/workspace/paddleOCR/PaddleOCR")

    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from ppocr.utils.save_load import load_model
    from tools.program import load_config, merge_config
    model_config = "./configs/rec/PP-FormuaNet/PP-FormulaNet-S.yaml"
    ckpt_path = "./models/rec_ppformulanet_s_train/best_accuracy.pdparams"

    # model_config = "./configs/rec/PP-FormuaNet/PP-FormulaNet_plus-S.yaml"
    # ckpt_path = "./models/rec_ppformulanet_plus_s_train/best_accuracy.pdparams"

    # model_config = "./configs/rec/PP-FormuaNet/PP-FormulaNet_plus-M.yaml"
    # ckpt_path = "./models/rec_ppformulanet_plus_m_train/best_accuracy.pdparams"

    config = load_config(model_config)
    config = merge_config(
        config,
        {"Global.pretrained_model": ckpt_path},
    )

    model = build_model(config["Architecture"])
    best_model_dict = load_model(
        config, model, model_type=config["Architecture"]["model_type"]
    )
    # delete useless layers in paddle model
    del model.backbone.pphgnet_b4.avg_pool
    del model.backbone.pphgnet_b4.last_conv
    del model.backbone.pphgnet_b4.act
    del model.backbone.pphgnet_b4.dropout
    del model.backbone.pphgnet_b4.flatten
    del model.backbone.pphgnet_b4.fc

    # paddle model that needs to be converted from
    paddle_encoder = model.backbone.pphgnet_b4

    # torch model that needs to be converted to
    from ptocr.formula_net.encoder import HGNetv2, encoder_config
    torch_encoder = HGNetv2(**encoder_config)

    paddle_encoder.eval()
    torch_encoder.eval()

    paddle_stats = paddle.summary(paddle_encoder, (1, 3, 384, 384))
    paddle_total_params = int(paddle_stats["total_params"])
    paddle_trainable_params = int(paddle_stats["trainable_params"])

    import torchinfo
    torch_stats = torchinfo.summary(torch_encoder, (1, 3, 384, 384), device='cpu')
    torch_total_params = torch_stats.total_params
    torch_trainable_params = torch_stats.trainable_params

    print(f"{paddle_total_params=}, {paddle_trainable_params=}")
    print(f"{torch_total_params=}, {torch_trainable_params=}")
    assert paddle_trainable_params == torch_trainable_params
    print("Paddle counts BN running stats as its non_trainable params while torch doesn't count them as params(but buffers)")

    convert_block(paddle_encoder, torch_encoder, 0, verbose=False)
    diffs = validate_block_params(paddle_encoder, torch_encoder)
    print(f"max absolute difference in params: {max(diffs)}")

    input_shape = (16, 3, 384, 384)
    x = np.random.rand(*input_shape).astype(np.float32)
    validate_forward(paddle_encoder, torch_encoder, x)

    torch.save(torch_encoder.state_dict(), "formulanet_encoder_hgnetv2.pt")

if __name__ == '__main__':
    main()
