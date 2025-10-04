# This file is used to convert paddle formula net to pytorch formula net.
# However to keep this project environment clean, the conversion is done in another environment
# where you need to properly download paddle package, paddleOCR repo and PPFormulaNet model.

from inspect import getmro
from itertools import product
from pathlib import Path

import numpy as np
import paddle
import paddle.nn as pnn
import torch
import torch.nn as tnn
import transformers
import transformers.activations

# refer to https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html#torch
# for api differences between paddle and torch
CONVERT_MAP: dict[tuple[type[pnn.Layer], type[tnn.Module]], dict[str, str]] = {
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
    (pnn.MaxPool2D, tnn.MaxPool2d): {},
    (pnn.Embedding, tnn.Embedding): {"weight": "weight"},
    (pnn.Linear, tnn.Linear): {"weight":"weight"},
    (pnn.LayerNorm, tnn.LayerNorm): {"weight":"weight", "bias":"bias"},
    
}

def isconvertable(conversion_pair: tuple[type[pnn.Layer], type[tnn.Module]]):
    mro1 = getmro(conversion_pair[0])
    mro2 = getmro(conversion_pair[1])
    for pair in product(mro1, mro2):
        if pair in CONVERT_MAP.keys():
            return CONVERT_MAP[pair]
    return False

def convert_tensor(pp_tensor:paddle.Tensor) -> torch.Tensor:
    pt_tensor = torch.from_numpy(pp_tensor.numpy())
    return pt_tensor

def convert_layer(pp_layer: pnn.Layer, pt_layer: tnn.Module):
    conversion_pair = (type(pp_layer), type(pt_layer))
    param_map = isconvertable(conversion_pair)
    if param_map is False:
        raise TypeError(f"{conversion_pair=} is invalid")
    pp_state_dict = pp_layer.state_dict()
    pt_state_dict = {}
    for pp_key, pt_key in param_map.items():
        if conversion_pair == (pnn.Linear, tnn.Linear):
            # NOTE paddle linear weight is (in_features, out_features) while torch linear weight is (out_features, in_features)
            pt_state_dict["weight"] = convert_tensor(pp_state_dict["weight"].T)
            if pp_state_dict.get("bias") is not None:
                pt_state_dict["bias"] = convert_tensor(pp_state_dict["bias"])
        else:
            pt_state_dict[pt_key] = convert_tensor(pp_state_dict[pp_key])

    res = pt_layer.load_state_dict(pt_state_dict, strict=True, assign=False)
    return res

def convert_block(pp_block: pnn.Layer, pt_block: tnn.Module, level, strict=True, verbose=True):
    pp_block_children = list(pp_block.named_children())
    pt_block_children = [(name, child) for name, child in pt_block.named_children() if not isinstance(child, transformers.activations.GELUActivation)] # ad-hoc: filter out GeLU
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
        convert_block(pp_layer_child, pt_layer_child, level+1, strict=strict, verbose=verbose)
        if verbose:
            print('    '*level+f"{pp_layer_name} converted")
    return

def validate_layer_params(pp_layer: pnn.Layer, pt_layer: tnn.Module):
    conversion_pair = (type(pp_layer), type(pt_layer))
    param_map = isconvertable(conversion_pair)
    if param_map is False:
        raise TypeError(f"{conversion_pair=} is invalid")
    pp_state_dict = pp_layer.state_dict()
    pt_state_dict = pt_layer.state_dict()
    diff = 0
    for pp_key, pt_key in param_map.items():
        pp_param = pp_state_dict[pp_key].numpy()
        pt_param = pt_state_dict[pt_key].numpy(force=True)
        if isinstance(pp_layer,pnn.Linear) and isinstance(pt_layer,tnn.Linear):
            pp_param = pp_param.T
        diff = np.abs(pp_param - pt_param).max()
        assert np.allclose(pp_param, pt_param), f"{pp_key=} vs {pt_key=} params mismatch in {pp_layer=} and {pt_layer=}, {diff=}"
    return diff

def validate_block_params(pp_block: pnn.Layer, pt_block: tnn.Module):
    diffs = []
    pp_block_children = list(pp_block.named_children())
    pt_block_children = [(name, child) for name, child in pt_block.named_children() if not isinstance(child, transformers.activations.GELUActivation)] # ad-hoc: filter out GeLU
    assert (l1:=len(pp_block_children)) == (l2:=len(pt_block_children)), f"{pp_block=} vs {pt_block=} layer numbers mismatch: {l1} vs {l2}"
    if pp_block_children == [] and pt_block_children == []:
        diff = validate_layer_params(pp_block, pt_block)
        diffs.append(diff)

    for (pp_layer_name, pp_layer_child), (pt_layer_name, pt_layer_child) in zip(pp_block_children, pt_block_children):
        assert pp_layer_name == pt_layer_name, f"layer name mismatch: {pp_layer_name=} vs {pt_layer_name=}"
        sub_diffs = validate_block_params(pp_layer_child, pt_layer_child)
        diffs.extend(sub_diffs)

    return diffs

def validate_encoder_forward(pp_layer: pnn.Layer, pt_layer: tnn.Module, random_array: np.ndarray, addition=None):

    pp_in = paddle.to_tensor(random_array, dtype=np.float32).cpu()
    pt_in = torch.from_numpy(random_array).cpu()
    pp_out: paddle.Tensor = pp_layer(pp_in).flatten(2).transpose((0, 2, 1))
    pt_out: torch.Tensor = pt_layer(pt_in).last_hidden_state

    if addition:
        pt_out = addition(pt_out)

    pp_out_shape, pt_out_shape = tuple(pp_out.shape), tuple(pt_out.shape)
    pp_out_numpy, pt_out_numpy = pp_out.numpy(), pt_out.numpy(force=True)

    assert pp_out_shape == pt_out_shape, f"Output shape mismatch: {pp_out_shape} vs {pt_out_shape}"
    print(f"output shape={pp_out_shape}")
    print(f"max absolute difference in output tensor: {np.abs(pp_out_numpy - pt_out_numpy).max()}")
    return pp_out_numpy - pt_out_numpy

def validate_decoder_forward(pp_layer: pnn.Layer, pt_layer: tnn.Module, random_arrays: list[np.ndarray], addition=None, debug=False):

    input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask = random_arrays
    pp_input_ids = paddle.to_tensor(input_ids).cpu()
    pt_input_ids = torch.from_numpy(input_ids).cpu()
    pp_attention_mask = paddle.to_tensor(attention_mask).cpu()
    pt_attention_mask = torch.from_numpy(attention_mask).cpu()
    pp_encoder_hidden_states = paddle.to_tensor(encoder_hidden_states).cpu()
    pt_encoder_hidden_states = torch.from_numpy(encoder_hidden_states).cpu()
    pp_encoder_attention_mask = paddle.to_tensor(np.ones_like(encoder_attention_mask)).cpu() if encoder_attention_mask else None
    pt_encoder_attention_mask = torch.from_numpy(encoder_attention_mask).cpu() if encoder_attention_mask else None

    if debug:
        pp_act = pp_register_activation_hook(pp_layer)
        pt_act = pt_register_activation_hook(pt_layer)

        pp_out: paddle.Tensor = pp_layer.forward(pp_input_ids, pp_attention_mask, pp_encoder_hidden_states, pp_encoder_attention_mask, return_dict=True).logits
        pt_out: torch.Tensor = pt_layer.forward(pt_input_ids, pt_attention_mask, pt_encoder_hidden_states, pt_encoder_attention_mask,return_dict=True).logits

        res = {}
        for key, pp_value in pp_act.items():
            pt_value = pt_act[key]
            if isinstance(pp_value, np.ndarray) and isinstance(pt_value, np.ndarray):
                res[key] = np.abs(pp_value - pt_value).max()
            elif isinstance(pp_value, list) and isinstance(pt_value, list):
                res[key] = [np.abs(p1 - p2).max() for p1, p2 in zip(pp_value, pt_value)]
            else:
                res[key] = "activation dismatch"

        return res, pp_act, pt_act, pp_out, pt_out

    pp_out: paddle.Tensor = pp_layer.forward(pp_input_ids, pp_attention_mask, pp_encoder_hidden_states, pp_encoder_attention_mask, return_dict=True).logits
    pt_out: torch.Tensor = pt_layer.forward(pt_input_ids, pt_attention_mask, pt_encoder_hidden_states, pt_encoder_attention_mask,return_dict=True).logits

    if addition:
        pt_out = addition(pt_out)

    pp_out_shape, pt_out_shape = tuple(pp_out.shape), tuple(pt_out.shape)
    pp_out_numpy, pt_out_numpy = pp_out.numpy(), pt_out.numpy(force=True)

    assert pp_out_shape == pt_out_shape, f"Output shape mismatch: {pp_out_shape} vs {pt_out_shape}"
    print(f"output shape={pp_out_shape}")
    print(f"max absolute difference in output tensor: {np.abs(pp_out_numpy - pt_out_numpy).max()}")
    print(f"difference in label prediction: {np.count_nonzero(pp_out_numpy.argmax(axis=-1) - pt_out_numpy.argmax(axis=-1))}")
    return pp_out_numpy , pt_out_numpy

def validate_whole_model(pp_model, pt_model, random_arrays, addition=None, debug=False):
    pixel_values, input_ids, attention_mask = random_arrays
    pp_pixel_values = paddle.to_tensor(pixel_values).cpu()
    pt_pixel_values = torch.from_numpy(pixel_values).cpu()
    pp_input_ids = paddle.to_tensor(input_ids).cpu()
    pt_input_ids = torch.from_numpy(input_ids).cpu()
    pp_attention_mask = paddle.to_tensor(attention_mask).cpu()
    pt_attention_mask = torch.from_numpy(attention_mask).cpu()
    pp_out: paddle.Tensor = pp_model.forward(pp_pixel_values, pp_input_ids, pp_attention_mask)[0]
    pt_out: torch.Tensor = pt_model.forward(pt_pixel_values, pt_input_ids, pt_attention_mask, return_dict=True).logits


    pp_out_shape, pt_out_shape = tuple(pp_out.shape), tuple(pt_out.shape)
    pp_out_numpy, pt_out_numpy = pp_out.numpy(), pt_out.numpy(force=True)

    assert pp_out_shape == pt_out_shape, f"Output shape mismatch: {pp_out_shape} vs {pt_out_shape}"
    print(f"output shape={pp_out_shape}")
    print(f"max absolute difference in output tensor: {np.abs(pp_out_numpy - pt_out_numpy).max()}")
    print(f"difference in label prediction: {np.count_nonzero(pp_out_numpy.argmax(axis=-1) - pt_out_numpy.argmax(axis=-1))}")
    return pp_out_numpy , pt_out_numpy

def pp_register_activation_hook(pp_model: pnn.Layer):
    act = {}
    def get_hook(layer_name, hook_type:str):
        def pre_hook(layer, args):
            if isinstance(args, paddle.Tensor):
                act[layer_name+"_pre"] = args.detach().numpy()
            elif isinstance(args, tuple):
                act[layer_name+"_pre"] = [o.detach().numpy() for o in args if isinstance(o, paddle.Tensor)]
            else:
                act[layer_name+"_pre"] = type(args)
 
        def post_hook(layer, args, output):
            if isinstance(output, paddle.Tensor):
                act[layer_name+"_post"] = output.detach().numpy()
            elif isinstance(output, tuple):
                act[layer_name+"_post"] = [o.detach().numpy() for o in output if isinstance(o, paddle.Tensor)]
            else:
                act[layer_name+"_post"] = type(output)

        if hook_type == "pre":
            return pre_hook
        elif hook_type == "post":
            return post_hook
        else:
            raise ValueError(f"invalid hook_type: {hook_type}")
    for name, layer in pp_model.named_sublayers(include_self=True):
        layer.register_forward_pre_hook(get_hook(name, "pre")) #type: ignore
        layer.register_forward_post_hook(get_hook(name, "post")) #type: ignore
    return act

def pt_register_activation_hook(pt_model: tnn.Module):
    act = {}
    def get_hook(layer_name, hook_type:str):
        def pre_hook(module, args):
            if isinstance(args, torch.Tensor):
                act[layer_name+"_pre"] = args.detach().numpy()
            elif isinstance(args, tuple):
                act[layer_name+"_pre"] = [o.detach().numpy() for o in args if isinstance(o, torch.Tensor)]
            else:
                act[layer_name+"_pre"] = type(args)

        def post_hook(module, args, output):
            if isinstance(output, torch.Tensor):
                act[layer_name+"_post"] = output.detach().numpy()
            elif isinstance(output, tuple):
                act[layer_name+"_post"] = [o.detach().numpy() for o in output if isinstance(o, torch.Tensor)]
            else:
                act[layer_name+"_post"] = type(output)

        if hook_type == "pre":
            return pre_hook
        elif hook_type == "post":
            return post_hook
        else:
            raise ValueError(f"invalid hook_type: {hook_type}")
 
    for name, module in pt_model.named_modules():
        module.register_forward_pre_hook(get_hook(name, "pre"))
        module.register_forward_hook(get_hook(name, "post")) #type: ignore
    return act

def dummy_inputs(batch_size=16, seq_len=100, img_size=384, vocab_size=50000, seed=42):
    np.random.seed(seed)
    pixel_values = np.random.rand(batch_size, 3, img_size, img_size).astype(np.float32)
    decoder_input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    decoder_attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
    encoder_hidden_states = np.random.rand(batch_size, 144, 384).astype(np.float32)
    return pixel_values, decoder_input_ids, decoder_attention_mask, encoder_hidden_states

def main():
    # dummy inputs
    pixel_values, decoder_input_ids, decoder_attention_mask, encoder_hidden_states = dummy_inputs()
    pp_pixel_values = paddle.to_tensor(pixel_values).cpu()
    pt_pixel_values = torch.from_numpy(pixel_values).cpu()
    pp_decoder_input_ids = paddle.to_tensor(decoder_input_ids).cpu()
    pt_decoder_input_ids = torch.from_numpy(decoder_input_ids).cpu()
    pp_decoder_attention_mask = paddle.to_tensor(decoder_attention_mask).cpu()
    pt_decoder_attention_mask = torch.from_numpy(decoder_attention_mask).cpu()

    import sys
    sys.path.append("/home/mao/workspace/paddleOCR/PaddleOCR")
    # sys.path.append("/home/mao/workspace/paddleOCR/ptocr")

    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    from ppocr.utils.save_load import load_model as load_pp_model
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

    # disable parallel decoding in FormulaNet-S
    assert config["Architecture"]["Head"]["use_parallel"] == False
    assert config["Architecture"]["Head"]["parallel_step"] == 3 # 保持3的原因是因为我篡改了paddle源码，这样保证可以全部读取ppformulanet-S的1029个embed_position的嵌入

    print("---load paddle model---")
    pp_model = build_model(config["Architecture"])
    best_model_dict = load_pp_model(
        config, pp_model, model_type=config["Architecture"]["model_type"]
    )
    # delete useless layers in encoder model (I guess it is useful for pretraining)
    del pp_model.backbone.pphgnet_b4.avg_pool
    del pp_model.backbone.pphgnet_b4.last_conv
    del pp_model.backbone.pphgnet_b4.act
    del pp_model.backbone.pphgnet_b4.dropout
    del pp_model.backbone.pphgnet_b4.flatten
    del pp_model.backbone.pphgnet_b4.fc

    pp_model.eval()

    # paddle model that needs to be converted from
    pp_encoder = pp_model.backbone.pphgnet_b4
    pp_decoder = pp_model.head.decoder

    print("\n---summarizing paddle model---")
    pp_stats = paddle.summary(pp_model, 
                              input_size=[(1,3,384,384),(1,100),(1,100)],
                              dtypes=["float32", "int32", "int32"])
    pp_total_params = int(pp_stats["total_params"])
    pp_trainable_params = int(pp_stats["trainable_params"])
    print(f"{pp_total_params=}, {pp_trainable_params=}")
    print("Paddle counts BN running stats as its non_trainable params while torch doesn't count them as params(but buffers)")

    from ptocr.texoodel.formulanet import FormulaNet
    from ptocr.texoodel.formulanet import load_model as load_pt_model

    print("\n---load torch model---")
    # torch model that needs to be converted to
    pt_model = load_pt_model()
    pt_model.eval()

    pt_encoder = pt_model.encoder
    pt_decoder = pt_model.decoder
    assert pt_encoder is not None and pt_decoder is not None

    print("\n---summarizing torch model---")
    import torchinfo
    pt_stats = torchinfo.summary(pt_model, input_data={'pixel_values': pt_pixel_values, 'decoder_input_ids': pt_decoder_input_ids, 'decoder_attention_mask': pt_decoder_attention_mask}, device='cpu')
    pt_total_params = pt_stats.total_params
    pt_trainable_params = pt_stats.trainable_params
    print(f"{pt_total_params=}, {pt_trainable_params=}")

    assert pp_trainable_params == pt_trainable_params

    print("\n---converting encoder---")
    convert_block(pp_encoder, pt_encoder, 0, verbose=False)
    diffs = validate_block_params(pp_encoder, pt_encoder)
    print(f"max absolute difference in params: {max(diffs)}")

    print("\n---validating encoder---")
    validate_encoder_forward(pp_encoder, pt_encoder, pixel_values)

    print("\n---converting decoder---")
    convert_block(pp_decoder, pt_decoder, 0, verbose=False)
    diffs = validate_block_params(pp_decoder, pt_decoder)
    print(f"max absolute difference in params: {max(diffs)}")

    print("\n---validating decoder---")
    validate_decoder_forward(pp_decoder, pt_decoder, [decoder_input_ids, decoder_attention_mask, encoder_hidden_states, None]) #type: ignore

    print("\n---convert projection layer---")
    convert_layer(pp_model.head.enc_to_dec_proj, pt_model.enc_to_dec_proj)
    diff = validate_layer_params(pp_model.head.enc_to_dec_proj, pt_model.enc_to_dec_proj)
    print(f"max absolute difference in params: {diff}")

    print("\n---validating whole model---")
    validate_whole_model(
        pp_model,
        pt_model,
        [pixel_values, decoder_input_ids, decoder_attention_mask],
    )

    path = Path(__file__).resolve().parent / "formulanet.pt"
    torch.save(pt_model.state_dict(),path)
    print(f"formulanet saved to {path}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d','--debug', action='store_true', help='debug mode')
    args = parser.parse_args()
    if args.debug:
        import debugpy
        try:
            debugpy.listen(('localhost', 9502))
            print('Waiting for debugger attach')
            debugpy.wait_for_client()
        except Exception as e:
            pass
    main()
