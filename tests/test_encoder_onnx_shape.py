"""Regression coverage for #12; no pretrained weights or downloads required."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from optimum.exporters.onnx import export
from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.model_configs import VisionEncoderDecoderOnnxConfig
from transformers import MBartConfig, VisionEncoderDecoderConfig

# Register Texo's model and ONNX config, as in the export entry point.
from scripts.python import export_onnx  # noqa: F401
from texo.model.hgnet2 import HGNetv2, HGNetv2Config


@pytest.fixture(scope="module")
def exported_encoder(tmp_path_factory):
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            encoder = HGNetv2(HGNetv2Config(hidden_size=2048)).eval()
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, MBartConfig()
        )
        onnx_config = VisionEncoderDecoderOnnxConfig(
            config, task="image-to-text", behavior=ConfigBehavior.ENCODER
        )
        path = tmp_path_factory.mktemp("encoder_onnx") / "encoder_model.onnx"
        # Use the same composite encoder config and dynamic-axis fix as main_export.
        # Its default 64x64 dummy input must not freeze the sequence axis to 4.
        export(encoder, onnx_config, path)
        model = onnx.load(path)
        onnx.checker.check_model(model)
        output = next(o for o in model.graph.output if o.name == "last_hidden_state")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        yield encoder, output.type.tensor_type.shape.dim, session
    finally:
        torch.set_num_threads(num_threads)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_encoder_output_shape(exported_encoder, batch_size, capfd):
    encoder, declared_dims, session = exported_encoder
    pixels = np.zeros((batch_size, 3, 384, 384), dtype=np.float32)
    with torch.no_grad():
        reference = encoder(torch.from_numpy(pixels)).last_hidden_state
    actual = session.run(["last_hidden_state"], {"pixel_values": pixels})[0]
    assert tuple(reference.shape) == actual.shape == (batch_size, 144, 2048)

    # Inspect the file's original metadata, not ORT's inferred/optimized shape.
    assert len(declared_dims) == actual.ndim
    for dim, size in zip(declared_dims, actual.shape):
        if dim.HasField("dim_value"):
            assert dim.dim_value == size
        else:
            assert dim.dim_param
    assert declared_dims[0].dim_param  # batch remains dynamic
    assert declared_dims[1].dim_param  # follows the dynamic spatial input axes
    assert declared_dims[2].dim_value == 2048
    assert "does not match actual shape" not in capfd.readouterr().err
