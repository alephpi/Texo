from pathlib import Path
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import ViTOnnxConfig, VisionEncoderDecoderOnnxConfig
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from texo.model.formulanet import FormulaNet

register_tasks_manager_onnx = TasksManager.create_register("onnx")
@register_tasks_manager_onnx("my_hgnetv2", *["feature-extraction"])
class HGNetv2OnnxConfig(ViTOnnxConfig):
    @property
    def inputs(self):
        return {"pixel_values": {0: "batch"}} # only dynamical axis is needed to list here
    @property
    def outputs(self):
        return {"last_hidden_state": {0: "batch"}}

def export_onnx():
    path='./model'
    # model = VisionEncoderDecoderModel.from_pretrained(path)
    # onnx_config_constructor = TasksManager.get_exporter_config_constructor(
    #     exporter="onnx",
    #     model=model,
    #     task="image-to-text",
    #     library_name="transformers",
    # )
    # onnx_config = onnx_config_constructor(model.config)
    # out = Path("./model/onnx")
    # out.mkdir(exist_ok=True)

    # inputs, outputs = export(model, 
    #                          onnx_config, 
    #                          out/"model.onnx", 
    #                          onnx_config.DEFAULT_ONNX_OPSET,
    #                          input_shapes={"pixel_values": [1, 3, 384, 384]},
    #                          )
    # print(inputs)
    # print(outputs)

    out = Path("./model/trio_onnx")
    out.mkdir(exist_ok=True)

    # NOTE: this can only export to encoder_model.onnx and decoder_model.onnx,
    # so I monkey-patched the source code to inject use_past=True to export the decoder_model_with_past.onnx
    main_export(
        path,
        task="image-to-text",
        output=out,
        # custom_onnx_configs=custom_onnx_config
    )

if __name__ == '__main__':
    import debugpy
    try:
        debugpy.listen(('localhost', 9501))
        print('Waiting for debugger attach')
        debugpy.wait_for_client()
    except Exception as e:
        pass
    export_onnx()
