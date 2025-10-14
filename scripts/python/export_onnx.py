from pathlib import Path
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import ViTOnnxConfig, VisionEncoderDecoderOnnxConfig
from optimum.exporters.onnx.base import ConfigBehavior
from texo.model.formulanet import FormulaNet

register_tasks_manager_onnx = TasksManager.create_register("onnx")
@register_tasks_manager_onnx("my_hgnetv2", *["feature-extraction"])
class HGNetv2OnnxConfig(ViTOnnxConfig):
    @property
    def inputs(self):
        return {"pixel_values": {0: "batch_size"}} # only dynamical axis is needed to list here

def export_onnx():
    path='./model'
    out = Path("./model/trio_onnx")
    out.mkdir(exist_ok=True)
    main_export(
        path,
        task="image-to-text-with-past", # to get trio onnx model, use "-with-past", otherwise use "image-to-text"
        output=out,
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
