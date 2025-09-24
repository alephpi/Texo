import logging

import lightning as L
import torch

from datamodule import MERDataModule
from syntex.utils.config import DictConfig, OmegaConf, hydra
from task import FormulaNetLit


@hydra.main(version_base="1.3.2",config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    torch.set_float32_matmul_precision("medium")

    # Set struct to False to allow for dynamic pop
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.training, False)

    model = FormulaNetLit(cfg.tokenizer_path, cfg.model, cfg.training)
    logging.log(logging.INFO, f"Model initialized.")

    datamodule = MERDataModule(data_config=cfg.data)
    logging.log(logging.INFO, f"Dataset initialized.")

    if cfg.trainer.profiler:
        from lightning.pytorch.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(
            dirpath=cfg.trainer.logger.save_dir,
            filename="profile",
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.trainer.logger.save_dir+"/tb_logs"),
            schedule=torch.profiler.schedule(skip_first=10, wait=2, warmup=2, active=50, repeat=10),
            )
    else:
        profiler = None
    trainer = hydra.utils.instantiate(cfg.trainer, profiler=profiler)
    logging.log(logging.INFO, f"Trainer initialized.")

    if cfg.training.resume_from_ckpt:
        logging.log(logging.INFO, f"Resuming from checkpoint: {cfg.training.resume_from_ckpt}")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.training.resume_from_ckpt)

if __name__ == "__main__":
    # import debugpy
    # try:
    #     debugpy.listen(('localhost', 9501))
    #     print('Waiting for debugger attach')
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    main()