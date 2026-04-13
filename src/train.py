import logging

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from datamodule import MERDataModule
from texo.utils.config import DictConfig, OmegaConf, hydra
from task import FormulaNetLit


@hydra.main(version_base="1.3.2",config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    torch.set_float32_matmul_precision("medium")

    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)

    model = FormulaNetLit( model_config, training_config)
    logging.log(logging.INFO, f"Model initialized.")

    datamodule = MERDataModule(data_config)
    logging.log(logging.INFO, f"Dataset initialized.")

    # Explicitly instantiate loggers so wandb.init() is called eagerly.
    # Relying on Hydra to recursively instantiate a list of logger _target_ configs
    # is unreliable — Lightning may receive raw OmegaConf objects and silently skip them.
    loggers = [
        WandbLogger(project="texo-replication", save_dir=cfg.output_dir),
        TensorBoardLogger(save_dir=cfg.output_dir, name="", version="", sub_dir="tb_logs"),
    ]

    if cfg.trainer.profiler:
        from lightning.pytorch.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(
            dirpath=cfg.output_dir,
            filename="profile",
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.output_dir+"/tb_logs"),
            schedule=torch.profiler.schedule(skip_first=10, wait=2, warmup=2, active=50, repeat=10),
            )
    else:
        profiler = None
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, profiler=profiler)
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