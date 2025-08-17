import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from syntex.formula_net.dataset import MERDataModule
from syntex.formula_net.model import FormulaNetLit


@hydra.main(version_base="1.3.2",config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    torch.set_float32_matmul_precision("medium")

    datamodule = MERDataModule(data_config=cfg.data)

    model = FormulaNetLit(cfg.model, cfg.training)

    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        log_every_n_steps=50,
        max_steps=cfg.training.max_steps,
        callbacks=[
            ModelCheckpoint(
                filename = "{step}-{train_loss:.8f}",
                save_top_k=10,
                monitor="train_loss",
                every_n_train_steps=1000,
            ),
            LearningRateMonitor("step"),
        ],
        logger=TensorBoardLogger(
            save_dir=cfg.output_dir
            )
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()