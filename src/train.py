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

    precision = "bf16" if torch.cuda.is_bf16_supported() else "16"

    trainer = L.Trainer(
        default_root_dir=cfg.output_dir,
        accelerator="auto",
        devices="auto",
        precision=precision, # type: ignore
        # limit_train_batches=2, # for debugging
        # limit_val_batches=2, # for debugging
        log_every_n_steps=50,
        max_epochs=cfg.training.max_epochs,
        val_check_interval=1000,
        callbacks=[
            ModelCheckpoint(
                filename = "{step}-{val_loss:.4e}-{BLEU:.4f}-{edit_distance:.4f}",
                save_top_k=10,
                save_last=True,
                monitor="val_loss",
                mode="min",
                every_n_epochs=1,
            ),
            LearningRateMonitor("step"),
        ],
        logger=TensorBoardLogger(
            save_dir=cfg.output_dir
            )
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.training.resume_from_ckpt)

if __name__ == "__main__":
    main()