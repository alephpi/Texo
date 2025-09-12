import logging
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from syntex.formula_net.dataset import MERDataModule
from syntex.formula_net.model import FormulaNetLit


@hydra.main(version_base="1.3.2",config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    torch.set_float32_matmul_precision("medium")

    model = FormulaNetLit(cfg.model, cfg.training)
    logging.log(logging.INFO, f"Model initialized.")

    datamodule = MERDataModule(data_config=cfg.data)
    logging.log(logging.INFO, f"Dataset initialized.")

    trainer = hydra.utils.instantiate(cfg.trainer)
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