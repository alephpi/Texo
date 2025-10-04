import hydra
import logging

import lightning as L
import torch

from datamodule import MERDataModule
from syntex.utils.config import DictConfig, OmegaConf, hydra
from task import FormulaNetLit

    
@hydra.main(version_base="1.3.2",config_path="../config", config_name="train_distill.yaml")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    torch.set_float32_matmul_precision("medium")

    # Set struct to False to allow for dynamic pop
    OmegaConf.set_struct(cfg.model, False)
    OmegaConf.set_struct(cfg.training, False)

    model = FormulaNetLit.load_from_checkpoint("/home/ids/smao-22/phd/SynTeX/outputs/bs=64-lr=1e-06-decoder_layers=2-hidden_size=384/2025-09-30_20-19-36/0/checkpoints/step=63219-val_loss=2.7654e-01-BLEU=0.8947-edit_distance=0.1026.ckpt", 
                                            hparams_file="/home/ids/smao-22/phd/SynTeX/outputs/bs=64-lr=1e-06-decoder_layers=2-hidden_size=384/2025-09-30_20-19-36/0/tb_logs/hparams.yaml")

    model.eval()
    logging.log(logging.INFO, f"Model checkpoint loaded.")

    datamodule = MERDataModule(data_config=cfg.data)
    logging.log(logging.INFO, f"Dataset initialized.")
    trainer = hydra.utils.instantiate(cfg.trainer)
    logging.log(logging.INFO, f"Trainer initialized.")

    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    # import debugpy
    # try:
    #     debugpy.listen(('localhost', 9501))
    #     print('Waiting for debugger attach')
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass
    main()