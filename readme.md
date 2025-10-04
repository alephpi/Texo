# texo
A light-weight LaTeXOCR model that runs in browser.

## Training

```sh
# train from scratch
python src/train.py
```
```sh
# resume from a checkpoint
python src/train.py training.resume_from_ckpt="<ckpt_path>"
```

```sh
# debug
python src/train.py --config-dir="./config" --config-name="train_debug.yaml"
```
### Config
We use `hydra` to manage configurations under [config](./config/) directory.

### View training log

```sh
tensorboard --logdir outputs
```