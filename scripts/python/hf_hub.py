from pathlib import Path
from task import FormulaNetLit, PreTrainedTokenizerFast
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from huggingface_hub import HfApi, create_repo, snapshot_download

def load(path='model'):
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(path)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def save(save_path='model'):
    ckpt_path = Path('outputs/bs=64-lr=1e-05-freeze=False/2025-10-04_22-15-03/0/checkpoints/step=85765-val_loss=2.9496e-01-BLEU=0.8356-edit_distance=0.1291.ckpt')
    task = FormulaNetLit.load_from_checkpoint(ckpt_path, 
                                            hparams_file=ckpt_path.parent.parent.joinpath("tb_logs/hparams.yaml"))
    tokenizer: PreTrainedTokenizerFast = task.tokenizer
    tokenizer.save_pretrained(save_path)
    model = task.model
    model.save_pretrained(save_path)

def push(args):
    repo_id = "alephpi/FormulaNet"
    create_repo(repo_id, exist_ok=True)

    api = HfApi()
    api.upload_folder(
        folder_path="./model",
        repo_id=repo_id,
        repo_type="model"
    )

def pull(args):
    repo_id= "alephpi/FormulaNet"
    if args.with_useful_ckpts:
    # 方法1: 使用 ignore_patterns 忽略 checkpoint 文件夹
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir="./model",
        )
    else:
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir="./model",
            ignore_patterns=["checkpoints"]
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()
    sub_parser = sub_parsers.add_parser("push", help="push model to HF hub")
    sub_parser.set_defaults(func=push)

    sub_parser = sub_parsers.add_parser("pull", help="pull model from HF hub")
    sub_parser.add_argument("--with_useful_ckpts", action="store_true", help="pull with useful checkpoints")
    sub_parser.set_defaults(func=pull)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
