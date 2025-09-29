# distill useless tokens from the Nougat/UniMERNet tokenizer as well as the MBart decoder
import argparse
import json
import os
from re import S
from typing import List, Set

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    MBartForCausalLM,
    PreTrainedTokenizerFast,
)


def filter_unused_tokens(
    texts: List[str],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 1,
    max_length: int = 4096,
) -> tuple[Set[int], Set[int]]:
    total_samples = len(texts)
    total_batches = (total_samples + batch_size - 1) // batch_size

    print(
        f"总共有{total_samples}条文本，分为{total_batches}个批次，batch_size={batch_size}"
    )

    all_used_token_ids = set()

    # 分批处理
    for i in tqdm(range(total_batches), desc="处理批次"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        batch_texts = texts[start_idx:end_idx]

        # 对当前批次进行编码
        batch_encoded = tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            is_split_into_words=False,
        )

        # 收集这个批次中使用的所有token
        batch_tokens = set(np.unique(batch_encoded.input_ids).tolist())
        all_used_token_ids.update(batch_tokens)

    # 计算未使用的tokens（假设vocab_size=50000）
    all_tokens = set(range(50000))
    unused_token_ids = all_tokens - all_used_token_ids

    print(
        f"处理完成！使用了{len(all_used_token_ids)}个不同的token，未使用{len(unused_token_ids)}个token"
    )

    return all_used_token_ids, unused_token_ids

def read_keep_tokens(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.strip()}

def load_backends(base_id: str):
    # Fast tokenizer exposes the Rust backend needed here:
    # https://huggingface.co/docs/transformers/en/main_classes/tokenizer
    hf_tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    backend: Tokenizer = hf_tok.backend_tokenizer
    return hf_tok, backend

def parse_vocab_merges(backend: Tokenizer):
    # tokenizer.json schema contains both vocab and merges
    tj = json.loads(backend.to_str())  # https://huggingface.co/docs/tokenizers/python/latest/api/reference.html
    model = tj["model"]
    vocab: dict[str, int] = model["vocab"]                     # token -> old_id
    merges_raw = model.get("merges", [])                       # ["A B", ...]
    merges = [tuple(m.split(" ")) for m in merges_raw]         # list[(left,right)]
    return vocab, merges, tj

def collect_specials(hf_tok: PreTrainedTokenizerFast) -> list[str]:
    # Stable source of specials:
    # https://huggingface.co/docs/transformers/en/main_classes/tokenizer
    all_special_tokens = list(hf_tok.all_special_tokens)
    print(list(hf_tok.get_added_vocab().keys()))
    return all_special_tokens

def ensure_keep_bytes(vocab_tok2id: dict[str,int], keep: set[str], nbytes: int):
    # For byte-level BPEs (GPT-2/RoBERTa style), first 256 ids are the byte alphabet.
    # Reference background:
    # https://christianjmills.com/posts/transformers-book-notes/chapter-10/index.html
    id2tok = {i:t for t,i in vocab_tok2id.items()}
    for i in range(min(nbytes, len(id2tok))):
        if i in id2tok:
            keep.add(id2tok[i])

def filter_merges_to_subset(merges: list[tuple[str,str]], keep: set[str]):
    # Keep merge (a,b) when (a+b) belongs to keep and join the a,b to keep to provide an accessible merge path to (a+b)
    # update the keep until no more merge paths can be found
    # BPE merges are greedy and ordered; preserve order.
    filtered_raw = []
    new_keep: Set[str] = set()
    while True:
        keep |= new_keep
        for a, b in merges:
            merged = a + b
            if merged in keep:
                if (a,b) not in filtered_raw:
                    filtered_raw.append((a,b))
                    new_keep.update((a,b))
        if new_keep - keep == set():
            break

    # reorder the filtered merges to preserve order as the raw will break the order as we add merges in multiple loops
    filtered = []
    for merge in merges:
        if merge in filtered_raw:
            filtered.append(merge)
    return filtered

def distill_vocab(args, unk_args):
    with open(args.corpus, "r", encoding="utf-8") as f:
        texts = [sentence.strip() for sentence in f.readlines()]
    
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    all_used_token_ids, _ = filter_unused_tokens(texts, tokenizer)
    all_used_token_ids.update(tokenizer.all_special_ids)
    all_used_tokens = set(tokenizer.convert_ids_to_tokens(all_used_token_ids))
    with open(os.path.join(args.keep_file), "w", encoding="utf-8") as f:
        for token in sorted(all_used_tokens):
            print(token, file=f)

def distill_tokenizer(args, unk_args):
    hf_tok, backend = load_backends(args.base)
    vocab_tok2id, merges, tokjson = parse_vocab_merges(backend)

    # 1) define keep set: user set ∩ existing vocab, plus specials, plus bytes if requested
    keep = read_keep_tokens(args.keep_file)
    keep &= set(vocab_tok2id.keys())
    keep |= set(hf_tok.all_special_tokens)
    if args.keep_bytes > 0:
        ensure_keep_bytes(vocab_tok2id, keep, args.keep_bytes)
    keep -= set(hf_tok.get_added_vocab().keys())
    keep_specials = set(collect_specials(hf_tok))
    keep |= keep_specials

    # 2) filter merges consistently
    filtered_merges = filter_merges_to_subset(merges, keep)

    # 3) reindex by original id order for determinism
    kept_tokens_sorted = [t for t,_ in sorted(((t, vocab_tok2id[t]) for t in keep), key=lambda x: x[1])]
    new_vocab_tok2id = {t:i for i,t in enumerate(kept_tokens_sorted)}  # token -> new_id

    # 4) rebuild a valid BPE with same pipeline
    new_model = BPE(vocab=new_vocab_tok2id, merges=filtered_merges, dropout=None, unk_token=None)
    new_tok = Tokenizer(new_model)
    new_tok.normalizer     = backend.normalizer
    new_tok.pre_tokenizer  = backend.pre_tokenizer
    new_tok.post_processor = backend.post_processor
    # Also mark specials so they are not split:
    # https://huggingface.co/docs/tokenizers/en/api/tokenizer#tokenizers.Tokenizer.add_special_tokens
    if keep_specials:
        new_tok.add_special_tokens(list(keep_specials & set(new_vocab_tok2id.keys())))

    os.makedirs(args.distill, exist_ok=True)
    out_tok = os.path.join(args.distill, "tokenizer.json")
    # new_tok.save(out_tok)

    # save as transformers tokenizer
    new_tok = PreTrainedTokenizerFast(tokenizer_object=new_tok)
    special_tokens_dict = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    # tokenizers 只处理底层 tokenizer 的逻辑，格式化 token 的在训练和推理时的语义，如 bos_token 等，在 transformers 的 api 中定义。
    new_tok.add_special_tokens(special_tokens_dict)
    new_tok.save_pretrained(args.distill)

    # Save old->new id map to drive weight remap
    old2new = {vocab_tok2id[t]: new_vocab_tok2id[t] for t in kept_tokens_sorted}
    with open(os.path.join(args.distill, "old_to_new_id.json"), "w", encoding="utf-8") as f:
        json.dump(old2new, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {out_tok}  | vocab={len(new_vocab_tok2id)} merges={len(filtered_merges)}")
    print("[OK] Saved old_to_new_id.json for embedding remap")

def verify(args, unk_args):
    from transformers import PreTrainedTokenizerFast
    tokenizer_base = PreTrainedTokenizerFast.from_pretrained(args.base)
    tokenizer_distill = PreTrainedTokenizerFast.from_pretrained(args.distill)
    old_to_new_id = json.load(open(os.path.join(args.distill, "old_to_new_id.json"), "r"))
    old_to_new_id = {int(k):v for k,v in old_to_new_id.items()}
    with open(args.corpus, "r", encoding="utf-8") as f:
        texts = [sentence.strip() for sentence in f.readlines()]

    for text in tqdm(texts):
        encoded_ids_base = tokenizer_base(text, max_length=4096, truncation=True)
        encoded_ids_distill = tokenizer_distill(text, max_length=4096, truncation=True)
        assert len(encoded_ids_base["input_ids"]) == len(encoded_ids_distill["input_ids"])
        for encoded_id_base, encoded_id_distill in zip(
            encoded_ids_base["input_ids"], encoded_ids_distill["input_ids"]
        ):
            assert old_to_new_id[encoded_id_base] == encoded_id_distill, (
                f"tokenization is not consistent for {text}\n{encoded_ids_base}\n{encoded_ids_distill}"
            )
    print("[OK] Validated.")
    return True

def distill_weight(args, unk_args):
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf as oc

    from syntex.model.formulanet import FormulaNet
    base_config = oc.load(args.config)
    oc.resolve(base_config)

    model: MBartForCausalLM = FormulaNet(base_config)
    decoder = model.decoder
    assert decoder is not None
    assert base_config.pretrained is not None, "base model config must have a pretrained checkpoint"
    base_ckpt_path = base_config.pretrained
    distill_ckpt_path = base_ckpt_path.replace(".pt", "_distill.pt")
    distill_config = args.config.replace(".yaml", "_distill.yaml")
    with open(args.map, "r", encoding="utf-8") as f:
        old_to_new_id = json.load(f)
    old_to_new_id = {int(k):v for k,v in old_to_new_id.items()}
    keep_old_ids = [old for old, new in sorted(old_to_new_id.items(), key=lambda item: item[1])]
    new_vocab_size = len(old_to_new_id)
    print(new_vocab_size)

    # resize and remap the weights in embed_tokens and lm_head
    old_emb: nn.Embedding = decoder.get_input_embeddings()
    new_emb = nn.Embedding(new_vocab_size, old_emb.embedding_dim)
    new_emb.weight.data = old_emb.weight.data[keep_old_ids]
    decoder.set_input_embeddings(new_emb)

    # 修改输出lm head
    old_head: nn.Linear = decoder.get_output_embeddings()
    new_head = nn.Linear(old_head.in_features, new_vocab_size, bias=False)
    new_head.weight.data = old_head.weight.data[keep_old_ids]
    decoder.set_output_embeddings(new_head)
    # print(model)
    torch.save(model.state_dict(), distill_ckpt_path)

    import torchinfo
    torchinfo.summary(model)

def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()
    sub_parser = sub_parsers.add_parser("vocab", help="Subset a tokenizer's useful vocab based on a corpus")
    sub_parser.add_argument("--corpus", type=str, required=True, help="text corpus, one sentence per line")
    sub_parser.add_argument("--base", type=str, required=True, help="HF repo id or local path for the base tokenizer")
    sub_parser.add_argument("--keep_file", type=str, required=True, help="file of the subset that needs to be kept in the vocab, one per line")
    sub_parser.set_defaults(func=distill_vocab)

    sub_parser = sub_parsers.add_parser("tokenizer", help="Distill a tokenizer's vocab and merges to a smaller subset based on the keep file")
    sub_parser.add_argument("--base", required=True, help="HF repo id or local path of the base tokenizer")
    sub_parser.add_argument("--distill", required=True, help="output directory for the distilled tokenizer")
    sub_parser.add_argument("--keep_file", required=True, help="file with tokens to keep, one per line")
    sub_parser.add_argument("--keep_bytes", type=int, default=279, help="0 to disable; 256 for byte-level BPEs, 279 for Nougat tokenizer since it uses 23 special tokens before the byte tokens")
    sub_parser.set_defaults(func=distill_tokenizer)

    sub_parser = sub_parsers.add_parser("verify", help="Verify the tokenization consistency of the distilled tokenizer with the original one on the corpus")
    sub_parser.add_argument("--base", required=True, help="HF repo id or local path")
    sub_parser.add_argument("--distill", required=True, help="output directory for the distilled tokenizer")
    sub_parser.add_argument("--corpus", type=str, required=True, help="text corpus, one sentence per line")
    sub_parser.set_defaults(func=verify)

    sub_parser = sub_parsers.add_parser("weight", help="Verify the tokenization consistency of the distilled tokenizer with the original one on the corpus")
    sub_parser.add_argument("--config", required=True, help="base model config path")
    sub_parser.add_argument("--map", type=str, required=True, help="json file mapping old token ids to new token ids for weight remapping")
    sub_parser.set_defaults(func=distill_weight)


    args, unk_args = parser.parse_known_args()
    if hasattr(args, "func"):
        args.func(args, unk_args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()