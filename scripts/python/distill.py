# distill useless tokens from the Nougat/UniMERNet tokenizer as well as the MBart decoder
import argparse
import json
import os
from typing import List, Set

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def filter_unused_tokens(
    texts: List[str],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 16,
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
    unused_tokens = all_tokens - all_used_token_ids

    print(
        f"处理完成！使用了{len(all_used_token_ids)}个不同的token，未使用{len(unused_tokens)}个token"
    )

    return all_used_token_ids, unused_tokens

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
    # print(all_special_tokens)
    # print(hf_tok.added_tokens_decoder)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF repo id or local path")
    ap.add_argument("--keep_file", required=True, help="file with tokens to keep, one per line")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep_bytes", type=int, default=0, help="0 to disable; 256 for byte-level BPEs")
    args = ap.parse_args()

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
    # remove potential repetitive merges
    # seen = set()
    # filtered_merges = []
    # for pair in filtered_merges_raw:
    #     if pair not in seen:
    #         filtered_merges.append(pair)
    #         seen.add(pair)
    #     else:
    #         print(f"重复的 merge: {pair}")

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

    os.makedirs(args.out_dir, exist_ok=True)
    out_tok = os.path.join(args.out_dir, "tokenizer.json")
    new_tok.save(out_tok)

    # Save old->new id map to drive weight remap
    old2new = {vocab_tok2id[t]: new_vocab_tok2id[t] for t in kept_tokens_sorted}
    with open(os.path.join(args.out_dir, "old_to_new_id.json"), "w", encoding="utf-8") as f:
        json.dump(old2new, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {out_tok}  | vocab={len(new_vocab_tok2id)} merges={len(filtered_merges)}")
    print("[OK] Saved old_to_new_id.json for embedding remap")

if __name__ == "__main__":
    main()