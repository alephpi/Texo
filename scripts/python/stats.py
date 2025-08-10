from collections import OrderedDict, defaultdict
from pathlib import Path

DATA_PATH = Path("./data/dataset/UniMER-1M_merged/train_normalized_.txt")
TOKEN_IN_DESIGN_PATH = Path("./data/tokenizer/support.txt")
TOKEM_IN_USE_PATH = Path("./data/dataset/UniMER-1M_merged/stats.txt")

def compute_stats(lines: list[str]):
    vocab: defaultdict = defaultdict(int)
    for line in lines:
        pretokens = line.split()
        tokens = []
        for pretoken in pretokens:
            if "\\\\" == pretoken:
                tokens.append(pretoken)
            elif "\\" in pretoken:
                subtokens = pretoken.split("\\")
                if len(subtokens) > 1:
                    subtokens = ["\\" + subtoken for subtoken in subtokens[1:]]
                tokens.extend(subtokens)
            else:
                tokens.append(pretoken)
        for token in tokens:
            vocab[token] += 1
    return vocab

def load_token_in_design(path=TOKEN_IN_DESIGN_PATH, ignore_group=["Color", "Class_Assignment", "Spacing", "Style"]):
    token_in_design = set()
    ignore = False
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            if not l.startswith("  "):
                ignore = False
                l = l.strip().strip(":")
                if l in ignore_group:
                    ignore = True
            elif not ignore:
                token = l.strip()
                token_in_design.add(token)
    return token_in_design


def main():
    print(f"compute stats in {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = [i.strip() for i in f]
    
    vocab = compute_stats(lines)
    vocab = OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
    output_file = DATA_PATH.with_name("stats.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for key, count in vocab.items():
            f.write(f"{key} {count}\n")
    print(f"saved to {output_file}")

    token_in_design = load_token_in_design()
    token_in_use = set(vocab.keys())
    token_in_common = token_in_use.intersection(token_in_design)
    residues = token_in_use.difference(token_in_common)

    output_file = DATA_PATH.with_name("residue_tokens.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for token, count in vocab.items():
            if token in residues:
                f.write(f"{token} {count}\n")
    print(f"saved to {output_file}")


if __name__ == "__main__":
    main()
