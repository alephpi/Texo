from pathlib import Path

from tqdm import tqdm

parent = Path(__file__).resolve().parent

equiv_env_path = parent / "tokenizer/katex/equiv_envs.txt"
equiv_symbol_path = parent / "tokenizer/katex/equiv_symbols.txt"

def read_dict(path) -> dict[str, str]:
    print(path)
    d = {}
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            token, ortho = line.split()
            d[token] = ortho
    return d

EQUIV_SYMBOLS = read_dict(equiv_symbol_path)
EQUIV_ENVS = read_dict(equiv_env_path)

DATA_PATH = parent / "dataset/UniMER-1M_merged/train.txt"

def normalize(data_path: Path=DATA_PATH):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [i.strip() for i in f]
    
    lines = [normalize_env(line) for line in tqdm(lines, total=len(lines))]

    new_lines = []
    for line in tqdm(lines, total=len(lines)):
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
        new_tokens = normalize_symbol(tokens)
        new_lines.append(' '.join(new_tokens))
    
    with open(data_path.with_name("train_normalized.txt"), 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(f"{line}\n")

def normalize_symbol(tokens, normalizer:dict[str, str]=EQUIV_SYMBOLS):
    return [normalizer.get(token, token) for token in tokens]

def normalize_env(text:str, normalizer:dict[str, str]=EQUIV_ENVS):
    for token, ortho in normalizer.items():
        text = text.replace(f"{{ {token} ", f"{ortho} {{ ").replace(f"{token} {{" ,f"{ortho} {{")
    return text

if __name__ == "__main__":
    normalize()