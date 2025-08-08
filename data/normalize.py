from pathlib import Path

from tqdm import tqdm

parent = Path(__file__).resolve().parent

equiv_env_path = parent / "tokenizer/katex/equiv_envs.txt"
equiv_symbol_path = parent / "tokenizer/katex/equiv_symbols.txt"
ad_hoc_path = parent / "tokenizer/katex/ad_hoc.txt"
equiv_expression_path = parent / "tokenizer/katex/equiv_expressions.txt"

def read_dict(path) -> dict[str, str]:
    print(path)
    d = {}
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            token, ortho = line.split("\t")
            if ortho == "None":
                d[token] = ''
            else:
                d[token] = ortho
    return d

AD_HOCS = read_dict(ad_hoc_path)
EQUIV_SYMBOLS = read_dict(equiv_symbol_path)
EQUIV_ENVS = read_dict(equiv_env_path)
EQUIV_EXPRESSIONS = read_dict(equiv_expression_path)

DATA_PATH = parent / "dataset/UniMER-1M_merged/train.txt"

def normalize(data_path: Path=DATA_PATH):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [i.strip() for i in f]
    
    lines = [normalize_env(line) for line in tqdm(lines, total=len(lines))]
    lines = [normalize_expression(line) for line in tqdm(lines, total=len(lines))]

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
        tokens = normalize_symbol(tokens)
        tokens = normalize_left_right(tokens)
        tokens = normalize_ad_hoc(tokens)
        new_lines.append(' '.join(tokens))
    
    with open(data_path.with_name("train_normalized.txt"), 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(f"{line}\n")

def normalize_symbol(tokens, normalizer:dict[str, str]=EQUIV_SYMBOLS):
    return [normalizer.get(token, token) for token in tokens]

def normalize_env(text:str, normalizer:dict[str, str]=EQUIV_ENVS):
    for token, ortho in normalizer.items():
        text = text.replace(f"{{ {token} ", f"{ortho} {{ ").replace(f"{token} {{" ,f"{ortho} {{")
    return text

def normalize_expression(text:str, normalizer:dict[str, str]=EQUIV_EXPRESSIONS):
    for token, ortho in normalizer.items():
        text = text.replace(token, ortho)
    return text

def normalize_left_right(tokens: list[str]):
    new_tokens: list[str] = []
    for token in tokens:
        if token.startswith("\\left") and token[-1] in ["(", ")", "[", "]", "|", "<", ">"]:
            left = "\\left"
            residue = token[-1]
            new_tokens.extend((left, residue))
        elif token.startswith("\\right") and token[-1] in ["(", ")", "[", "]", "|", "<", ">"]:
            right = "\\right"
            residue = token[-1]
            new_tokens.extend((right, residue))
        else:
            new_tokens.append(token)

    return new_tokens

def normalize_ad_hoc(tokens: list[str], normalizer:dict[str, str]=AD_HOCS):
    return [normalizer.get(token, token) for token in tokens]

if __name__ == "__main__":
    normalize()