import re
from pathlib import Path

from tqdm import tqdm

parent = Path(__file__).resolve().parent

equiv_env_path = parent / "tokenizer/katex/equiv_envs.txt"
equiv_symbol_path = parent / "tokenizer/katex/equiv_symbols.txt"
ad_hoc_path = parent / "tokenizer/katex/ad_hoc.txt"
equiv_expression_path = parent / "tokenizer/katex/equiv_expressions.txt"
equiv_format_path = parent / "tokenizer/katex/equiv_formats.txt"

def read_dict(path) -> dict[str, str]:
    print(path)
    d = {}
    with open(path, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
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
EQUIV_FORMATS = read_dict(equiv_format_path)

DATA_PATH = parent / "dataset/UniMER-1M_merged/train.txt"

def normalize(data_path: Path=DATA_PATH):
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [i.strip() for i in f]
    
    new_lines = []
    for line in tqdm(lines, total=len(lines)):
        line = normalize_format(line)
        line = normalize_spacing(line)
        line = normalize_env(line)
        line = normalize_expression(line)
        new_lines.append(line)
    lines = new_lines

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
        tokens = normalize_scope(tokens)
        tokens = normalize_symbol(tokens)
        tokens = normalize_left_right(tokens)
        tokens = normalize_ad_hoc(tokens)
        tokens = [token for token in tokens if token != ""]
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

def normalize_format(text:str, normalizer:dict[str, str]=EQUIV_FORMATS):
    for token, ortho in normalizer.items():
        text = text.replace(token, ortho)
    return text

def normalize_spacing(text:str):
    text = text.replace(r" \kern - \nulldelimiterspace ", " ")
    return text

def normalize_left_right(tokens: list[str]):
    new_tokens: list[str] = []
    delimiters =  ["(", ")", "[", "]", "|", "<", ">", "/"]
    for token in tokens:
        if token.startswith("\\left") and token[-1] in delimiters:
            left = "\\left"
            residue = token[-1]
            new_tokens.extend((left, residue))
        elif token.startswith("\\right") and token[-1] in delimiters:
            right = "\\right"
            residue = token[-1]
            new_tokens.extend((right, residue))
        else:
            new_tokens.append(token)

    return new_tokens

def normalize_ad_hoc(tokens: list[str], normalizer:dict[str, str]=AD_HOCS):
    return [normalizer.get(token, token) for token in tokens]

# def normalize_scope(tokens: list[str]):
#     """
#     Remove { \command  xxx }
#     """
#     commands = ["\\phantom","\\hphantom","\\vphantom"]
#     is_open = False
#     scopes: list[int] = []
#     for i in range(len(tokens)):
#         if tokens[i] in commands:
#             left_brace_idx = i-1
#             right_brace_idx = i+1
#             for j in range(i-1, -1, -1):
#                 if tokens[j] == "}":
#                     is_open = True
#                 elif tokens[j] == "{":
#                     if is_open:
#                         is_open = False
#                     else:
#                         left_brace_idx = j
#                         break
#             for k in range(i+1, len(tokens), 1):
#                 if tokens[k] == "{":
#                     is_open = True
#                 elif tokens[k] == "}":
#                     if is_open:
#                         is_open = False
#                     else:
#                         right_brace_idx = k
#                         break
#             scopes.extend(list(range(left_brace_idx+1, right_brace_idx)))
#     for i in sorted(scopes, reverse=True):
#         tokens.pop(i)
#     return tokens

def normalize_scope(tokens: list[str]):
    """
    Remove \\command { xxx }
    """
    commands = ["\\phantom","\\hphantom","\\vphantom"]
    parity = 0 # handle nested braces
    scopes: list[int] = []
    for i in range(len(tokens)):
        if tokens[i] in commands:
            left_brace_idx = i
            for j in range(i+1, len(tokens)):
                if tokens[j] == "{":
                    left_brace_idx = j
                    break
            if left_brace_idx != i:
                right_brace_idx = left_brace_idx
                for k in range(left_brace_idx+1, len(tokens)):
                    if tokens[k] == "{":
                        parity += 1
                    elif tokens[k] == "}":
                        if parity != 0:
                            parity -= 1
                        else:
                            right_brace_idx = k
                            break
                scopes.extend(list(range(i, right_brace_idx + 1)))
            else:
                # 若左括号没有找到，说明是 { \command  xxx }
                for k in range(i+1, len(tokens)):
                    if tokens[k] == "}":
                        right_brace_idx = k
                        break
                scopes.extend(list(range(i, right_brace_idx)))
    for i in sorted(scopes, reverse=True):
        tokens.pop(i)
    return tokens

if __name__ == "__main__":
    normalize()