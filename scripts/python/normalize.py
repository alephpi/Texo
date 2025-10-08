# equivalently normalize the latex codes to make it shorter and easier for model learning
from pathlib import Path

from tqdm import tqdm

NORMALIZER_PATH = Path("./data/tokenizer/normalizer")

normalize_env_path = NORMALIZER_PATH / "envs.txt"
normalize_symbol_path = NORMALIZER_PATH / "symbols.txt"
normalize_ad_hoc_path = NORMALIZER_PATH / "ad_hocs.txt"
normalize_expression_path = NORMALIZER_PATH / "expressions.txt"
normalize_macros_path = NORMALIZER_PATH / "macros.txt"

def read_dict(path) -> dict[str, str]:
    print(f"load {path}")
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

AD_HOCS_normalizer = read_dict(normalize_ad_hoc_path)
SYMBOLS_normalizer = read_dict(normalize_symbol_path)
ENVS_normalizer = read_dict(normalize_env_path)
EXPRESSIONS_normalizer = read_dict(normalize_expression_path)
MACROS_normalizer = read_dict(normalize_macros_path)

DATA_PATH = Path("./data/dataset/UniMER-1M_merged/train.txt")

def normalize(data_path: Path=DATA_PATH):
    print(f"normalize {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [i.strip() for i in f]
    
    new_lines: list[str] = []
    for line in tqdm(lines, total=len(lines)):
        line = normalize_macros(line)
        line = normalize_spacing(line)
        line = normalize_env(line)
        line = normalize_expression(line)
        new_lines.append(line)
    lines = new_lines

    new_lines = []
    for line in tqdm(lines, total=len(lines)):
        tokens = []
        # need to separate \left\command and \right\command
        # as there can be a lot of combinations, and it may be benefitial to 
        # make the model learn to pairing \left and \right individually
        for pretoken in line.split():
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
    
    out_file = data_path.with_name("train_normalized.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(f"{line}\n")
    print(f"saved to {out_file}")

def normalize_symbol(tokens, normalizer:dict[str, str]=SYMBOLS_normalizer):
    return [normalizer.get(token, token) for token in tokens]

def normalize_env(text:str, normalizer:dict[str, str]=ENVS_normalizer):
    for token, ortho in normalizer.items():
        text = text.replace(f"{{ {token} ", f"{ortho} {{ ").replace(f"{token} {{" ,f"{ortho} {{")
    return text

def normalize_expression(text:str, normalizer:dict[str, str]=EXPRESSIONS_normalizer):
    for token, ortho in normalizer.items():
        text = text.replace(token, ortho)
    return text

def normalize_macros(text:str, normalizer:dict[str, str]=MACROS_normalizer):
    for token, ortho in normalizer.items():
        text = text.replace(token, ortho)
    return text

def normalize_spacing(text:str):
    text = text.replace(r" \kern - \nulldelimiterspace ", " ")
    return text

def normalize_left_right(tokens: list[str]):
    new_tokens: list[str] = []
    delimiters =  ["(", ")", "[", "]", "|", "<", ">", "/", "."]
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

def normalize_ad_hoc(tokens: list[str], normalizer:dict[str, str]=AD_HOCS_normalizer):
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