from tokenizers import Tokenizer, models
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def build_tokenizer(vocab_path, output_dir):
    """
    创建一个使用空格预分词并基于给定词表的tokenizer
    
    Args:
        vocab_path: 现有词表文件路径
        output_dir: 保存tokenizer的目录
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [l.strip('\n') for l in f.readlines()]
    special_tokens = ['<pad>','<unk>','<s>','</s>']
    vocab = special_tokens + vocab
    vocab = {token: idx for idx, token in enumerate(vocab)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.decoder = None # 禁用解码器，默认用空格连接

    # 对接 tokenizers.Tokenizer 到 transformers.PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

# 使用示例
if __name__ == "__main__":

    build_tokenizer(
        vocab_path="./data/tokenizer/common_tokens.txt",
        output_dir="./data/tokenizer/tokenizer.json"
    )

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./data/tokenizer/tokenizer.json")

    # 测试tokenizer
    pos_text = r"\begin{array} { r l } { \vec { v _ { i } } ^ { 0 } } & { { } = \frac { \Gamma } { 2 \pi } \sum _ { i \neq j } ^ { N _ { v } } \kappa _ { j } \hat { z } \times \frac { \vec { r } _ { i } - \vec { r } _ { j } } { \left | \vec { r } _ { i } - \vec { r } _ { j } \right | ^ { 2 } } + \frac { 1 } { 2 } \left ( \frac { \Gamma } { 2 \pi } \sum _ { j = 1 } ^ { N _ { v } } \kappa _ { j } \hat { z } \times \frac { \vec { r } _ { i } - \vec { 0 } } { \left | \vec { r } _ { i } - \vec { 0 } \right | ^ { 2 } } \right ) } \end{array}"
    neg_text = r"\sqrt [ [object Object] ] { x ^ { 3 } }"
    pos_encoded = tokenizer.encode(pos_text)
    neg_encoded = tokenizer.encode(neg_text)
    
    print("Original:", pos_text)
    print("Decoded:", pos_res := tokenizer.decode(pos_encoded))
    assert pos_text == pos_res

    print("Original:", neg_text)
    print("Decoded:", neg_res := tokenizer.decode(neg_encoded))
    assert '<unk>' in neg_res
