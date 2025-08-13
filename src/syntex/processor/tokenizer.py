from transformers import NougatTokenizerFast
from ftfy import fix_text


class DonutTokenizer:
    def __init__(self, path):
        self.tokenizer: NougatTokenizerFast = NougatTokenizerFast.from_pretrained(
            path)
        self.max_seq_len = 2048
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, texts, max_length=None):
        if not max_length:
            max_length = self.max_seq_len
        text_inputs = self.tokenizer(
            texts,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        return text_inputs

    @staticmethod
    def post_process(text):
        text = fix_text(text)
        return text

    def token2str(self, tokens) -> list:
        generated_text = self.tokenizer.batch_decode(
            tokens, skip_special_tokens=True)
        generated_text = [self.post_process(text) for text in generated_text]
        return generated_text

    def detokenize(self, tokens):
        toks = [self.tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for tok in toks:
            for char in reversed(tok):
                if char is None:
                    char = ''
                char = char.replace('Ġ', ' ').strip()
                if char in ([self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]):
                    del char
        return toks
