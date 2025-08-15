from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

TextProcessorConfig = {
    "tokenizer_path": "./data/tokenizer/tokenizer.json",
    "max_length": 1024,
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt",
    "return_attention_mask": True
}

class TextProcessor:
    def __init__(self, tokenizer_path: str, tokenizer_config: dict[str,str]):
        self.tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer_config = tokenizer_config

        # 确保tokenizer有pad_token，如果没有则使用eos_token
        assert self.tokenizer.pad_token is not None, "Tokenizer must have the pad_token"

    def __call__(self, text: str):
        return self.process(text)
    
    def process(self, text: str):
        encoding = self.tokenizer(
            text=text,
            **self.tokenizer_config,
        )

        return encoding