from transformers import PreTrainedTokenizerFast

text_processor_default_config = {
    "tokenizer_path": "data/tokenizer",
    "tokenizer_config":{
        "add_special_tokens": True,
        "max_length": 1024,
        "padding": "longest",
        "truncation": True,
        "return_tensors": "pt",
        "return_attention_mask": False,
    }
}

class TextProcessor:
    def __init__(self, config=text_processor_default_config):
        self.tokenizer: PreTrainedTokenizerFast = \
         PreTrainedTokenizerFast.from_pretrained(config["tokenizer_path"])
        self.tokenizer_config = config["tokenizer_config"]

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