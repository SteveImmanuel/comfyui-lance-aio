from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer


class LanceTokenizerLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("TOKENIZER",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_dir": ("STRING",),
            }
        }

    def load(self, ckpt_dir: str):
        tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_dir)
        return (tokenizer,)
