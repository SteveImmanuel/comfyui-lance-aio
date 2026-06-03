from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer
from ..config.config_factory import ModelArguments


class LanceTokenizerLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("TOKENIZER",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_args": ("MODEL_ARGS",),
            }
        }

    def load(self, model_args: ModelArguments):
        tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
        return (tokenizer,)
