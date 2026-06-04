import os
import os.path as osp

from ..constants import CKPT_ROOT_DIR
from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer


class LanceTokenizerLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("TOKENIZER",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        dirs = (
            sorted(d for d in os.listdir(CKPT_ROOT_DIR) if osp.isdir(osp.join(CKPT_ROOT_DIR, d)))
            if osp.isdir(CKPT_ROOT_DIR)
            else []
        )
        return {
            "required": {
                "ckpt_dir": (dirs, {"default": "Lance_3B"}),
            }
        }

    def load(self, ckpt_dir: str):
        tokenizer = Qwen2Tokenizer.from_pretrained(osp.join(CKPT_ROOT_DIR, ckpt_dir))
        return (tokenizer,)
