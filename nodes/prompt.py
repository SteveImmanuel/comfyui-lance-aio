import json
import tempfile

from torch.utils.data import DataLoader

from ..data.dataset_base import DataConfig, simple_custom_collate
from ..data.datasets_custom import ValidationDataset
from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer


class LancePrompt:
    CATEGORY = "Lance"
    RETURN_TYPES = ("DATA_LOADER",)
    FUNCTION = "load"
    # OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "data_config": ("DATA_CONFIG",),
                "tokenizer": ("TOKENIZER",),
                "new_token_ids": ("NEW_TOKEN_IDS",),
            }
        }

    def load(
        self,
        prompt: str,
        data_config: DataConfig,
        tokenizer: Qwen2Tokenizer,
        new_token_ids: dict,
    ):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps({"index": 0, "data": prompt}))
            prompt_path = f.name

        dataset = ValidationDataset(
            jsonl_path=prompt_path,
            tokenizer=tokenizer,
            data_args=None,
            model_args=None,
            training_args=None,
            new_token_ids=new_token_ids,
            dataset_config=data_config,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=simple_custom_collate,
            drop_last=True,
            prefetch_factor=None,
            persistent_workers=False,
            multiprocessing_context=None,
        )

        return (data_loader,)
