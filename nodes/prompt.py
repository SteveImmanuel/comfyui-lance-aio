from torch.utils.data import DataLoader

from ..config.config_factory import DataArguments, InferenceArguments, ModelArguments
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
                "data_args": ("DATA_ARGS",),
                "model_args": ("MODEL_ARGS",),
                "inference_args": ("INFERENCE_ARGS",),
                "data_config": ("DATA_CONFIG",),
                "tokenizer": ("TOKENIZER",),
                "new_token_ids": ("NEW_TOKEN_IDS",),
            }
        }

    def load(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        inference_args: InferenceArguments,
        data_config: DataConfig,
        tokenizer: Qwen2Tokenizer,
        new_token_ids: dict,
    ):
        # with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        #     json.dump({"0": prompt}, f)
        #     prompt_path = f.name

        dataset = ValidationDataset(
            jsonl_path=data_args.val_dataset_config_file,
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=inference_args,
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
