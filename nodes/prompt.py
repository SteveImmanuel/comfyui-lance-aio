import json
import tempfile

import imageio
import torch
from comfy_api.latest import Input
from torch.utils.data import DataLoader

from ..constants import UNDERSTANDING_TASKS
from ..data.dataset_base import DataConfig, simple_custom_collate
from ..data.datasets_custom import ValidationDataset
from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer


def _media_record(prompt: str, media_path: str, dtype: str, task: str) -> dict:
    if task in UNDERSTANDING_TASKS:
        vision = "image" if dtype == "image" else "video"
        payload = [f"Look at the {vision} carefully and answer the question.", prompt, "..."]
        return {
            "index": 0,
            "data": {
                "interleave_array": [media_path, payload],
                "element_dtype_array": [dtype, "text"],
                "istarget_in_interleave": [0, 1],
            },
        }

    return {
        "index": 0,
        "data": {
            "interleave_array": [prompt, media_path],
            "element_dtype_array": ["text", dtype],
            "istarget_in_interleave": [0, 0],
        },
    }


def _write_records(records: list) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in records))
        return f.name


def _save_image(image: torch.Tensor) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    imageio.imwrite(path, (image.cpu().numpy() * 255).round().astype("uint8"))
    return path


def _save_video(video: Input.Video) -> str:
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    video.save_to(path)
    return path


class LanceTextPrompt:
    CATEGORY = "Lance"
    RETURN_TYPES = ("DATA_LOADER",)
    FUNCTION = "load"

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

    def _build_loader(self, prompt_path: str, data_config: DataConfig, tokenizer: Qwen2Tokenizer, new_token_ids: dict):
        dataset = ValidationDataset(
            jsonl_path=prompt_path,
            tokenizer=tokenizer,
            data_args=None,
            model_args=None,
            training_args=None,
            new_token_ids=new_token_ids,
            dataset_config=data_config,
        )

        return DataLoader(
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

    def load(
        self,
        prompt: str,
        data_config: DataConfig,
        tokenizer: Qwen2Tokenizer,
        new_token_ids: dict,
    ):
        prompt_path = _write_records([{"index": 0, "data": prompt}])
        return (self._build_loader(prompt_path, data_config, tokenizer, new_token_ids),)


class LanceTextImagePrompt(LanceTextPrompt):
    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["image"] = ("IMAGE",)
        return types

    def load(
        self,
        prompt: str,
        data_config: DataConfig,
        tokenizer: Qwen2Tokenizer,
        new_token_ids: dict,
        image,
    ):
        record = _media_record(prompt, _save_image(image[0]), "image", data_config.task)
        prompt_path = _write_records([record])
        return (self._build_loader(prompt_path, data_config, tokenizer, new_token_ids),)


class LanceTextVideoPrompt(LanceTextPrompt):
    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["video"] = ("VIDEO",)
        return types

    def load(
        self,
        prompt: str,
        data_config: DataConfig,
        tokenizer: Qwen2Tokenizer,
        new_token_ids: dict,
        video,
    ):
        record = _media_record(prompt, _save_video(video), "video", data_config.task)
        prompt_path = _write_records([record])
        return (self._build_loader(prompt_path, data_config, tokenizer, new_token_ids),)
