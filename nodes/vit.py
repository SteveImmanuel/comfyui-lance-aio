import logging
import os
import os.path as osp

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
import torch
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from ..constants import CKPT_ROOT_DIR


class VitLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("VIT", "VIT_CONFIG")
    FUNCTION = "load"
    # OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        dirs = (
            sorted(d for d in os.listdir(CKPT_ROOT_DIR) if osp.isdir(osp.join(CKPT_ROOT_DIR, d)))
            if osp.isdir(CKPT_ROOT_DIR)
            else []
        )
        return {
            "required": {
                "ckpt_dir": (dirs, {"default": "Qwen2.5-VL-ViT"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, ckpt_dir: str):
        full = osp.join(CKPT_ROOT_DIR, ckpt_dir)
        for f in ("vit.safetensors", "config.json"):
            if not osp.isfile(osp.join(full, f)):
                return f"{f} not found in {full}"
        return True

    def load(self, ckpt_dir: str):
        ckpt_dir = osp.join(CKPT_ROOT_DIR, ckpt_dir)
        ckpt_path = osp.join(ckpt_dir, "vit.safetensors")
        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(ckpt_dir)

        vit = Qwen2VLVisionTransformer(
            hidden_size=vit_config.hidden_size,
            output_hidden_size=vit_config.out_hidden_size,
            intermediate_size=vit_config.intermediate_size,
            num_heads=vit_config.num_heads,
            num_layers=vit_config.depth,
            patch_size=vit_config.patch_size,
            temporal_patch_size=vit_config.temporal_patch_size,
            spatial_merge_size=vit_config.spatial_merge_size,
            window_size=vit_config.window_size,
            device=mm.unet_offload_device(),
            dtype=torch.bfloat16,
            ops=comfy.ops.manual_cast,
        )
        vit.eval()

        patcher = comfy.model_patcher.CoreModelPatcher(
            vit,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )
        patcher.set_model_compute_dtype(torch.bfloat16)

        ckpt = comfy.utils.load_torch_file(ckpt_path)
        missing, unexpected = vit.load_state_dict(ckpt, strict=False, assign=patcher.is_dynamic())
        if missing or unexpected:
            logging.warning(
                "[Lance VitLoader] state_dict mismatch: missing=%d unexpected=%d %s",
                len(missing),
                len(unexpected),
                (missing[:5] + unexpected[:5]),
            )

        return ({"patcher": patcher, "module": vit}, vit_config)
