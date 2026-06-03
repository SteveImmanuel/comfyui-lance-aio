import os.path as osp
import folder_paths
import comfy.model_management as mm
import torch
import logging
import comfy.ops
import comfy.utils
import comfy.model_patcher
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer
import os.path as osp
from ..config.config_factory import ModelArguments


class VitLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("VIT", "VIT_CONFIG")
    FUNCTION = "load"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_args": ("MODEL_ARGS",),
            }
        }

    def load(self, model_args: ModelArguments):
        ckpt_path = osp.join(model_args.vit_path, "vit.safetensors")
        vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
        
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
                len(missing), len(unexpected), (missing[:5] + unexpected[:5]),
            )

        return ({'patcher': patcher, 'module': vit}, vit_config)

