import os
import os.path as osp

import folder_paths

from ..common.utils.misc import tuple_mul
from ..config.config_factory import InferenceArguments, ModelArguments
from ..constants import ALL_TASKS, IMAGE_OUTPUT_TASKS, RESOLUTION_CONFIGS, VAE_CONFIG
from ..data.dataset_base import DataConfig


class LanceArgs:
    CATEGORY = "Lance/config"
    RETURN_TYPES = ("MODEL_ARGS", "INFERENCE_ARGS", "DATA_CONFIG", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_args", "inference_args", "data_config", "LANCE_CKPT_DIR", "VIT_CKPT_DIR", "WAN_CKPT_PATH")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        dirs = [x for x in os.listdir(folder_paths.models_dir) if osp.isdir(osp.join(folder_paths.models_dir, x))]
        return {
            "required": {
                "task": (ALL_TASKS, {"default": "t2i"}),
                "resolution": (RESOLUTION_CONFIGS, {"default": "image_768res"}),
                "video_height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16}),
                "video_width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 50, "min": 1, "max": 1024}),
                "text_template": ("BOOLEAN", {"default": True}),
                "use_KVcache": ("BOOLEAN", {"default": True}),
                "validation_num_timesteps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "validation_timestep_shift": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "ckpt_root_dir": (dirs, {"default": "lance"}),
            }
        }

    def build(self, **kw):
        cfg_text_scale = kw.pop("cfg_text_scale")
        ckpt_root_dir = kw.pop("ckpt_root_dir")

        inference_args = InferenceArguments(
            apply_qwen_2_5_vl_pos_emb=True,
            vae_model_type="wan",
            save_path_gen="",
            visual_gen=True,
            visual_und=True,
            copy_init_moe=True,
            enhance_prompt=False,
            **kw,
        )

        model_args = ModelArguments(
            vit_type="qwen_2_5_vl_original",
            latent_patch_size=[1, 1, 1],
            max_latent_size=64,
            max_num_frames=121,
            cfg_text_scale=cfg_text_scale,
            tie_word_embeddings=False,
            llm_qk_norm=True,
            llm_qk_norm_und=True,
            llm_qk_norm_gen=True,
        )

        data_cfg = DataConfig()

        if inference_args.visual_und:
            data_cfg.vit_patch_size = model_args.vit_patch_size
            data_cfg.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            data_cfg.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

        if inference_args.visual_gen:
            data_cfg.latent_patch_size = model_args.latent_patch_size
            data_cfg.vae_downsample = tuple_mul(
                model_args.latent_patch_size,
                (VAE_CONFIG.downsample_temporal, VAE_CONFIG.downsample_spatial, VAE_CONFIG.downsample_spatial),
            )
            data_cfg.max_latent_size = model_args.max_latent_size
            data_cfg.max_num_frames = model_args.max_num_frames

        data_cfg.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        data_cfg.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        data_cfg.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        data_cfg.num_frames = inference_args.num_frames
        data_cfg.H = inference_args.video_height
        data_cfg.W = inference_args.video_width
        data_cfg.task = inference_args.task
        data_cfg.target_modality = "image" if inference_args.task in IMAGE_OUTPUT_TASKS else "video"
        data_cfg.resolution = inference_args.resolution
        data_cfg.text_template = inference_args.text_template
        data_cfg.enhance_prompt = inference_args.enhance_prompt
        data_cfg.system_prompt_type = inference_args.system_prompt_type

        if inference_args.task in IMAGE_OUTPUT_TASKS:
            lance_ckpt_dir = osp.join(folder_paths.models_dir, ckpt_root_dir, "Lance_3B")
        else:
            lance_ckpt_dir = osp.join(folder_paths.models_dir, ckpt_root_dir, "Lance_3B_Video")
        vit_ckpt_dir = osp.join(folder_paths.models_dir, ckpt_root_dir, "Qwen2.5-VL-ViT")
        wan_ckpt_path = osp.join(folder_paths.models_dir, ckpt_root_dir, "Wan2.2_VAE.pth")

        return (model_args, inference_args, data_cfg, lance_ckpt_dir, vit_ckpt_dir, wan_ckpt_path)
