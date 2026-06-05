from ..common.utils.misc import AutoEncoderParams, tuple_mul
from ..config.config_factory import InferenceArguments, ModelArguments
from ..data.dataset_base import DataConfig
from ..constants import ALL_TASKS, RESOLUTION_CONFIGS, VAE_CONFIG


class LanceArgs:
    CATEGORY = "Lance/config"
    RETURN_TYPES = ("INFERENCE_ARGS","MODEL_ARGS", "DATA_CONFIG")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
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
            }
        }

    def build(self, **kw):
        cfg_text_scale = kw.pop("cfg_text_scale")

        inference_args = InferenceArguments(
            apply_qwen_2_5_vl_pos_emb=True,
            vae_model_type='wan',
            save_path_gen='',
            visual_gen=True,
            visual_und=True,
            copy_init_moe=True,
            enhance_prompt=False,
            **kw
        )

        model_args = ModelArguments(
            vit_type= 'qwen_2_5_vl_original',
            latent_patch_size=[1,1,1], 
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
        data_cfg.target_modality = "image" if inference_args.task in ("t2i", "x2t_image", "image_edit") else "video"
        data_cfg.resolution = inference_args.resolution
        data_cfg.text_template = inference_args.text_template
        data_cfg.enhance_prompt = inference_args.enhance_prompt
        data_cfg.system_prompt_type = inference_args.system_prompt_type

        print(model_args)
        print()
        print(inference_args)
        print()
        print(data_cfg)
        print()

        return (inference_args, model_args, data_cfg)

