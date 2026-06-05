from ..common.utils.misc import AutoEncoderParams, tuple_mul
from ..config.config_factory import DataArguments, InferenceArguments, ModelArguments
from ..data.dataset_base import DataConfig


def _parse_ints(s):
    return tuple(int(x) for x in s.replace(" ", "").split(","))


class LanceModelArgs:
    CATEGORY = "Lance/config"
    RETURN_TYPES = ("MODEL_ARGS",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_module": ("STRING", {"default": "Qwen2MoTDecoderLayer"}),
                "llm_qk_norm": ("BOOLEAN", {"default": True}),
                "llm_qk_norm_und": ("BOOLEAN", {"default": True}),
                "llm_qk_norm_gen": ("BOOLEAN", {"default": True}),
                "tie_word_embeddings": ("BOOLEAN", {"default": False}),
                "latent_patch_size": ("STRING", {"default": "1,1,1"}),
                "max_latent_size": ("INT", {"default": 64, "min": 1, "max": 1024}),
                "max_num_frames": ("INT", {"default": 121, "min": 1, "max": 1024}),
                "vit_type": ("STRING", {"default": "qwen_2_5_vl_original"}),
                "vit_path": ("STRING", {"default": ""}),
                "vit_patch_size": ("INT", {"default": 14, "min": 1, "max": 64}),
                "vit_max_num_patch_per_side": ("INT", {"default": 70, "min": 1, "max": 1024}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            }
        }

    def build(self, latent_patch_size, **kw):
        args = ModelArguments(latent_patch_size=_parse_ints(latent_patch_size), **kw)
        return (args,)


class LanceInferenceArgs:
    CATEGORY = "Lance/config"
    RETURN_TYPES = ("INFERENCE_ARGS",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (["t2i", "t2v", "i2v", "image_edit", "video_edit", "x2t_image", "x2t_video"], {"default": "t2i"}),
                "resolution": (["video_192p", "video_360p", "video_480p", "image_256res", "image_512res", "image_768res"], {"default": "image_768res"}),
                "video_height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16}),
                "video_width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 50, "min": 1, "max": 1024}),
                "text_template": ("BOOLEAN", {"default": True}),
                "apply_qwen_2_5_vl_pos_emb": ("BOOLEAN", {"default": True}),
                "visual_gen": ("BOOLEAN", {"default": True}),
                "visual_und": ("BOOLEAN", {"default": True}),
                "freeze_und": ("BOOLEAN", {"default": False}),
                "copy_init_moe": ("BOOLEAN", {"default": True}),
                "use_KVcache": ("BOOLEAN", {"default": True}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "timestep_shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "validation_num_timesteps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "validation_timestep_shift": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_renorm_type": ("STRING", {"default": "global"}),
                "cfg_renorm_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "system_prompt_type": ("STRING", {"default": "SP0"}),
            }
        }

    def build(self, **kw):
        inference_args = InferenceArguments(**kw)
        inference_args.vae_model_type = "wan"
        inference_args.save_path_gen = ""

        return (inference_args,)


class LanceDataConfig:
    CATEGORY = "Lance/config"
    RETURN_TYPES = ("DATA_CONFIG",)
    FUNCTION = "build"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_args": ("MODEL_ARGS",),
                "inference_args": ("INFERENCE_ARGS",),
                "vae_config": ("VAE_CONFIG",),
            }
        }

    def build(
        self,
        model_args: ModelArguments,
        inference_args: InferenceArguments,
        vae_config: AutoEncoderParams,
    ):
        cfg = DataConfig()

        if inference_args.visual_und:
            cfg.vit_patch_size = model_args.vit_patch_size
            cfg.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            cfg.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side

        if inference_args.visual_gen:
            cfg.latent_patch_size = model_args.latent_patch_size
            cfg.vae_downsample = tuple_mul(
                model_args.latent_patch_size,
                (vae_config.downsample_temporal, vae_config.downsample_spatial, vae_config.downsample_spatial),
            )
            cfg.max_latent_size = model_args.max_latent_size
            cfg.max_num_frames = model_args.max_num_frames

        cfg.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        cfg.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        cfg.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        cfg.num_frames = inference_args.num_frames
        cfg.H = inference_args.video_height
        cfg.W = inference_args.video_width
        cfg.task = inference_args.task
        cfg.target_modality = "image" if inference_args.task in ("t2i", "x2t_image", "image_edit") else "video"
        cfg.resolution = inference_args.resolution
        cfg.text_template = inference_args.text_template
        cfg.enhance_prompt = inference_args.enhance_prompt
        cfg.system_prompt_type = inference_args.system_prompt_type

        return (cfg,)
