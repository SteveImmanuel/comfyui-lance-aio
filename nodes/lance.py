import folder_paths
import comfy.utils
import comfy.model_management as mm

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from ..modeling.lance.lance import Lance, LanceConfig
from ..config.config_factory import InferenceArguments
from ..common.utils.misc import AutoEncoderParams


class LanceLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("LANCE_MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen2_causal_lm": ("Qwen2CausalLM",),
                "vae": ("VAE",),
                "ckpt_path": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                "vit": ("VIT",),
            },
        }

    def load(self, qwen2_causal_lm, vae, ckpt_path, vit=None):
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", ckpt_path)

        llm_patcher = qwen2_causal_lm
        vit_patcher = vit

        language_model = llm_patcher.model
        vit_model = vit_patcher.model if vit_patcher is not None else None
        llm_config = language_model.config

        # Lance-level config (mirrors inference_lance.py:487-500). z_channels/downsample
        # describe the Wan2.2 VAE; latent_patch_size=(1,1,1) gives the shipped patch_latent_dim=48.
        vae_config = AutoEncoderParams(z_channels=48, downsample_spatial=16, downsample_temporal=4)
        vit_config = Qwen2_5_VLVisionConfig.from_json_file(...) if vit_model is not None else None
        training_args = InferenceArguments(
            visual_gen=True,
            visual_und=vit_model is not None,
            freeze_und=False,
            apply_qwen_2_5_vl_pos_emb=True,
            timestep_shift=1.0,
        )

        config = LanceConfig(
            visual_gen=True,
            visual_und=vit_model is not None,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=(1, 1, 1),
            max_num_frames=121,
            max_latent_size=64,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            interpolate_pos=False,
            timestep_shift=training_args.timestep_shift,
        )

        lance = Lance(
            language_model=language_model,
            vit_model=vit_model,
            vit_type="qwen_2_5_vl_original",
            config=config,
            training_args=training_args,
        )

        ckpt = comfy.utils.load_torch_file(ckpt_path)
        glue_sd = {
            k: v for k, v in ckpt.items()
            if not k.startswith(("language_model.", "vit_model."))
            and k != "latent_pos_embed.pos_embed"
        }
        missing, unexpected = lance.load_state_dict(glue_sd, strict=False)

        return ({
            "lance": lance,
            "llm_patcher": llm_patcher,
            "vit_patcher": vit_patcher,
            "vae": vae,
        },)
