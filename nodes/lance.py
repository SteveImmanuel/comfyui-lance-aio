import logging
import os.path as osp

import comfy.utils
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from ..config.config_factory import InferenceArguments, ModelArguments
from ..constants import VAE_CONFIG
from ..data.data_utils import add_special_tokens
from ..modeling.lance.lance import Lance, LanceConfig
from ..modeling.lance.qwen2_navit import Qwen2ForCausalLM
from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer


class LanceLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("LANCE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_dir": ("STRING",),
                "model_args": ("MODEL_ARGS",),
                "inference_args": ("INFERENCE_ARGS",),
                "qwen2_causal_lm": ("QWEN_2_CAUSAL_LM",),
            },
            "optional": {
                "vit": ("VIT",),
                "vit_config": ("VIT_CONFIG",),
            },
        }

    def load(
        self,
        ckpt_dir: str,
        model_args: ModelArguments,
        inference_args: InferenceArguments,
        qwen2_causal_lm: dict,
        vit: dict = None,
        vit_config: Qwen2_5_VLVisionConfig = None,
    ):
        ckpt_path = osp.join(ckpt_dir, "model.safetensors")

        language_model: Qwen2ForCausalLM = qwen2_causal_lm["module"]
        vit_model: Qwen2VLVisionTransformer = vit["module"] if vit is not None else None
        llm_config = language_model.config

        config = LanceConfig(
            visual_gen=inference_args.visual_gen,
            visual_und=inference_args.visual_und,
            llm_config=llm_config,
            vit_config=vit_config if inference_args.visual_und else None,
            vae_config=VAE_CONFIG if inference_args.visual_gen else None,
            latent_patch_size=model_args.latent_patch_size,
            max_num_frames=model_args.max_num_frames,
            max_latent_size=model_args.max_latent_size,
            vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            connector_act=model_args.connector_act,
            interpolate_pos=model_args.interpolate_pos,
            timestep_shift=inference_args.timestep_shift,
        )

        lance = Lance(
            language_model=language_model,
            vit_model=vit_model,
            vit_type=model_args.vit_type,
            config=config,
            training_args=inference_args,
        )

        ckpt = comfy.utils.load_torch_file(ckpt_path)
        glue_sd = {
            k: v
            for k, v in ckpt.items()
            if not k.startswith(("language_model.", "vit_model.")) and k != "latent_pos_embed.pos_embed"
        }
        missing, unexpected = lance.load_state_dict(glue_sd, strict=False)
        if missing or unexpected:
            logging.warning(
                "[Lance VitLoader] state_dict mismatch: missing=%d unexpected=%d %s",
                len(missing),
                len(unexpected),
                (missing[:5] + unexpected[:5]),
            )
        return (lance,)


class LanceConfigure:
    CATEGORY = "Lance"
    RETURN_TYPES = ("LANCE", "NEW_TOKEN_IDS")
    FUNCTION = "configure"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lance": ("LANCE",),
                "tokenizer": ("TOKENIZER",),
            },
        }

    def configure(
        self,
        lance: Lance,
        tokenizer: Qwen2Tokenizer,
    ):
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)

        if num_new_tokens > 0:
            lance.language_model.resize_token_embeddings(len(tokenizer))
            lance.config.llm_config.vocab_size = len(tokenizer)
            lance.language_model.config.vocab_size = len(tokenizer)

        image_token_id = lance.language_model.config.video_token_id
        new_token_ids.update({"image_token_id": image_token_id})
        lance.update_tokenizer(tokenizer=tokenizer)

        assert (
            lance.language_model.get_input_embeddings().weight.data.data_ptr()
            != lance.language_model.get_output_embeddings().weight.data.data_ptr()
        ), "tie_word_embeddings conflict"

        lance.eval()
        return (lance, new_token_ids)
