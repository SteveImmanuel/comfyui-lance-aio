import comfy.latent_formats
import comfy.model_management as mm
import torch
from torch.utils.data import DataLoader

from ..common.val.utils import make_padded_latent
from ..config.config_factory import InferenceArguments, ModelArguments
from ..constants import (
    GENERATION_TASKS,
    MAX_GENERATION_LENGTH,
    TASK_I2V,
    TASK_IMAGE_EDIT,
    TASK_VIDEO_EDIT,
    UNDERSTANDING_TASKS,
)
from ..data.dataset_base import SimpleCustomBatch
from ..modeling.lance.lance import Lance
from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer
from .vae import ComfyVAEAdapter


def _clean_memory(*objects):
    """Clear temporary container references and release unused GPU allocator cache."""
    for obj in objects:
        if isinstance(obj, dict):
            obj.clear()
        elif isinstance(obj, (list, set)):
            obj.clear()
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class LanceGenerate:
    CATEGORY = "Lance"
    RETURN_TYPES = ()
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # NaN != NaN → always considered changed

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_args": ("MODEL_ARGS",),
                "inference_args": ("INFERENCE_ARGS",),
                "data_loader": ("DATA_LOADER",),
                "lance": ("LANCE",),
                "new_token_ids": ("NEW_TOKEN_IDS",),
                "tokenizer": ("TOKENIZER",),
                "qwen2_causal_lm": ("QWEN_2_CAUSAL_LM",),
            },
            "optional": {
                "vit": ("VIT",),
                "vae": ("VAE",),
            },
        }

    def generate(
        self,
        model_args: ModelArguments,
        inference_args: InferenceArguments,
        data_loader: DataLoader,
        lance: Lance,
        new_token_ids: dict,
        tokenizer: Qwen2Tokenizer,
        qwen2_causal_lm: dict,
        vit: dict = None,
        vae=None,
    ):
        device = mm.get_torch_device()

        lance.time_embedder.to(device)
        lance.vae2llm.to(device)
        lance.llm2vae.to(device)
        lance.latent_pos_embed.to(device)

        batch: SimpleCustomBatch = next(iter(data_loader))
        data_dict: dict = batch.cuda(device).to_dict()
        image_token_id = lance.language_model.config.video_token_id

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            # encode phase: comfy's VAE.encode stages itself (and tiles on OOM)
            if "padded_videos" in data_dict.keys():
                data_dict["padded_latent"] = make_padded_latent(
                    data_dict["padded_videos"], data_dict["vae_data_mode"], ComfyVAEAdapter(vae)
                )

            # # denoise phase: stage LLM(+ViT), reserving headroom for the KV cache + per-step activations
            # llm_cfg = lance.language_model.config
            # n_tokens = int(sum(data_dict["sample_lens"]))
            # head_dim = llm_cfg.hidden_size // llm_cfg.num_attention_heads
            # kv_bytes = n_tokens * llm_cfg.num_hidden_layers * 2 * llm_cfg.num_key_value_heads * head_dim * 2
            # act_bytes = n_tokens * llm_cfg.hidden_size * 2 * 64  # ~64 live bf16 tensors per token, empirical fudge
            patchers = [qwen2_causal_lm["patcher"]]
            if vit:
                patchers.append(vit["patcher"])
            # mm.load_models_gpu(patchers, memory_required=kv_bytes + act_bytes)
            mm.load_models_gpu(patchers)

            if inference_args.task in GENERATION_TASKS:
                params = {
                    "val_packed_text_ids": data_dict["packed_text_ids"],
                    "val_packed_text_indexes": data_dict["packed_text_indexes"],
                    "val_sample_lens": data_dict["sample_lens"],
                    "val_packed_position_ids": data_dict["packed_position_ids"],
                    "val_split_lens": data_dict["split_lens"],
                    "val_attn_modes": data_dict["attn_modes"],
                    "val_sample_N_target": data_dict["sample_N_target"],
                    "val_packed_vae_token_indexes": data_dict["packed_vae_token_indexes"],
                    "timestep_shift": inference_args.validation_timestep_shift,
                    "num_timesteps": inference_args.validation_num_timesteps,
                    "val_mse_loss_indexes": data_dict.get("mse_loss_indexes", None),
                    "val_padded_latent": data_dict["padded_latent"],
                    "video_sizes": data_dict["video_sizes"],
                    "cfg_text_scale": model_args.cfg_text_scale,
                    "cfg_interval": inference_args.cfg_interval,
                    "cfg_renorm_min": inference_args.cfg_renorm_min,
                    "cfg_renorm_type": inference_args.cfg_renorm_type,
                    "device": device,
                    "dtype": torch.bfloat16,
                    "new_token_ids": new_token_ids,
                    "max_samples": inference_args.validation_max_samples,
                    "validation_noise_seed": inference_args.validation_noise_seed,
                    "apply_chat_template": inference_args.apply_chat_template,
                    "apply_qwen_2_5_vl_pos_emb": inference_args.apply_qwen_2_5_vl_pos_emb,
                    "image_token_id": image_token_id,
                    "val_packed_vit_token_indexes": data_dict.get("packed_vit_token_indexes", None),
                    "val_packed_vit_tokens": data_dict.get("packed_vit_tokens", None),
                    "vit_video_grid_thw": data_dict.get("vit_video_grid_thw", None),
                    "vae_video_grid_thw": data_dict["vae_video_grid_thw"],
                    "video_grid_thw": data_dict.get("video_grid_thw", None),
                    "caption": data_dict.get(
                        "caption", None
                    ),  # The dataset uses "caption" as the default caption field.
                    "sample_task": data_dict["sample_task"],
                    "sample_modality": data_dict["sample_modality"],
                    "cfg_type": inference_args.cfg_type,
                    "cfg_uncond_token_id": inference_args.cfg_uncond_token_id,
                    "index": data_dict["index"],
                    # "val_padded_videos": data_dict["padded_videos"] if save_source_video else None,
                }
                if inference_args.use_KVcache:
                    denoise_latent, captions, padded_videos, index = lance.validation_gen_KVcache(**params)
                else:
                    denoise_latent, captions, padded_videos, index = lance.validation_gen(**params)

                # decode phase: drop denoise-phase refs; only the VAE is needed from here on
                del params, padded_videos, batch
                data_dict.clear()
                mm.unload_all_models()
                _clean_memory()

                # comfy VAE path: un-normalize via Wan22 latent format, decode with
                # comfy's own staging + tiled-on-OOM fallback
                lf = comfy.latent_formats.Wan22()
                for i_val, latent in enumerate(denoise_latent):
                    if inference_args.task in {TASK_I2V, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}:
                        target_latents = [latent[-1]]
                    else:
                        target_latents = latent

                    frames = []
                    for latent_ in target_latents:
                        z = lf.process_out(latent_.unsqueeze(0).movedim(-1, 1))  # [t,h,w,c] -> [1,c,t,h,w], *std+mean
                        img = vae.decode(z)  # [B,T,H,W,C] float 0..1
                        if img.ndim == 5:
                            img = img.reshape(-1, *img.shape[-3:])
                        frames.append(img.cpu())
                    image = torch.cat(frames, dim=0)
                    return (image,)

            elif inference_args.task in UNDERSTANDING_TASKS:
                params = {
                    "val_packed_text_ids": data_dict["packed_text_ids"],
                    "val_packed_text_indexes": data_dict["packed_text_indexes"],
                    "val_packed_position_ids": data_dict["packed_position_ids"],
                    "val_sample_N_target": data_dict["sample_N_target"],
                    "val_split_lens": data_dict["split_lens"],
                    "val_attn_modes": data_dict["attn_modes"],
                    "val_sample_lens": data_dict["sample_lens"],
                    "val_sample_type": data_dict["sample_type"],
                    "val_packed_vit_tokens": data_dict["packed_vit_tokens"],
                    "val_vit_video_grid_thw": data_dict["vit_video_grid_thw"],
                    "val_ce_loss_indexes": data_dict["ce_loss_indexes"],
                    "max_samples": inference_args.validation_max_samples,
                    "max_length": MAX_GENERATION_LENGTH,
                    "device": device,
                    "dtype": torch.bfloat16,
                    "new_token_ids": new_token_ids,
                    "pad_token_id": tokenizer.pad_token_id,
                    "vocab_size": len(tokenizer),
                    "caption": data_dict.get("caption_cn", None),
                    "tokenizer": tokenizer,
                    "apply_chat_template": inference_args.apply_chat_template,
                    "apply_qwen_2_5_vl_pos_emb": inference_args.apply_qwen_2_5_vl_pos_emb,
                    "do_sample": False,
                    "image_token_id": image_token_id,
                    "index": data_dict["index"],
                }
                if inference_args.use_KVcache:
                    generated_sequence_all, captions, index = lance.validation_und_KVcache(**params)
                else:
                    generated_sequence_all, captions, index = lance.validation_video_to_text(**params)

                for i_val, generated_sequence in enumerate(generated_sequence_all):
                    cap = tokenizer.decode(generated_sequence[:, 0])
                    # inference_args.prompt_data_dict[index] = f"target_caption: {captions} /// generated_caption: {cap} "
                    inference_args.prompt_data_dict[index] = f"{cap}"
                    del generated_sequence

                del generated_sequence_all, captions, params
                _clean_memory()

        del data_dict
        _clean_memory()
