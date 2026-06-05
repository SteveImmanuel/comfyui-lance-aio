import gc

import comfy.model_management as mm
import torch
from torch.utils.data import DataLoader

from ..common.val.utils import make_padded_latent
from ..config.config_factory import InferenceArguments, ModelArguments
from ..constants import (
    GENERATION_TASKS,
    LATENT_FORMAT,
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

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _prepare(lance: Lance, data_loader: DataLoader, device):
    for m in (lance.time_embedder, lance.vae2llm, lance.llm2vae, lance.latent_pos_embed):
        m.to(device)
    batch: SimpleCustomBatch = next(iter(data_loader))
    data_dict: dict = batch.cuda(device).to_dict()
    return batch, data_dict


class LanceGeneration:
    CATEGORY = "Lance"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_args": ("MODEL_ARGS",),
                "inference_args": ("INFERENCE_ARGS",),
                "data_loader": ("DATA_LOADER",),
                "lance": ("LANCE",),
                "new_token_ids": ("NEW_TOKEN_IDS",),
                "qwen2_causal_lm": ("QWEN_2_CAUSAL_LM",),
                "vae": ("VAE",),
            },
            "optional": {
                "vit": ("VIT",),
            },
        }

    def generate(
        self,
        model_args: ModelArguments,
        inference_args: InferenceArguments,
        data_loader: DataLoader,
        lance: Lance,
        new_token_ids: dict,
        qwen2_causal_lm: dict,
        vae,
        vit: dict = None,
    ):
        if inference_args.task not in GENERATION_TASKS:
            raise ValueError(f"task '{inference_args.task}' is not a generation task")

        device = mm.get_torch_device()
        batch, data_dict = _prepare(lance, data_loader, device)
        image_token_id = lance.language_model.config.video_token_id

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if "padded_videos" in data_dict.keys():
                data_dict["padded_latent"] = make_padded_latent(
                    data_dict["padded_videos"], data_dict["vae_data_mode"], ComfyVAEAdapter(vae)
                )

            patchers = [qwen2_causal_lm["patcher"]]
            if vit:
                patchers.append(vit["patcher"])
            mm.load_models_gpu(patchers)

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
                "caption": data_dict.get("caption", None),
                "sample_task": data_dict["sample_task"],
                "sample_modality": data_dict["sample_modality"],
                "cfg_type": inference_args.cfg_type,
                "cfg_uncond_token_id": inference_args.cfg_uncond_token_id,
                "index": data_dict["index"],
            }

            if inference_args.use_KVcache:
                denoise_latent, _, padded_videos, _ = lance.validation_gen_KVcache(**params)
            else:
                denoise_latent, _, padded_videos, _ = lance.validation_gen(**params)

            # decode phase, only the VAE is needed from here on
            del params, padded_videos, batch
            data_dict.clear()
            mm.unload_all_models()
            _clean_memory()

            if inference_args.task in {TASK_I2V, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}:
                target_latents = [denoise_latent[0][-1]]
            else:
                target_latents = denoise_latent[0]

            frames = []
            for latent in target_latents:
                z = LATENT_FORMAT.process_out(latent.unsqueeze(0).movedim(-1, 1))  # [t,h,w,c] -> [1,c,t,h,w], *std+mean

                img = vae.decode(z)  # [B,T,H,W,C] float 0..1
                if img.ndim == 5:
                    img = img.reshape(-1, *img.shape[-3:])
                frames.append(img.cpu())

            image = torch.cat(frames, dim=0)
            return (image,)


class LanceUnderstanding:
    CATEGORY = "Lance"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_args": ("INFERENCE_ARGS",),
                "data_loader": ("DATA_LOADER",),
                "lance": ("LANCE",),
                "new_token_ids": ("NEW_TOKEN_IDS",),
                "tokenizer": ("TOKENIZER",),
                "qwen2_causal_lm": ("QWEN_2_CAUSAL_LM",),
                "vit": ("VIT",),
            },
        }

    def generate(
        self,
        inference_args: InferenceArguments,
        data_loader: DataLoader,
        lance: Lance,
        new_token_ids: dict,
        tokenizer: Qwen2Tokenizer,
        qwen2_causal_lm: dict,
        vit: dict,
    ):
        if inference_args.task not in UNDERSTANDING_TASKS:
            raise ValueError(f"task '{inference_args.task}' is not an understanding task")

        device = mm.get_torch_device()
        batch, data_dict = _prepare(lance, data_loader, device)
        image_token_id = lance.language_model.config.video_token_id

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            mm.load_models_gpu([qwen2_causal_lm["patcher"], vit["patcher"]])

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
                generated_sequence_all, captions, _ = lance.validation_und_KVcache(**params)
            else:
                generated_sequence_all, captions, _ = lance.validation_video_to_text(**params)

            caps = [tokenizer.decode(seq[:, 0]).replace("<|im_end|>", "").strip() for seq in generated_sequence_all]

            del generated_sequence_all, captions, params, batch
            data_dict.clear()
            _clean_memory()

        text = "\n".join(caps)
        return {"ui": {"text": (text,)}, "result": (text,)}
