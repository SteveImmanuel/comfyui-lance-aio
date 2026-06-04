import comfy.model_management as mm
import torch
from torch.utils.data import DataLoader

from ..common.val.utils import decode_video_tensor, make_padded_latent
from ..config.config_factory import InferenceArguments, ModelArguments
from ..data.dataset_base import SimpleCustomBatch
from ..modeling.lance.lance import Lance
from ..modeling.qwen2.tokenization_qwen2_fast import Qwen2Tokenizer
from ..modeling.vae.wan.model import WanVideoVAE

TASK_T2V = "t2v"
TASK_T2I = "t2i"
TASK_I2V = "i2v"
TASK_X2T_IMAGE = "x2t_image"
TASK_X2T_VIDEO = "x2t_video"
TASK_IMAGE_EDIT = "image_edit"
TASK_VIDEO_EDIT = "video_edit"
GENERATION_TASKS = {
    TASK_T2V,
    TASK_T2I,
    TASK_I2V,
    TASK_IMAGE_EDIT,
    TASK_VIDEO_EDIT,
}
UNDERSTANDING_TASKS = {
    TASK_X2T_IMAGE,
    TASK_X2T_VIDEO,
}
MAX_GENERATION_LENGTH = 256


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
                "wan_vae": ("WAN_VAE",),
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
        wan_vae: dict = None,
    ):
        device = mm.get_torch_device()

        patchers = [qwen2_causal_lm["patcher"]]
        if vit:
            patchers.append(vit["patcher"])
        if wan_vae:
            patchers.append(wan_vae["patcher"])

        mm.load_models_gpu(patchers)
        lance.time_embedder.to(device)
        lance.vae2llm.to(device)
        lance.llm2vae.to(device)
        lance.latent_pos_embed.to(device)

        batch: SimpleCustomBatch = next(iter(data_loader))
        data_dict: dict = batch.cuda(device).to_dict()
        image_token_id = lance.language_model.config.video_token_id
        wan_vae_module: WanVideoVAE = wan_vae["module"]

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if "padded_videos" in data_dict.keys():
                data_dict["padded_latent"] = make_padded_latent(
                    data_dict["padded_videos"], data_dict["vae_data_mode"], wan_vae_module
                )

            if inference_args.task in GENERATION_TASKS:
                save_fps = int(data_dict.get("save_fps", 12))
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

                # Decode.
                for i_val, latent in enumerate(denoise_latent):
                    if inference_args.task in {TASK_I2V, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}:
                        target_latents = [latent[-1]]
                    else:
                        target_latents = latent

                    v_list = []
                    for latent_ in target_latents:
                        v_list.append(wan_vae_module.vae_decode([latent_])[0])

                    save_item_name = f"{index:06d}" if isinstance(index, int) else index
                    v_thwc = decode_video_tensor(
                        v_list,
                        save_path=inference_args.save_path_gen,
                        # save_path=inference_args.save_path_gen,
                        save_half=False,
                        save_item_name=save_item_name,
                        save_fps=save_fps,
                    )

                    if v_thwc.shape[0] > 1:
                        prompt_data_path = f"{save_item_name}.mp4"
                    else:
                        prompt_data_path = f"{save_item_name}.png"
                    

                    image = torch.tensor(v_thwc) / 255.0
                    # return torch.tensor(v_thwc) / 255.0
                    return(image,)
                    # print(v_thwc.shape, v_thwc.max(), v_thwc.min())
                    # inference_args.prompt_data_dict[prompt_data_path] = captions[i_val]

                    # if save_source_video:
                    #     curr_padded_videos = padded_videos[i_val * 2 : (i_val + 1) * 2]
                    #     v_thwc_gt = decode_video_tensor(curr_padded_videos[-1:], save_path=save_path_gt, save_item_name=save_item_name, save_fps=save_fps)
                    #     del curr_padded_videos, v_thwc_gt

                    del v_list, v_thwc, latent, target_latents
                    _clean_memory()

                del denoise_latent, captions, padded_videos, params
                _clean_memory()

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
