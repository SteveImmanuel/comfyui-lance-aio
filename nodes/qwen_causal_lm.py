import os.path as osp
import torch
import comfy.model_management as mm
import logging
import comfy.utils
import comfy.model_patcher
import os.path as osp
from ..modeling.qwen2.modeling_qwen2 import Qwen2Config
from ..modeling.lance.qwen2_navit import Qwen2ForCausalLM
from .utils import swap_to_manual_cast
from ..config.config_factory import ModelArguments, InferenceArguments

class Qwen2CausalLMLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("Qwen2CausalLM",)
    FUNCTION = "load"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_args": ("MODEL_ARGS",),
                "inference_args": ("INFERENCE_ARGS",),
                "low_memory": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, model_args: ModelArguments, inference_args: InferenceArguments, low_memory: bool):
        ckpt_path = osp.join(model_args.model_path, "model.safetensors")
        llm_config: Qwen2Config = Qwen2Config.from_json_file(osp.join(model_args.model_path, "llm_config.json"))

        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.qk_norm_und = model_args.llm_qk_norm_und
        llm_config.qk_norm_gen = model_args.llm_qk_norm_gen

        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = inference_args.freeze_und
        llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

        if low_memory:
            with torch.device("meta"):
                language_model = Qwen2ForCausalLM(llm_config)
        else:
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)

            try:
                language_model = Qwen2ForCausalLM(llm_config)
            finally:
                torch.set_default_dtype(default_dtype)
        language_model.eval()

        swap_to_manual_cast(language_model)
        patcher = comfy.model_patcher.CoreModelPatcher(
            language_model,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )
        patcher.set_model_compute_dtype(torch.bfloat16)

        ckpt = comfy.utils.load_torch_file(ckpt_path)
        # remap ckpt because lang model is fused together with lance
        prefix = "language_model."
        llm_sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}

        if low_memory:
            missing, unexpected = language_model.load_state_dict(llm_sd, strict=False, assign=True)
            # meta init leaves the non-persistent rope buffer on meta, rebuild it
            language_model.model.rotary_emb = type(language_model.model.rotary_emb)(config=llm_config)
        else:
            missing, unexpected = language_model.load_state_dict(llm_sd, strict=False, assign=patcher.is_dynamic())

        if missing or unexpected:
            logging.warning(
                "[Lance VitLoader] state_dict mismatch: missing=%d unexpected=%d %s",
                len(missing), len(unexpected), (missing[:5] + unexpected[:5]),
            )

        return (patcher,)