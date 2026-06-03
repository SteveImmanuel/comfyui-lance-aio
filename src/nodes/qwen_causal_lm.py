import os.path as osp
import folder_paths
import torch
import comfy.model_management as mm
import logging
import comfy.ops
import comfy.utils
import comfy.model_patcher
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from comfy.text_encoders.qwen_vl import Qwen2VLVisionTransformer
import os.path as osp
from ..model.qwen2.modeling_qwen2 import Qwen2Config
from ..model.lance.qwen2_navit import Qwen2ForCausalLM
from .utils import swap_to_manual_cast

class Qwen2CausalLMLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("Qwen2CausalLM",)
    FUNCTION = "load"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": (folder_paths.get_filename_list("diffusion_models"),),
                "low_memory": ("BOOLEAN", {"default": False}),
            }
        }

    def load(self, ckpt_path: str, low_memory: bool):
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", ckpt_path)
        config_path = osp.join(osp.dirname(__file__), '..', '..', 'config', 'qwen_2_causal_lm.json')
        
        llm_config = Qwen2Config.from_json_file(config_path)

        llm_config.layer_module = 'Qwen2MoTDecoderLayer'
        llm_config.qk_norm = True
        llm_config.qk_norm_und = True
        llm_config.qk_norm_gen = True
        llm_config.tie_word_embeddings = False
        llm_config.freeze_und = False
        llm_config.apply_qwen_2_5_vl_pos_emb = True

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