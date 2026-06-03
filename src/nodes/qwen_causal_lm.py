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


def _swap_to_manual_cast(module: torch.nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            new = comfy.ops.manual_cast.Linear(
                child.in_features, 
                child.out_features,
                bias=child.bias is not None,
                device=torch.device("meta"), 
                dtype=child.weight.dtype,
            )
            new.weight = child.weight
            if child.bias is not None:
                new.bias = child.bias
            setattr(module, name, new)
        elif isinstance(child, torch.nn.Embedding):
            new = comfy.ops.manual_cast.Embedding(
                child.num_embeddings, 
                child.embedding_dim,
                padding_idx=child.padding_idx,
                device=torch.device("meta"), 
                dtype=child.weight.dtype,
            )
            new.weight = child.weight
            setattr(module, name, new)
        else:
            _swap_to_manual_cast(child)


class Qwen2CausalLMLoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("Qwen2CausalLM",)
    FUNCTION = "load"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }

    def load(self, ckpt_path: str):
        # ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", ckpt_path)
        config_path = osp.join(osp.dirname(__file__), '..', '..', 'config', 'qwen_2_causal_lm.json')
        
        llm_config = Qwen2Config.from_json_file(config_path)

        llm_config.layer_module = 'Qwen2MoTDecoderLayer'
        llm_config.qk_norm = True
        llm_config.qk_norm_und = True
        llm_config.qk_norm_gen = True
        llm_config.tie_word_embeddings = False
        llm_config.freeze_und = False
        llm_config.apply_qwen_2_5_vl_pos_emb = True

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

        try:
            language_model = Qwen2ForCausalLM(llm_config)
        finally:
            torch.set_default_dtype(default_dtype)

        _swap_to_manual_cast(language_model)

        # ckpt = comfy.utils.load_torch_file(ckpt_path)
        import pdb;pdb.set_trace()

        # return (patcher,)


a = Qwen2CausalLMLoader()
a.load('a')