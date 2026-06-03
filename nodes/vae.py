import folder_paths
import torch
import comfy.model_management as mm
from copy import deepcopy
from comfy.model_patcher import CoreModelPatcher

from ..modeling.vae.wan.model import WanVideoVAE
from ..modeling.vae.wan.vae2_2 import Wan2_2_VAE


class _WrapWanVideoVAE(WanVideoVAE):
    # WanVideoVAE resolves the weight path via get_model_path; override to inject our own.
    def __init__(self, vae_pth, **kwargs):
        self._vae_pth = vae_pth
        super().__init__(**kwargs)

    def configure_vae_model(self):
        self.vae = Wan2_2_VAE(vae_pth=self._vae_pth, device=self.device, dtype=self.dtype)


class WANVAELoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("WAN_VAE", "VAE_CONFIG")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_path": (folder_paths.get_filename_list("vae"),),
            }
        }

    def load(self, vae_path):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_path)
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        wrapper = _WrapWanVideoVAE(vae_pth=vae_path, device=offload_device, dtype=torch.bfloat16)
        patcher = CoreModelPatcher(wrapper.vae.model, load_device=load_device, offload_device=offload_device)
        patcher.set_model_compute_dtype(torch.bfloat16)

        return ({"patcher": patcher, "module": wrapper}, deepcopy(wrapper.vae_config))
