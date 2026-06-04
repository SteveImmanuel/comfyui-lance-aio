import folder_paths
import torch
import comfy.model_management as mm
from copy import deepcopy
from einops import rearrange
from comfy.model_patcher import CoreModelPatcher

from ..modeling.vae.wan.model import WanVideoVAE, reparameterize
from ..modeling.vae.wan.vae2_2 import Wan2_2_VAE
import folder_paths
import os
import os.path as osp
from ..constants import CKPT_ROOT_DIR

class _WrapWanVideoVAE(WanVideoVAE):
    # WanVideoVAE resolves the weight path via get_model_path; override to inject our own.
    def __init__(self, vae_pth, **kwargs):
        self._vae_pth = vae_pth
        super().__init__(**kwargs)

    def configure_vae_model(self):
        self.vae = Wan2_2_VAE(vae_pth=self._vae_pth, device=self.device, dtype=self.dtype)

    # follow wherever the patcher placed the module; scale is out-of-band, so align it per call.
    def _model_device(self):
        return next(self.vae.model.parameters()).device

    @torch.no_grad()
    def vae_decode(self, latents, **kwargs):
        device = self._model_device()
        self.vae.scale = [s.to(device) for s in self.vae.scale]
        samples = []
        with torch.autocast(device_type=device.type, dtype=self.dtype):
            for u in latents:
                u = u.unsqueeze(0).to(device=device)
                u = rearrange(u, "b ... c -> b c ...")
                x_hat = self.vae.decode(u)
                samples.append(x_hat.squeeze(0))
        return samples

    @torch.no_grad()
    def vae_encode(self, samples, **kwargs):
        device = self._model_device()
        self.vae.scale = [s.to(device) for s in self.vae.scale]
        latents = []
        with torch.autocast(device_type=device.type, dtype=self.dtype):
            for x in samples:
                x = x.to(device=device).unsqueeze(0)
                u, log_var = self.vae.encode(x)
                if self.use_sample:
                    u = reparameterize(u, log_var)
                u = rearrange(u, "b c ... -> b ... c")
                latents.append(u.squeeze(0))
        return latents


class WANVAELoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("WAN_VAE", "VAE_CONFIG")
    FUNCTION = "load"
    # OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        paths = [x for x in os.listdir(CKPT_ROOT_DIR) if x.endswith('.pth')]
        return {
            "required": {
                "ckpt_path": (paths, {'default': 'Wan2.2_VAE.pth'}),
            }
        }

    def load(self, ckpt_path):

        ckpt_path = osp.join(CKPT_ROOT_DIR, ckpt_path)
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        wrapper = _WrapWanVideoVAE(vae_pth=ckpt_path, device=offload_device, dtype=torch.bfloat16)
        patcher = CoreModelPatcher(wrapper.vae.model, load_device=load_device, offload_device=offload_device)
        patcher.set_model_compute_dtype(torch.bfloat16)

        return ({"patcher": patcher, "module": wrapper}, deepcopy(wrapper.vae_config))
