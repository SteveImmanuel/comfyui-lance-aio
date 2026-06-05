import os
import os.path as osp

import comfy.utils
import torch
from comfy.sd import VAE
from ..constants import CKPT_ROOT_DIR, LATENT_FORMAT


class WANVAELoader:
    CATEGORY = "Lance"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        paths = [x for x in os.listdir(CKPT_ROOT_DIR) if x.endswith(".pth")]
        return {
            "required": {
                "ckpt_path": (paths, {"default": "Wan2.2_VAE.pth"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, ckpt_path):
        if not osp.isfile(osp.join(CKPT_ROOT_DIR, ckpt_path)):
            return f"{ckpt_path} not found in {CKPT_ROOT_DIR}"
        return True

    def load(self, ckpt_path):
        sd = comfy.utils.load_torch_file(osp.join(CKPT_ROOT_DIR, ckpt_path))
        vae = VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return (vae,)


class ComfyVAEAdapter:
    def __init__(self, vae: VAE):
        self.vae = vae

    @torch.no_grad()
    def vae_encode(self, samples: torch.Tensor, **kwargs):
        # follows modeling/vae/wan/model.py vae_encode for compatibility
        latents = []
        for x in samples:  # [C,T,H,W] in [-1,1]
            pixels = (x.movedim(0, -1).unsqueeze(0) + 1.0) / 2.0  # -> [1,T,H,W,C] in [0,1]
            z = self.vae.encode(pixels)  # [1,48,t,h,w] raw; comfy stages + tiles on OOM
            z = LATENT_FORMAT.process_in(z)  # (z - mean) / std, matching Lance's normalization
            latents.append(z.squeeze(0).movedim(0, -1))  # [t,h,w,48]
        return latents
