import os
import os.path as osp

import comfy.latent_formats
import comfy.sd
import comfy.utils
import torch

from ..constants import CKPT_ROOT_DIR


class WANVAELoader:
    # stock VAELoader minus the models/vae dir constraint; arch is sniffed from the state dict
    CATEGORY = "Lance"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        paths = [x for x in os.listdir(CKPT_ROOT_DIR) if x.endswith(".pth")] if osp.isdir(CKPT_ROOT_DIR) else []
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
        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()
        return (vae,)


class ComfyVAEAdapter:
    """Duck-types Lance's WanVideoVAE.vae_encode on top of a comfy VAE for make_padded_latent."""

    def __init__(self, vae):
        self.vae = vae
        self.latent_format = comfy.latent_formats.Wan22()

    @torch.no_grad()
    def vae_encode(self, samples, **kwargs):
        latents = []
        for x in samples:  # [C,T,H,W] in [-1,1]
            pixels = (x.movedim(0, -1).unsqueeze(0) + 1.0) / 2.0  # -> [1,T,H,W,C] in [0,1]
            z = self.vae.encode(pixels)  # [1,48,t,h,w] raw; comfy stages + tiles on OOM
            z = self.latent_format.process_in(z)  # (z - mean) / std, matching Lance's normalization
            latents.append(z.squeeze(0).movedim(0, -1))  # [t,h,w,48]
        return latents
