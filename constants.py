import os.path as osp

import comfy.latent_formats
import folder_paths

from .common.utils.misc import AutoEncoderParams

CKPT_ROOT_DIR = osp.join(folder_paths.models_dir, "lance")
TASK_T2V = "t2v"
TASK_T2I = "t2i"
TASK_I2V = "i2v"
TASK_X2T_IMAGE = "x2t_image"
TASK_X2T_VIDEO = "x2t_video"
TASK_IMAGE_EDIT = "image_edit"
TASK_VIDEO_EDIT = "video_edit"
ALL_TASKS = [
    TASK_I2V,
    TASK_T2V,
    TASK_T2I,
    TASK_X2T_IMAGE,
    TASK_X2T_VIDEO,
    TASK_IMAGE_EDIT,
    TASK_VIDEO_EDIT,
]
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
RESOLUTION_CONFIGS = ["video_192p", "video_360p", "video_480p", "image_256res", "image_512res", "image_768res"]

VAE_CONFIG = AutoEncoderParams(
    downsample_spatial=16,
    downsample_temporal=4,
    z_channels=48,
)
LATENT_FORMAT = comfy.latent_formats.Wan22()
