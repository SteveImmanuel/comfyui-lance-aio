__author__ = """SteveImmanuel"""
__email__ = "steveandreasimanuel@gmail.com"
__version__ = "0.1.0"

from .nodes.vit import VitLoader
from .nodes.qwen_causal_lm import Qwen2CausalLMLoader
from .nodes.lance import LanceLoader
from .nodes.text_encode import TextEncoder
from .nodes.config import (
    LanceVaeConfig,
    LanceModelArgs,
    LanceInferenceArgs,
    LanceDataArgs,
    ApplyDefaultArgs,
    LanceDataConfig,
)

NODE_CLASS_MAPPINGS = {
    "ViT Loader": VitLoader,
    "Qwen 2 Causal LM Loader": Qwen2CausalLMLoader,
    "Lance Loader": LanceLoader,
    "Text Encoder": TextEncoder,
    "Lance VAE Config": LanceVaeConfig,
    "Lance Model Args": LanceModelArgs,
    "Lance Inference Args": LanceInferenceArgs,
    "Lance Data Args": LanceDataArgs,
    "Lance Apply Default Args": ApplyDefaultArgs,
    "Lance Data Config": LanceDataConfig,
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    # "NODE_DISPLAY_NAME_MAPPINGS",
]