__author__ = """SteveImmanuel"""
__email__ = "steveandreasimanuel@gmail.com"
__version__ = "0.1.0"

from .nodes.vit import VitLoader
from .nodes.qwen_causal_lm import Qwen2CausalLMLoader
from .nodes.lance import LanceLoader, LanceConfigure
from .nodes.prompt import LancePrompt
from .nodes.tokenizer import LanceTokenizerLoader
from .nodes.vae import WANVAELoader
from .nodes.config import (
    LanceModelArgs,
    LanceInferenceArgs,
    LanceDataArgs,
    ApplyDefaultArgs,
    LanceDataConfig,
)

NODE_CLASS_MAPPINGS = {
    "ViT Loader": VitLoader,
    "Qwen 2 Causal LM Loader": Qwen2CausalLMLoader,
    "WAN VAE Loader": WANVAELoader,
    "Lance Loader": LanceLoader,
    "Lance Configure": LanceConfigure,
    "Lance Prompt": LancePrompt,
    "Lance Tokenizer Loader": LanceTokenizerLoader,
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