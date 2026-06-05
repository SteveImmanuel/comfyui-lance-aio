__author__ = """SteveImmanuel"""
__email__ = "steveandreasimanuel@gmail.com"
__version__ = "0.1.0"

from .nodes.config import (
    ApplyDefaultArgs,
    LanceDataConfig,
    LanceInferenceArgs,
    LanceModelArgs,
)
from .nodes.generate import LanceGenerate
from .nodes.lance import LanceConfigure, LanceLoader
from .nodes.prompt import LancePrompt
from .nodes.qwen_causal_lm import Qwen2CausalLMLoader
from .nodes.tokenizer import LanceTokenizerLoader
from .nodes.vae import WANVAELoader
from .nodes.vit import VitLoader

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
    "Lance Apply Default Args": ApplyDefaultArgs,
    "Lance Data Config": LanceDataConfig,
    "Lance Generate": LanceGenerate,
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
]
