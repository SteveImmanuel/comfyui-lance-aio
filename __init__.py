__author__ = """SteveImmanuel"""
__email__ = "steveandreasimanuel@gmail.com"
__version__ = "0.1.0"

from .nodes.config import LanceArgs
from .nodes.generate import LanceGeneration, LanceUnderstanding
from .nodes.lance import LanceConfigure, LanceLoader
from .nodes.prompt import LanceTextImagePrompt, LanceTextPrompt, LanceTextVideoPrompt
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
    "Lance Text Prompt": LanceTextPrompt,
    "Lance Text Image Prompt": LanceTextImagePrompt,
    "Lance Text Video Prompt": LanceTextVideoPrompt,
    "Lance Tokenizer Loader": LanceTokenizerLoader,
    "Lance Args": LanceArgs,
    "Lance Generation": LanceGeneration,
    "Lance Understanding": LanceUnderstanding,
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
]
