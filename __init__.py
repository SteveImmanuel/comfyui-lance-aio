__author__ = """SteveImmanuel"""
__email__ = "steveandreasimanuel@gmail.com"
__version__ = "0.1.0"

from .src.vit import VitLoader

NODE_CLASS_MAPPINGS = {
    "ViT Loader": VitLoader,
}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    # "NODE_DISPLAY_NAME_MAPPINGS",
]