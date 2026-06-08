# ComfyUI Lance AIO
This repository implements ComfyUI custom nodes in order to run Lance-3B model using ComfyUI. 

It supports 7 different tasks:
- Text to Image
- Text to Video
- Image-Text to Video
- Image Editing
- Video Editing
- Image Understanding
- Video Understanding

This codebase is a faithful port to the official codebase, with additional support for machine with **limited VRAM**.

> [!NOTE]
> All tasks have been tested and runs well on Ubuntu machine with 12GB VRAM + 32GB RAM


> [!CAUTION]
> In theory, image-related tasks should run on at least 8GB VRAM, while video-related tasks may require at least 12GB VRAM. However, I have not confirmed the lower-bound requirement as I don't have the hardware to test.

## Installation
Clone this repository into ComfyUI's custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SteveImmanuel/comfyui-lance-aio
```
Install the dependencies. I highly recommend that ComfyUI is setup using virtual environment because it needs to co-exist without conflict with the custom node's dependencies.

```bash
cd comfyui-lance-aio
pip install -r requirements.txt
```

## Optional
Install `flash-attn` (recommended for speed). If it is not installed, there is fallback mechanism to use PyTorch SDPA, which produces equivalent results but runs slower. 

You can either build it from source following the [official instruction](https://github.com/dao-ailab/flash-attention#installation-and-features) or you can use pre-built wheel from third-party sources.

Following the official codebase instruction, for example with Python 3.13, CUDA 12, you can use:
```bash
pip install --no-cache-dir --no-deps --force-reinstall https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.8/cu12/abiTRUE/cp313/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl
```

### Tested Environment
- Python 3.13.13
- CUDA 12.8
- `torch==2.8.0`
- `torchvision==0.23.0`

The official Lance codebase is not compatible with `transformers>=5`, so you need to downgrade to `transformers>=4.50.3,<5` as specified in the `requirements.txt`.

## Model Checkpoint
Download Lance-3B checkpoint from hugginface and place them into ComfyUI `models` directory. Follow this structure convention:
```
📂 ComfyUI/
├── 📂 models/
    ├── 📂 lance/
│   │   ├── 📂 Lance_3B
│   │   │   ├── generation_config.json
│   │   │   ├── llm_config.json
│   │   │   ├── merges.txt
│   │   │   ├── model.safetensors
│   │   │   ├── tokenizer.json
│   │   │   └── vocab.json
│   │   ├── 📂 Lance_3B_Video
│   │   │   ├── generation_config.json
│   │   │   ├── llm_config.json
│   │   │   ├── merges.txt
│   │   │   ├── model.safetensors
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.json
│   │   ├── 📂 Qwen2.5-VL-ViT
│   │   │   ├── config.json
│   │   │   └── vit.safetensors
│   │   ├── Wan2.2_VAE.pth
```

## Usage
I have included workflow templates in [example_workflows](./workflows/), one for each of the task. 

They should show up in the ComfyUI template browser automatically. If they don't, simply drag and drop the `json` file into ComfyUI to open it.



## Acknowledgement
Lance model page: https://huggingface.co/bytedance-research/Lance

Lance official codebase: https://github.com/bytedance/Lance

AI was used in order to understand the structure of official codebase as well as ComfyUI API to build the custom nodes. However, I manually **wrote and reviewed** every single line of code in the implementation.

## License
The code under `modeling/`, `data/`, `common/`, and `config/` is adapted from the official codebase, Copyright (c) 2025 ByteDance Ltd. and/or its affiliates, licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). 

The rest of this repository is licensed under the [MIT License](./LICENSE).
