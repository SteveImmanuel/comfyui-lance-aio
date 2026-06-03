import torch
import comfy.ops

def swap_to_manual_cast(module: torch.nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            new = comfy.ops.manual_cast.Linear(
                child.in_features, 
                child.out_features,
                bias=child.bias is not None,
                device=torch.device("meta"), 
                dtype=child.weight.dtype,
            )
            new.weight = child.weight
            if child.bias is not None:
                new.bias = child.bias
            setattr(module, name, new)
        elif isinstance(child, torch.nn.Embedding):
            new = comfy.ops.manual_cast.Embedding(
                child.num_embeddings, 
                child.embedding_dim,
                padding_idx=child.padding_idx,
                device=torch.device("meta"), 
                dtype=child.weight.dtype,
            )
            new.weight = child.weight
            setattr(module, name, new)
        else:
            swap_to_manual_cast(child)

