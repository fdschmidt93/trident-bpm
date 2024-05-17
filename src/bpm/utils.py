def get_torch_dtype(type_: str):
    import torch

    return getattr(torch, type_)
