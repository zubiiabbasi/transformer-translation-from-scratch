import torch


def load_training_checkpoint(path, map_location):
    """Load full training checkpoint (state dict + optimizer + metadata)."""
    kwargs = {"map_location": map_location}
    try:
        return torch.load(path, **kwargs, weights_only=False)
    except TypeError:
        return torch.load(path, **kwargs)
