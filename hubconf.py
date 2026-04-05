"""
PyTorch Hub entrypoints for loading OpenAI CLIP checkpoints.

Function names in ``globals()`` are sanitized model identifiers required by
``torch.hub``; only internal helpers follow manuscript-style naming here.
"""

import re
import string

from third_party_clip.clip import available_models as _available_clip_models
from third_party_clip.clip import load as _load_clip_checkpoint
from third_party_clip.clip import tokenize as _clip_tokenize

dependencies = ["torch", "torchvision", "ftfy", "regex", "tqdm"]

_HUB_SAFE_MODEL_NAMES = {
    name: re.sub(f"[{string.punctuation}]", "_", name) for name in _available_clip_models()
}


def _build_clip_hub_entrypoint(model_name: str):
    """
    Factory: return a ``torch.hub``-compatible loader for ``model_name``.
    """

    def load_clip_with_hub_kwargs(**kwargs):
        return _load_clip_checkpoint(model_name, **kwargs)

    load_clip_with_hub_kwargs.__doc__ = f"""Load the {model_name} CLIP model.

        Parameters
        ----------
        device : str or torch.device
            Device for loaded weights.
        jit : bool
            If True, load the TorchScript/JIT build if available.
        download_root : str
            Cache directory (default: ~/.cache/clip).

        Returns
        -------
        model : torch.nn.Module
        preprocess : Callable[[PIL.Image.Image], torch.Tensor]
        """
    return load_clip_with_hub_kwargs


def tokenize():
    """PyTorch Hub legacy accessor; same behavior as the original ``hubconf``."""
    return _clip_tokenize


_CLIP_HUB_ENTRYPOINTS = {
    _HUB_SAFE_MODEL_NAMES[name]: _build_clip_hub_entrypoint(name)
    for name in _available_clip_models()
}

globals().update(_CLIP_HUB_ENTRYPOINTS)
