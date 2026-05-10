from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def parse_ks(text: str) -> tuple[int, ...]:
    ks = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not ks:
        raise ValueError("expected at least one k")
    if any(k <= 0 for k in ks):
        raise ValueError("k values must be positive")
    return ks


def load_lm_tokens(fpath: Path) -> np.ndarray:
    """Load a saved LM generation tensor as final-layer token features."""
    tensor = torch.load(fpath, map_location="cpu").float()
    if tensor.ndim != 2:
        raise ValueError(f"expected LM token tensor (T, D), got {tuple(tensor.shape)} at {fpath}")
    return tensor.detach().cpu().numpy().astype(np.float32)


def load_target_vector(fpath: Path) -> np.ndarray:
    """Load a saved reference vector."""
    tensor = torch.load(fpath, map_location="cpu").float()
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(
            f"expected target vector (D,) or (1, D), got {tuple(tensor.shape)} at {fpath}"
        )
    return tensor.detach().cpu().numpy().astype(np.float32)
