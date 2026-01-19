from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs for more reproducible experiments.

    Args:
        seed: Integer random seed.
        deterministic: If True and torch is installed, configure CuDNN for
            deterministic behavior (may reduce speed).

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False