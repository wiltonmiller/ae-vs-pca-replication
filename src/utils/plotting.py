from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from src.utils.io import ensure_dir

PathLike = Union[str, Path]

def _to_img(x: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a single image into 2D format for visualization.

    Supported inputs:
        - Flattened vector of shape (H*W,)
        - 2D array of shape (H, W)

    Args:
        x: Image array to convert.
        image_shape: Target (H, W) shape used when x is flattened.

    Returns:
        A 2D numpy array of shape (H, W) suitable for imshow.

    Raises:
        ValueError: If x cannot be interpreted as an image.
    """
    if x.ndim == 1:
        return x.reshape(image_shape)
    if x.ndim == 2 and x.shape == image_shape:
        return x
    raise ValueError(f"Unexpected image shape: {x.shape}")

def save_recon_grid(
    path: PathLike,
    X: np.ndarray,
    X_hat: np.ndarray,
    title: Optional[str] = None,
    n: int = 64,
    image_shape: Tuple[int, int] = (28, 28),
) -> None:
    """
    Save a reconstruction grid comparing original images and reconstructions.

    This is used in the report to qualitatively compare PCA vs AE.

    Layout:
        Samples are arranged in a grid with 2 rows per sample-row:
        - Even rows: original images
        - Odd rows: reconstructed images

    Args:
        path: Output path for the saved image file.
        X: Original images array of shape (N, 784) or (N, H, W).
        X_hat: Reconstructed images array of same shape as X.
        title: Optional figure title.
        n: Number of samples to include (default 64).
        image_shape: Shape (H, W) for reshaping flattened images.

    Returns:
        None

    Raises:
        ValueError: If X and X_hat do not have the same shape.
    """
    p = Path(path)
    ensure_dir(p.parent)

    if X.shape != X_hat.shape:
        raise ValueError(f"X and X_hat must have same shape, got {X.shape} vs {X_hat.shape}")

    k = min(n, X.shape[0])
    cols = 16 if k >= 16 else k
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(2 * rows, cols, figsize=(cols, 2 * rows))
    if title is not None:
        fig.suptitle(title)

    if isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape(2 * rows, cols)

    for idx in range(2 * rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.axis("off")

        sample_idx = (r // 2) * cols + c
        if sample_idx >= k:
            continue

        img = _to_img(X[sample_idx], image_shape) if r % 2 == 0 else _to_img(X_hat[sample_idx], image_shape)
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)

    plt.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)