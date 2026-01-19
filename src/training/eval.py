from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.io import save_json, save_npz
from src.utils.plotting import save_recon_grid


@dataclass(frozen=True)
class EvalResult:
    """
    Container for evaluation outputs.

    Attributes:
        method: Name of method (e.g., "pca" or "ae").
        latent_dim: Bottleneck dimension used.
        seed: Seed used to select visualization subset.
        test_mse: Mean squared reconstruction error on the test set.
        metrics_path: Path to metrics JSON file.
        samples_path: Path to NPZ sample file.
        grid_path: Path to reconstruction grid image.
    """
    method: str
    latent_dim: int
    seed: int
    test_mse: float
    metrics_path: str
    samples_path: str
    grid_path: str


def reconstruction_mse(X: np.ndarray, X_hat: np.ndarray) -> float:
    """
    Compute mean squared reconstruction error.

    Args:
        X: Ground truth inputs of shape (N, D).
        X_hat: Reconstructions of shape (N, D).

    Returns:
        Scalar mean squared error.
    """
    return float(np.mean((X - X_hat) ** 2))


def select_fixed_subset(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a deterministic subset of examples for visualization.

    This ensures that recon grids are comparable across methods and latent dims.

    Args:
        X: Input array of shape (N, D).
        y: Labels array of shape (N,).
        n: Number of examples to select.
        seed: Random seed used for subset selection.

    Returns:
        (X_sub, y_sub) where:
            X_sub has shape (n, D)
            y_sub has shape (n,)
    """
    rng = np.random.default_rng(seed)
    n = min(n, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


def eval_autoencoder_reconstructions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run an autoencoder over a DataLoader and return inputs and reconstructions.

    Args:
        model: Autoencoder model mapping x -> x_hat.
        loader: DataLoader yielding (x, y).
        device: Torch device to run inference on.

    Returns:
        Tuple (X, X_hat, y) where:
            X: numpy float32 array of shape (N, D)
            X_hat: numpy float32 array of shape (N, D)
            y: numpy int array of shape (N,)
    """
    model.eval()

    xs = []
    xhats = []
    ys = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_hat = model(x)

            xs.append(x.detach().cpu().numpy())
            xhats.append(x_hat.detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())

    X = np.concatenate(xs, axis=0).astype(np.float32)
    X_hat = np.concatenate(xhats, axis=0).astype(np.float32)
    y_all = np.concatenate(ys, axis=0)

    return X, X_hat, y_all


def save_recon_artifacts(
    method: str,
    latent_dim: int,
    X_test: np.ndarray,
    X_test_hat: np.ndarray,
    y_test: np.ndarray,
    results_dir: str = "results",
    subset_seed: int = 0,
    n_vis: int = 64,
) -> Dict:
    """
    Save standardized reconstruction artifacts (metrics + samples + grid).

    Artifacts:
        - results/metrics/{method}_latent{d}.json
        - results/reconstructions/{method}_latent{d}_samples.npz
        - results/plots/{method}_latent{d}_grid.png

    Args:
        method: Method name ("ae" or "pca").
        latent_dim: Bottleneck dimension.
        X_test: Test set inputs of shape (N, D).
        X_test_hat: Reconstructed test inputs of shape (N, D).
        y_test: Test labels of shape (N,).
        results_dir: Base directory for saving artifacts.
        subset_seed: Seed controlling which samples are visualized/saved.
        n_vis: Number of samples to visualize/save.

    Returns:
        Dictionary version of EvalResult, suitable for JSON/CSV.
    """
    test_mse = reconstruction_mse(X_test, X_test_hat)

    # Select consistent subset for saving recon samples and grid
    X_sub, y_sub = select_fixed_subset(X_test, y_test, n=n_vis, seed=subset_seed)
    # Need corresponding recon subset: match indices via re-selection
    rng = np.random.default_rng(subset_seed)
    idx = rng.choice(X_test.shape[0], size=min(n_vis, X_test.shape[0]), replace=False)
    X_hat_sub = X_test_hat[idx]

    metrics_path = Path(results_dir) / "metrics" / f"{method}_latent{latent_dim}.json"
    samples_path = Path(results_dir) / "reconstructions" / f"{method}_latent{latent_dim}_samples.npz"
    grid_path = Path(results_dir) / "plots" / f"{method}_latent{latent_dim}_grid.png"

    save_json(
        metrics_path,
        {
            "method": method,
            "latent_dim": int(latent_dim),
            "subset_seed": int(subset_seed),
            "test_mse": float(test_mse),
        },
    )

    save_npz(samples_path, X=X_sub, X_hat=X_hat_sub, y=y_sub)

    save_recon_grid(
        grid_path,
        X=X_sub,
        X_hat=X_hat_sub,
        title=f"{method.upper()} reconstructions (latent_dim={latent_dim})",
        n=min(n_vis, X_sub.shape[0]),
        image_shape=(28, 28),
    )

    result = EvalResult(
        method=method,
        latent_dim=int(latent_dim),
        seed=int(subset_seed),
        test_mse=float(test_mse),
        metrics_path=str(metrics_path),
        samples_path=str(samples_path),
        grid_path=str(grid_path),
    )

    return asdict(result)