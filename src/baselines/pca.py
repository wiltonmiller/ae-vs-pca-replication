from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from src.data.mnist import get_mnist_numpy
from src.utils.io import save_json, save_npz
from src.utils.plotting import save_recon_grid

@dataclass(frozen=True)
class PCAResult:
    """
    Container for PCA experiment outputs.

    Attributes:
        method: Name of the method.
        latent_dim: Bottleneck dimension used for PCA.
        seed: Random seed used for PCA initialization.
        train_mse: Mean squared reconstruction error on the training set.
        test_mse: Mean squared reconstruction error on the test set.
    """
    method: str
    latent_dim: int
    seed: int
    train_mse: float
    test_mse: float


def fit_pca(X_train: np.ndarray, latent_dim: int, seed: int = 0) -> PCA:
    """
    Fit a PCA model on training data.

    Args:
        X_train: Training data array of shape (N, D).
        latent_dim: Number of principal components to keep.
        seed: Random seed for PCA.

    Returns:
        A fitted sklearn.decomposition.PCA instance.
    """
    pca = PCA(n_components=latent_dim, random_state=seed)
    pca.fit(X_train)
    return pca


def pca_reconstruct(pca: PCA, X: np.ndarray) -> np.ndarray:
    """
    Reconstruct input data using a fitted PCA model.

    Args:
        pca: Fitted PCA model.
        X: Data array of shape (N, D).

    Returns:
        Reconstructed array X_hat of shape (N, D), float32.
    """
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    return X_hat.astype(np.float32)


def reconstruction_mse(X: np.ndarray, X_hat: np.ndarray) -> float:
    """
    Compute mean squared reconstruction error.

    Args:
        X: Ground-truth inputs of shape (N, D).
        X_hat: Reconstructions of shape (N, D).

    Returns:
        Mean squared error as a float.
    """
    return float(np.mean((X - X_hat) ** 2))


def run_pca_experiment(
    latent_dim: int,
    data_dir: str = "data",
    results_dir: str = "results",
    seed: int = 0,
    n_vis: int = 64,
) -> Dict:
    """
    Run a PCA reconstruction experiment on MNIST and save artifacts.

    This function:
    1) Loads MNIST as flattened NumPy arrays.
    2) Fits PCA on X_train.
    3) Reconstructs X_train and X_test.
    4) Computes reconstruction MSE on train/test.
    5) Saves:
       - JSON metrics to results/metrics/pca_latent{d}.json
       - NPZ samples (X, X_hat, y) to results/reconstructions/pca_latent{d}_samples.npz
       - Recon grid image to results/plots/pca_latent{d}_grid.png

    Args:
        latent_dim: Number of PCA components (bottleneck dimension).
        data_dir: Directory containing MNIST download/cache.
        results_dir: Base directory for saving experiment artifacts.
        seed: Random seed for PCA.
        n_vis: Number of test samples to visualize in recon grid / NPZ.

    Returns:
        A dictionary version of PCAResult (suitable for JSON/CSV).
    """
    data = get_mnist_numpy(data_dir=data_dir, normalize=True, flatten=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    pca = fit_pca(X_train=X_train, latent_dim=latent_dim, seed=seed)

    X_train_hat = pca_reconstruct(pca, X_train)
    X_test_hat = pca_reconstruct(pca, X_test)

    result = PCAResult(
        method="pca",
        latent_dim=int(latent_dim),
        seed=int(seed),
        train_mse=reconstruction_mse(X_train, X_train_hat),
        test_mse=reconstruction_mse(X_test, X_test_hat),
    )

    metrics_path = Path(results_dir) / "metrics" / f"pca_latent{latent_dim}.json"
    save_json(metrics_path, asdict(result))

    # Save a deterministic slice of samples for later comparisons/figures.
    k = min(n_vis, X_test.shape[0])
    samples_path = Path(results_dir) / "reconstructions" / f"pca_latent{latent_dim}_samples.npz"
    save_npz(samples_path, X=X_test[:k], X_hat=X_test_hat[:k], y=y_test[:k])

    grid_path = Path(results_dir) / "plots" / f"pca_latent{latent_dim}_grid.png"
    save_recon_grid(
        grid_path,
        X=X_test[:k],
        X_hat=X_test_hat[:k],
        title=f"PCA reconstructions (latent_dim={latent_dim})",
        n=k,
        image_shape=(28, 28),
    )

    return asdict(result)
