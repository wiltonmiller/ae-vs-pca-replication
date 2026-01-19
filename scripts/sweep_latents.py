"""
Run a PCA vs MLP Autoencoder sweep over latent dimensions and save a summary table.

This is the main experiment runner for the replication:
- PCA baseline: sklearn PCA reconstruction MSE on MNIST
- Autoencoder: PyTorch MLP autoencoder trained with MSE loss
- Latent dimensions: user-specified (default: 2 16 32 64)

Artifacts produced:
- PCA artifacts are written by src.baselines.pca.run_pca_experiment
- AE training artifacts (checkpoint, history) are written by src.training.train_ae.train_autoencoder
- AE eval artifacts (metrics, samples, grid) are written by src.training.eval.save_recon_artifacts
- A consolidated CSV summary is saved to results/metrics/summary.csv (or user-specified path)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.baselines.pca import run_pca_experiment
from src.data.mnist import get_mnist_dataloaders
from src.models.ae_mlp import build_mlp_autoencoder
from src.training.eval import eval_autoencoder_reconstructions, save_recon_artifacts
from src.training.train_ae import load_checkpoint, train_autoencoder
from src.utils.seed import set_seed
from src.utils.io import save_csv


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the sweep script.

    Returns:
        argparse.Namespace containing parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Sweep PCA vs AE across latent dimensions on MNIST.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--summary-csv", type=str, default="results/metrics/summary.csv")
    parser.add_argument("--latents", type=int, nargs="+", default=[2, 16, 32, 64])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-vis", type=int, default=64)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def get_device(no_cuda: bool) -> torch.device:
    """
    Determine the torch device to use.
    """
    if no_cuda:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_latent(latent_dim: int, args: argparse.Namespace, device: torch.device) -> List[Dict]:
    """
    Run PCA and AE experiments for a single latent dimension.
    """
    rows: List[Dict] = []

    # PCA
    pca_metrics = run_pca_experiment(
        latent_dim=latent_dim,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        seed=args.seed,
        n_vis=args.n_vis,
    )

    rows.append({
        "method": "pca",
        "latent_dim": latent_dim,
        "seed": args.seed,
        "epochs": None,
        "train_mse": pca_metrics["train_mse"],
        "test_mse": pca_metrics["test_mse"],
    })

    # AE
    loaders = get_mnist_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        normalize=True,
        flatten=True,
        pin_memory=(device.type == "cuda"),
    )

    model = build_mlp_autoencoder(latent_dim).to(device)

    train_res = train_autoencoder(
        model=model,
        latent_dim=latent_dim,
        loaders=loaders,
        device=device,
        out_dir=args.results_dir,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=100,
    )

    load_checkpoint(train_res.checkpoint_path, model, optimizer=None, device=device)

    X, X_hat, y = eval_autoencoder_reconstructions(model, loaders["test"], device)

    eval_res = save_recon_artifacts(
        method="ae",
        latent_dim=latent_dim,
        X_test=X,
        X_test_hat=X_hat,
        y_test=y,
        results_dir=args.results_dir,
        subset_seed=args.subset_seed,
        n_vis=args.n_vis,
    )

    rows.append({
        "method": "ae",
        "latent_dim": latent_dim,
        "seed": args.seed,
        "epochs": args.epochs,
        "test_mse": eval_res["test_mse"],
        "best_epoch": train_res.best_epoch,
        "best_test_mse": train_res.best_test_mse,
        "final_test_mse": train_res.final_test_mse,
    })

    return rows


def main() -> None:
    """
    Entry pointfor the sweep script.
    """
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = get_device(args.no_cuda)

    all_rows: List[Dict] = []

    for d in args.latents:
        print(f"\n=== latent_dim={d} ===")
        all_rows.extend(run_one_latent(d, args, device))

    df = pd.DataFrame(all_rows)
    out_path = Path(args.summary_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSummary written to {out_path}")
    print(df)


if __name__ == "__main__":
    main()