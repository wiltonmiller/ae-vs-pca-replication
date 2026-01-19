from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.io import ensure_dir, save_json


@dataclass(frozen=True)
class TrainConfig:
    """
    Configuration container for autoencoder training.

    Attributes:
        epochs: Number of full passes over the training set.
        lr: Learning rate for Adam.
        weight_decay: L2 regularization strength for Adam (0 disables).
        log_every: How often (in steps) to log training loss.
    """
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 100


@dataclass
class TrainResult:
    """
    Container for training outputs.

    Attributes:
        method: Name of method ("ae").
        latent_dim: Bottleneck dimension used for the AE.
        seed: Random seed used for training.
        epochs: Number of epochs actually run.
        best_epoch: Epoch index (1-based) achieving best test MSE.
        best_test_mse: Best observed test reconstruction MSE.
        final_test_mse: Test MSE at the final epoch.
        checkpoint_path: Path to the saved best checkpoint.
        history_path: Path to the saved training history JSON.
    """
    method: str
    latent_dim: int
    seed: int
    epochs: int
    best_epoch: int
    best_test_mse: float
    final_test_mse: float
    checkpoint_path: str
    history_path: str


def evaluate_recon_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """
    Compute reconstruction MSE over a DataLoader.

    Args:
        model: Autoencoder model mapping x -> x_hat.
        loader: DataLoader yielding (x, y) where x has shape (B, 784).
        device: Torch device for computation.
        criterion: Loss function (typically nn.MSELoss()).

    Returns:
        Mean loss over all batches (float).
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return float(total_loss / max(total_examples, 1))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_test_mse: float,
) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        path: Output file path.
        model: Model to save.
        optimizer: Optimizer to save.
        epoch: Current epoch (1-based).
        best_test_mse: Best test MSE observed so far.

    Returns:
        None
    """
    p = Path(path)
    ensure_dir(p.parent)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_test_mse": best_test_mse,
        },
        p,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict:
    """
    Load a training checkpoint from disk.

    Args:
        path: Checkpoint path.
        model: Model instance to populate.
        optimizer: Optimizer instance to populate. If None, optimizer is not loaded.
        device: Device mapping for torch.load.

    Returns:
        A dict containing checkpoint metadata (epoch, best_test_mse).
    """
    ckpt = torch.load(Path(path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return {
        "epoch": ckpt.get("epoch"),
        "best_test_mse": ckpt.get("best_test_mse"),
    }


def train_autoencoder(
    model: nn.Module,
    latent_dim: int,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    out_dir: str,
    seed: int,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    log_every: int = 100,
) -> TrainResult:
    """
    Train an autoencoder on MNIST and save the best checkpoint by test MSE.

    This function assumes loaders contains:
        - loaders["train"]
        - loaders["test"]

    Args:
        model: Autoencoder model mapping x -> x_hat.
        latent_dim: Bottleneck dimension used (for logging/artifacts).
        loaders: Dict containing "train" and "test" DataLoaders.
        device: Torch device (cpu or cuda).
        out_dir: Base output directory for checkpoints and histories.
        seed: Random seed for this run (logged).
        epochs: Number of epochs to train.
        lr: Learning rate for Adam.
        weight_decay: Weight decay for Adam.
        log_every: Log frequency in training steps.

    Returns:
        TrainResult containing best checkpoint path and summary metrics.
    """
    model = model.to(device)

    train_loader = loaders["train"]
    test_loader = loaders["test"]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_dir = Path(out_dir) / "ae" / f"latent{latent_dim}" / f"seed{seed}"
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    history_path = run_dir / "history.json"

    ensure_dir(run_dir / "checkpoints")

    history = {
        "method": "ae",
        "latent_dim": int(latent_dim),
        "seed": int(seed),
        "epochs": int(epochs),
        "train_mse": [],
        "test_mse": [],
    }

    best_test = float("inf")
    best_epoch = 0
    final_test = float("inf")

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, _ in train_loader:
            global_step += 1
            x = x.to(device)

            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

            if log_every > 0 and (global_step % log_every == 0):
                avg = running_loss / max(seen, 1)
                print(f"[epoch {epoch:02d} step {global_step:05d}] train_mse={avg:.6f}")

        train_mse_epoch = float(running_loss / max(seen, 1))
        test_mse_epoch = evaluate_recon_mse(model, test_loader, device, criterion)

        history["train_mse"].append(train_mse_epoch)
        history["test_mse"].append(test_mse_epoch)

        print(f"[epoch {epoch:02d}] train_mse={train_mse_epoch:.6f} test_mse={test_mse_epoch:.6f}")

        final_test = test_mse_epoch

        # save best checkpoint by test MSE
        if test_mse_epoch < best_test:
            best_test = test_mse_epoch
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, best_test_mse=best_test)

    save_json(history_path, history)

    return TrainResult(
        method="ae",
        latent_dim=int(latent_dim),
        seed=int(seed),
        epochs=int(epochs),
        best_epoch=int(best_epoch),
        best_test_mse=float(best_test),
        final_test_mse=float(final_test),
        checkpoint_path=str(ckpt_path),
        history_path=str(history_path),
    )
