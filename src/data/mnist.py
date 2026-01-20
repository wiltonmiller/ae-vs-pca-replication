from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class MNISTConfig:
    """
    Configuration container for MNIST loading.

    Attributes:
        data_dir: Directory where MNIST will be downloaded/stored.
        batch_size: Batch size for DataLoader usage.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory.
        normalize: If True, use ToTensor() scaling pixels to [0,1].
        flatten: If True, flatten each image to a length-784 vector.
    """
    data_dir: str = "data"
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = True
    normalize: bool = True
    flatten: bool = True


class _FlattenTransform:
    """
    Torchvision style transform that flattens a single MNIST image tensor.

    Input shape:
        (1, 28, 28)

    Output shape:
        (784,)
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


def _build_transform(normalize: bool, flatten: bool) -> transforms.Compose:
    """
    Build the torchvision transform pipeline for MNIST.

    Args:
        normalize: If True, use ToTensor() to convert to float32 in [0,1].
            If False, use PILToTensor() which yields integer-like tensors.
        flatten: If True, append a flattening transform to produce (784,).

    Returns:
        A torchvision.transforms.Compose object.
    """
    tfms = []

    if normalize:
        tfms.append(transforms.ToTensor())
    else:
        tfms.append(transforms.PILToTensor())

    if flatten:
        tfms.append(_FlattenTransform())

    return transforms.Compose(tfms)


def get_mnist_dataloaders(
    batch_size: int,
    data_dir: str,
    normalize: bool = True,
    flatten: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for the MNIST dataset.

    Args:
        batch_size: Number of samples per batch.
        data_dir: Directory where MNIST is stored/downloaded.
        normalize: If True, pixel values are scaled to [0,1] via ToTensor().
        flatten: If True, each image is flattened to shape (784,).
        num_workers: Number of subprocesses for data loading.
        pin_memory: Whether to pin memory for faster GPU transfers.

    Returns:
        A dict with:
            "train": DataLoader for the training set
            "test": DataLoader for the test set

        Each batch yields:
            x: Tensor of shape (B, 784) if flatten=True else (B, 1, 28, 28)
            y: Tensor of shape (B,)
    """
    transform = _build_transform(normalize=normalize, flatten=flatten)

    train_ds = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "test": test_loader}


def get_mnist_numpy(
    data_dir: str,
    normalize: bool = True,
    flatten: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load the MNIST dataset fully into NumPy arrays.

    Args:
        data_dir: Directory where MNIST is stored/downloaded.
        normalize: If True, pixel values are scaled to [0,1] via ToTensor().
        flatten: If True, each image is flattened to shape (784,).

    Returns:
        A dict with:
            X_train: float32 array of shape (60000, 784) if flatten=True
            y_train: int array of shape (60000,)
            X_test: float32 array of shape (10000, 784) if flatten=True
            y_test: int array of shape (10000,)
    """
    transform = _build_transform(normalize=normalize, flatten=flatten)

    train_ds = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))], dim=0)
    y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))], dtype=torch.long)

    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))], dim=0)
    y_test = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)

    X_train_np = X_train.numpy().astype(np.float32)
    X_test_np = X_test.numpy().astype(np.float32)

    return {
        "X_train": X_train_np,
        "y_train": y_train.numpy(),
        "X_test": X_test_np,
        "y_test": y_test.numpy(),
    }
