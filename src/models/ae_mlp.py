from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AEConfig:
    """
    Configuration container for the MLP autoencoder architecture.

    Attributes:
        input_dim: Dimensionality of the input vector (784 for flattened MNIST).
        latent_dim: Bottleneck dimension used for compression.
        hidden_dims: Hidden layer widths for encoder/decoder (symmetric).
        activation: Activation function name for hidden layers.
        use_sigmoid_output: Whether to apply sigmoid at output to constrain to [0,1].
    """
    input_dim: int = 784
    latent_dim: int = 32
    hidden_dims: List[int] = None
    activation: str = "relu"
    use_sigmoid_output: bool = True


def _make_mlp(
    layer_dims: List[int],
    activation: str = "relu",
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """
    Build an MLP as an nn.Sequential from a list of layer dimensions.

    Example:
        layer_dims = [784, 512, 256, 32]
        produces: Linear(784->512), ReLU, Linear(512->256), ReLU, Linear(256->32)

    Args:
        layer_dims: List of dimensions specifying consecutive Linear layers.
        activation: Hidden activation function name ("relu" supported).
        final_activation: Optional module to apply after the last Linear layer.

    Returns:
        nn.Sequential implementing the MLP.

    Raises:
        ValueError: If an unsupported activation is requested.
    """
    if activation.lower() != "relu":
        raise ValueError(f"Unsupported activation: {activation}. Only 'relu' is supported.")

    layers: List[nn.Module] = []
    for i in range(len(layer_dims) - 1):
        in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))

        # Add activation after every layer except the last (unless final_activation handles it)
        is_last = i == len(layer_dims) - 2
        if not is_last:
            layers.append(nn.ReLU())

    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    """
    A simple MLP autoencoder for flattened MNIST.

    Architecture:
        Encoder: 784 -> 512 -> 256 -> latent_dim
        Decoder: latent_dim -> 256 -> 512 -> 784

    Input:
        x of shape (B, 784)

    Output:
        x_hat of shape (B, 784)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        use_sigmoid_output: bool = True,
    ) -> None:
        """
        Initialize the MLP autoencoder.

        Args:
            input_dim: Input dimensionality (784 for flattened MNIST).
            latent_dim: Bottleneck size.
            hidden_dims: Hidden layer widths used symmetrically for encoder/decoder.
                Example: [512, 256]
            activation: Activation function for hidden layers ("relu" supported).
            use_sigmoid_output: If True, apply sigmoid to output (recommended for [0,1] pixels).

        Returns:
            None
        """
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one hidden layer width.")

        # Encoder dims: [input_dim] + hidden_dims + [latent_dim]
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = _make_mlp(enc_dims, activation=activation, final_activation=None)

        # Decoder dims: [latent_dim] + reversed(hidden_dims) + [input_dim]
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        out_act = nn.Sigmoid() if use_sigmoid_output else None
        self.decoder = _make_mlp(dec_dims, activation=activation, final_activation=out_act)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode inputs into latent vectors.

        Args:
            x: Input tensor of shape (B, input_dim).

        Returns:
            Latent tensor z of shape (B, latent_dim).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors back into input space.

        Args:
            z: Latent tensor of shape (B, latent_dim).

        Returns:
            Reconstructed tensor x_hat of shape (B, input_dim).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder.

        Args:
            x: Input tensor of shape (B, input_dim).

        Returns:
            Reconstruction x_hat of shape (B, input_dim).
        """
        z = self.encode(x)
        return self.decode(z)


def build_mlp_autoencoder(latent_dim: int, input_dim: int = 784) -> MLPAutoencoder:
    """
    Convenience constructor for the default MNIST MLP autoencoder.

    Args:
        latent_dim: Bottleneck dimension.
        input_dim: Input dimensionality (default 784).

    Returns:
        MLPAutoencoder instance with the standard hidden sizes [512, 256].
    """
    return MLPAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256],
        activation="relu",
        use_sigmoid_output=True,
    )
