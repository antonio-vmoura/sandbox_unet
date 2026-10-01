"""PyTorch port of the original Keras U-Net (``get_unet_baseline`` / ``get_unet``).

The port reproduces the Keras model layer by layer so that results remain
attributable to the same architecture:

* **Blocks.** ``conv2d_block``: (Conv 3×3 "same" → BatchNorm → activation) × 2.
  Encoder: 4 blocks of ``n_filters × {1, 2, 4, 8}``, each followed by
  MaxPool 2×2 and Dropout; bottleneck ``16 × n_filters``. Decoder: transposed
  conv 3×3 stride 2 "same" → concatenate with the skip → Dropout → block.
  Head: Conv 1×1 + sigmoid (here returned as **logits**; apply
  :func:`torch.sigmoid` or use :meth:`UNet.predict_proba`).
* **Transposed-conv padding.** Keras/TF ``Conv2DTranspose(padding="same")``
  with kernel 3 and stride 2 equals the un-padded PyTorch transposed
  convolution (output ``2n + 1``) **cropped to its first** ``2n`` rows and
  columns. The common PyTorch idiom ``padding=1, output_padding=1`` keeps the
  *last* ``2n`` instead and is shifted by one pixel; it is therefore not used.
* **BatchNorm.** Keras ``momentum=0.99`` (weight of the *old* running value) is
  PyTorch ``momentum=0.01``; ``eps=1e-3`` as in Keras.
* **Initialisation.** Keras ``he_normal`` (truncated normal at ±2σ with
  σ = sqrt(2 / fan_in) / 0.87962566103423978) for the block convolutions;
  ``glorot_uniform`` for the transposed convolutions and the 1×1 head (Keras
  layer defaults); all biases zero. PyTorch's own defaults (Kaiming-uniform
  weights, uniform biases) are not used.

Equivalence with the Keras model is verified numerically by transferring the
weights of a Keras model and comparing the outputs (see the Batch 1 test).
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: Keras ``truncated_normal`` correction: std of a standard normal truncated at ±2.
_TRUNC_STD: float = 0.87962566103423978

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": lambda: nn.LeakyReLU(0.1), "swish": nn.SiLU}


def keras_he_normal_(weight: torch.Tensor, fan_in: int) -> None:
    """In-place Keras ``he_normal`` (truncated normal, scaled for truncation)."""
    std = math.sqrt(2.0 / fan_in) / _TRUNC_STD
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


def keras_glorot_uniform_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    """In-place Keras ``glorot_uniform``."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -limit, limit)


def keras_bn(channels: int) -> nn.BatchNorm2d:
    """BatchNorm with Keras defaults (momentum 0.99 ≡ PyTorch 0.01, eps 1e-3)."""
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)


class ConvBlock(nn.Module):
    """Keras ``conv2d_block``: (Conv 3×3 same → BN → act) × 2."""

    def __init__(self, c_in: int, c_out: int, batchnorm: bool, activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(2):
            conv = nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1)
            keras_he_normal_(conv.weight, fan_in=conv.in_channels * 9)
            nn.init.zeros_(conv.bias)
            layers.append(conv)
            if batchnorm:
                layers.append(keras_bn(c_out))
            layers.append(ACTIVATIONS[activation]())
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class KerasSameConvTranspose2d(nn.Module):
    """``Conv2DTranspose(c_out, 3, strides=2, padding="same")`` with Keras alignment."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=0)
        keras_glorot_uniform_(self.conv.weight, fan_in=9 * c_out, fan_out=9 * c_in)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        return self.conv(x)[..., : 2 * h, : 2 * w]


class UNet(nn.Module):
    """U-Net of the original Keras baseline (4 levels + bottleneck)."""

    def __init__(
        self,
        n_filters: int = 16,
        dropout: float = 0.1,
        batchnorm: bool = True,
        activation: str = "relu",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        f = n_filters
        widths = [f, 2 * f, 4 * f, 8 * f]
        self.encoders = nn.ModuleList()
        c = in_channels
        for w in widths:
            self.encoders.append(ConvBlock(c, w, batchnorm, activation))
            c = w
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)
        self.bottleneck = ConvBlock(8 * f, 16 * f, batchnorm, activation)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        c = 16 * f
        for w in reversed(widths):
            self.ups.append(KerasSameConvTranspose2d(c, w))
            self.decoders.append(ConvBlock(2 * w, w, batchnorm, activation))
            c = w
        self.head = nn.Conv2d(f, 1, 1)
        keras_glorot_uniform_(self.head.weight, fan_in=f, fan_out=1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel **logits** of shape ``(B, 1, H, W)``."""
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.drop(self.pool(x))
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(self.drop(torch.cat([up(x), skip], dim=1)))
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities (the Keras model's output)."""
        return torch.sigmoid(self(x))


def build_model(protocol: dict) -> UNet:
    """Build the U-Net from a protocol dict (base setup + hyperparameters)."""
    return UNet(
        n_filters=protocol["n_filters"],
        dropout=protocol["dropout"],
        batchnorm=protocol["batchnorm"],
        activation=protocol["activation"],
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (Keras ``trainable_params`` equivalent)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fuse(model: UNet) -> UNet:
    """Fold every BatchNorm into its preceding convolution (inference form), in place.

    The deployment form used by the efficiency benchmark — the same
    transformation Ultralytics applies to YOLO26 at inference
    (``model.fuse()``), so parameter counts, GFLOPs and latency of both
    architectures are measured on equivalent graphs. Valid in eval mode only.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    model.eval()
    for block in [*model.encoders, model.bottleneck, *model.decoders]:
        layers = list(block.body)
        fused: list[nn.Module] = []
        i = 0
        while i < len(layers):
            if (isinstance(layers[i], nn.Conv2d) and i + 1 < len(layers)
                    and isinstance(layers[i + 1], nn.BatchNorm2d)):
                fused.append(fuse_conv_bn_eval(layers[i], layers[i + 1]))
                i += 2
            else:
                fused.append(layers[i])
                i += 1
        block.body = nn.Sequential(*fused)
    return model
