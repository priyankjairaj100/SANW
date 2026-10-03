"""Zero-initialized residual adapters over frozen image and text features."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualAdapter(nn.Module):
    """Two independent, bias-free maps followed by unit-L2 normalization.

    At initialization both maps are zero, so the adapter reproduces normalized
    frozen features. A 512-dimensional instance has exactly 524,288 parameters.
    """

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.image = nn.Linear(dim, dim, bias=False)
        self.text = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.image.weight)
        nn.init.zeros_(self.text.weight)

    def encode_image(self, x: Tensor) -> Tensor:
        return F.normalize(x + self.image(x), dim=-1)

    def encode_text(self, x: Tensor) -> Tensor:
        return F.normalize(x + self.text(x), dim=-1)

    def forward(self, images: Tensor, texts: Tensor) -> tuple[Tensor, Tensor]:
        return self.encode_image(images), self.encode_text(texts)
