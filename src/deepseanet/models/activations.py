"""Activation functions used by the DeepSeaNet backbone and neck.

The paper argues for Swish over ReLU in the backbone: it is smooth, keeps a
non-zero gradient for small negative inputs, and therefore damps weak noisy
activations instead of clipping them to zero. On murky frames a large share of
activations sit close to zero, which is exactly where the two functions differ.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["Swish", "swish", "swish_derivative"]


def swish(x: Tensor, beta: float | Tensor = 1.0) -> Tensor:
    r"""Functional Swish, :math:`\mathrm{Swish}(x) = x \cdot \sigma(\beta x)`."""
    return x * torch.sigmoid(beta * x)


def swish_derivative(x: Tensor, beta: float | Tensor = 1.0) -> Tensor:
    r"""Analytic derivative of :func:`swish`.

    .. math::
        \frac{d}{dx}\mathrm{Swish}(x)
        = \sigma(\beta x) + \beta x\,\sigma(\beta x)\bigl(1 - \sigma(\beta x)\bigr)

    Provided for the activation figure in the paper and for tests; autograd is
    used everywhere in the model itself.
    """
    s = torch.sigmoid(beta * x)
    return s + beta * x * s * (1.0 - s)


class Swish(nn.Module):
    r"""Swish activation, optionally with a learnable :math:`\beta`.

    Args:
        beta: Initial value of the slope parameter.
        trainable: If ``True``, ``beta`` becomes an ``nn.Parameter`` and is
            learned jointly with the network weights, as described in the
            paper. If ``False`` the activation is the standard SiLU and the
            faster fused kernel is used.
    """

    def __init__(self, beta: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        self.trainable = trainable
        if trainable:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(self, x: Tensor) -> Tensor:
        if not self.trainable and float(self.beta) == 1.0:
            return nn.functional.silu(x)
        return swish(x, self.beta)

    def extra_repr(self) -> str:
        return f"beta={float(self.beta):.3f}, trainable={self.trainable}"
