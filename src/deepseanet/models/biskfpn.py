"""Feature-pyramid necks: the original BiFPN and the proposed BiSkFPN.

BiFPN, the neck of EfficientDet, fuses a pyramid of feature maps with a
top-down pass followed by a bottom-up pass, weighting every input edge by a
learned scalar (fast normalised fusion).

BiSkFPN (Bidirectional Skip-connection FPN) is the modification proposed in
the paper. Each output node is built from three sources instead of one:

.. math::
    \\mathrm{BiSkFPN}(P)_i = \\mathrm{concat}\\bigl(
        P_i,\\;
        \\mathrm{deconv}(P_{i+1}),\\;
        \\mathrm{skip}_{i\\to i+1}(P_{i-1})
    \\bigr)

* ``P_i`` is the feature map at the current level.
* ``deconv(P_{i+1})`` is the next-coarser level brought back up with a
  *transposed convolution* rather than a parameter-free resize, so the
  up-sampling filter is learned and some of the spatial detail lost during
  down-sampling is recovered.
* ``skip(P_{i-1})`` carries the next-finer level straight across, past the
  deconvolution branch, so low-level cues (edges, small blobs, the few dozen
  pixels of a shrimp) reach the head without being repeatedly re-mixed.

The concatenated tensor is projected back to ``channels`` with a 1x1
convolution. The claimed benefit is robustness: when the input is perturbed,
the fine-detail path does not have to survive every fusion stage to reach the
prediction layers.

Boundary levels have no ``P_{i+1}`` or no ``P_{i-1}``; those branches are
simply dropped and the fuse convolution is sized accordingly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import Swish

__all__ = [
    "SeparableConvBlock",
    "WeightedFusion",
    "BiFPNLayer",
    "BiSkFPNLayer",
    "BiFPN",
    "BiSkFPN",
    "build_neck",
]

_EPS = 1e-4


class SeparableConvBlock(nn.Module):
    """Depthwise-separable conv + batch norm, optionally followed by Swish."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        activate: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not norm)
        self.norm = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3) if norm else nn.Identity()
        self.act = Swish() if activate else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class WeightedFusion(nn.Module):
    """Fast normalised fusion of ``n`` inputs, as used by BiFPN.

    .. math:: O = \\sum_i \\frac{\\mathrm{ReLU}(w_i)}{\\epsilon + \\sum_j \\mathrm{ReLU}(w_j)} I_i

    The ReLU keeps the weights non-negative so the normalisation is stable,
    which is the whole point of "fast" fusion over a softmax.
    """

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))

    def forward(self, inputs: Sequence[Tensor]) -> Tensor:
        if len(inputs) != self.weight.numel():
            raise ValueError(
                f"WeightedFusion expects {self.weight.numel()} inputs, got {len(inputs)}"
            )
        w = F.relu(self.weight)
        w = w / (w.sum() + _EPS)
        out = inputs[0] * w[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i] * w[i]
        return out


def _resize_to(x: Tensor, ref: Tensor) -> Tensor:
    """Nearest-neighbour resize of ``x`` onto the spatial size of ``ref``."""
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


class BiFPNLayer(nn.Module):
    """One bidirectional pass: top-down then bottom-up, with weighted fusion.

    Args:
        channels: Channel width shared by every pyramid level.
        n_levels: Number of pyramid levels, finest first (P3 ... P7).
    """

    def __init__(self, channels: int, n_levels: int) -> None:
        super().__init__()
        if n_levels < 2:
            raise ValueError("BiFPN needs at least two pyramid levels")
        self.n_levels = n_levels
        # Top-down: every level except the coarsest fuses (self, coarser).
        self.td_fuse = nn.ModuleList(WeightedFusion(2) for _ in range(n_levels - 1))
        self.td_conv = nn.ModuleList(SeparableConvBlock(channels) for _ in range(n_levels - 1))
        # Bottom-up: every level except the finest fuses (self, top-down, finer);
        # the coarsest level has no top-down node, so it fuses two inputs.
        self.bu_fuse = nn.ModuleList(
            WeightedFusion(2 if i == n_levels - 1 else 3) for i in range(1, n_levels)
        )
        self.bu_conv = nn.ModuleList(SeparableConvBlock(channels) for _ in range(n_levels - 1))

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        if len(features) != self.n_levels:
            raise ValueError(f"expected {self.n_levels} feature maps, got {len(features)}")

        # --- top-down (coarse -> fine) ---
        td: list[Tensor] = [None] * self.n_levels  # type: ignore[list-item]
        td[-1] = features[-1]
        for i in range(self.n_levels - 2, -1, -1):
            up = _resize_to(td[i + 1], features[i])
            td[i] = self.td_conv[i](self.td_fuse[i]([features[i], up]))

        # --- bottom-up (fine -> coarse) ---
        out: list[Tensor] = [None] * self.n_levels  # type: ignore[list-item]
        out[0] = td[0]
        for i in range(1, self.n_levels):
            down = _resize_to(out[i - 1], features[i])
            parts = [features[i], down] if i == self.n_levels - 1 else [features[i], td[i], down]
            out[i] = self.bu_conv[i - 1](self.bu_fuse[i - 1](parts))
        return out


class BiSkFPNLayer(nn.Module):
    """One BiSkFPN pass: BiFPN fusion plus deconvolution and skip branches.

    On top of the bidirectional fusion, each output level concatenates a
    learned up-sample of the coarser level and an untouched skip from the finer
    level, then projects the result back to ``channels``.

    Args:
        channels: Channel width shared by every pyramid level.
        n_levels: Number of pyramid levels, finest first.
        deconv_kernel: Kernel size of the transposed convolution used to
            up-sample the coarser level.
    """

    def __init__(self, channels: int, n_levels: int, *, deconv_kernel: int = 3) -> None:
        super().__init__()
        if n_levels < 2:
            raise ValueError("BiSkFPN needs at least two pyramid levels")
        self.n_levels = n_levels
        self.channels = channels
        self.bifpn = BiFPNLayer(channels, n_levels)

        # Learned up-sampling of P_{i+1} onto the grid of P_i. Defined for every
        # level that has a coarser neighbour, i.e. all but the last.
        pad = deconv_kernel // 2
        self.deconv = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels,
                    channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=pad,
                    output_padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                Swish(),
            )
            for _ in range(n_levels - 1)
        )

        # Skip connection carrying P_{i-1} across, strided to match P_i.
        # Defined for every level that has a finer neighbour, i.e. all but the first.
        self.skip = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
            )
            for _ in range(n_levels - 1)
        )

        # concat(P_i, deconv, skip) -> channels. Boundary levels get fewer parts.
        self.fuse = nn.ModuleList()
        for i in range(n_levels):
            n_parts = 1 + int(i < n_levels - 1) + int(i > 0)
            self.fuse.append(SeparableConvBlock(channels * n_parts, channels))

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        fused = self.bifpn(features)
        out: list[Tensor] = []
        for i, feat in enumerate(fused):
            parts = [feat]
            if i < self.n_levels - 1:  # deconv(P_{i+1}) -> grid of P_i
                parts.append(_resize_to(self.deconv[i](fused[i + 1]), feat))
            if i > 0:  # skip_{i-1 -> i}, bypassing the deconv branch
                parts.append(_resize_to(self.skip[i - 1](fused[i - 1]), feat))
            out.append(self.fuse[i](torch.cat(parts, dim=1)))
        return out


class _Neck(nn.Module):
    """Shared plumbing: lateral 1x1 projections then ``n_layers`` fusion passes."""

    layer_cls: type[nn.Module]

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 64,
        n_layers: int = 3,
        **layer_kwargs: object,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.n_levels = len(self.in_channels)
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
            )
            for c in self.in_channels
        )
        self.layers = nn.ModuleList(
            self.layer_cls(channels, self.n_levels, **layer_kwargs)  # type: ignore[arg-type]
            for _ in range(n_layers)
        )

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        if len(features) != self.n_levels:
            raise ValueError(f"expected {self.n_levels} feature maps, got {len(features)}")
        xs = [lat(f) for lat, f in zip(self.lateral, features)]
        for layer in self.layers:
            xs = layer(xs)
        return xs


class BiFPN(_Neck):
    """Stacked :class:`BiFPNLayer` — the neck of the original EfficientDet."""

    layer_cls = BiFPNLayer


class BiSkFPN(_Neck):
    """Stacked :class:`BiSkFPNLayer` — the neck proposed in the paper."""

    layer_cls = BiSkFPNLayer


def build_neck(
    kind: Literal["bifpn", "biskfpn"],
    in_channels: Sequence[int],
    channels: int = 64,
    n_layers: int = 3,
) -> _Neck:
    """Factory used by :class:`~deepseanet.models.detector.DeepSeaNet`."""
    kinds = {"bifpn": BiFPN, "biskfpn": BiSkFPN}
    if kind not in kinds:
        raise ValueError(f"unknown neck {kind!r}, expected one of {sorted(kinds)}")
    return kinds[kind](in_channels, channels=channels, n_layers=n_layers)
