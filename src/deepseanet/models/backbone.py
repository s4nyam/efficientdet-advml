"""A compact EfficientNet-style backbone built from MBConv blocks.

This is a *reference* backbone, not a re-implementation of EfficientNet-B0
weight for weight. It exists so the neck and head can be exercised, profiled
and unit-tested end to end without pulling in a heavyweight model zoo. For
real training, prefer a pretrained backbone (``timm``, ``torchvision`` or the
TFLite Model Maker spec used in the committed notebooks) and feed its
intermediate feature maps into :class:`~deepseanet.models.biskfpn.BiSkFPN`.

The MBConv block follows the paper's description: a 1x1 expansion, a depthwise
k x k convolution, squeeze-and-excitation, a 1x1 projection, and a residual
shortcut when the shapes allow it. Swish is used throughout.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .activations import Swish

__all__ = ["MBConv", "SqueezeExcite", "BackboneConfig", "EfficientNetLite", "build_backbone"]


def _round_channels(channels: float, divisor: int = 8) -> int:
    """Round channel counts to a multiple of ``divisor`` (compound scaling)."""
    new = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if new < 0.9 * channels:  # never drop more than 10%
        new += divisor
    return int(new)


class SqueezeExcite(nn.Module):
    """Channel attention: global pool -> bottleneck -> sigmoid gate."""

    def __init__(self, channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = Swish()

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = self.expand(self.act(self.reduce(s)))
        return x * torch.sigmoid(s)


class MBConv(nn.Module):
    """Mobile inverted bottleneck convolution.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        expand_ratio: Expansion factor ``T`` of the first 1x1 convolution.
        kernel_size: Spatial size of the depthwise convolution.
        stride: Stride of the depthwise convolution.
        se_ratio: Squeeze-and-excitation ratio; ``0`` disables the SE block.
        drop_path: Stochastic-depth probability applied to the residual branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_path = float(drop_path)

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden, momentum=0.01, eps=1e-3),
                Swish(),
            ]
        layers += [
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden, momentum=0.01, eps=1e-3),
            Swish(),
        ]
        if se_ratio > 0:
            layers.append(SqueezeExcite(hidden, se_ratio))
        layers += [
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if not self.use_residual:
            return out
        if self.training and self.drop_path > 0.0:
            keep = 1.0 - self.drop_path
            mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) < keep
            out = out * mask / keep
        return x + out


@dataclass(frozen=True)
class BackboneConfig:
    """Compound-scaling knobs, mirroring EfficientNet's ``phi``.

    Args:
        width: Channel multiplier.
        depth: Repeat multiplier.
        stem_channels: Channels after the stem convolution.
        drop_path: Maximum stochastic-depth rate, ramped linearly with depth.
    """

    width: float = 1.0
    depth: float = 1.0
    stem_channels: int = 32
    drop_path: float = 0.0

    @classmethod
    def from_phi(cls, phi: int = 0, **kwargs: object) -> BackboneConfig:
        """EfficientNet's scaling rule: ``width = 1.2^phi``, ``depth = 1.1^phi``."""
        return cls(width=1.2**phi, depth=1.1**phi, **kwargs)  # type: ignore[arg-type]


# (expand_ratio, out_channels, repeats, stride, kernel_size) per stage.
_STAGES: tuple[tuple[int, int, int, int, int], ...] = (
    (1, 16, 1, 1, 3),
    (6, 24, 2, 2, 3),
    (6, 40, 2, 2, 5),  # -> P3
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),  # -> P4
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),  # -> P5
)
_FEATURE_STAGES = (2, 4, 6)  # stage indices whose outputs become P3, P4, P5


class EfficientNetLite(nn.Module):
    """MBConv backbone emitting a five-level pyramid ``(P3, P4, P5, P6, P7)``.

    P6 and P7 are produced by strided convolutions on top of P5, exactly as
    EfficientDet extends the backbone pyramid.

    Args:
        config: Compound-scaling configuration.
        in_channels: Channels of the input image (3 for RGB).
        extra_levels: How many extra coarse levels to synthesise above P5.
    """

    def __init__(
        self,
        config: BackboneConfig | None = None,
        *,
        in_channels: int = 3,
        extra_levels: int = 2,
    ) -> None:
        super().__init__()
        cfg = config or BackboneConfig()
        self.config = cfg

        stem = _round_channels(cfg.stem_channels * cfg.width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem, momentum=0.01, eps=1e-3),
            Swish(),
        )

        total = sum(max(1, int(round(r * cfg.depth))) for _, _, r, _, _ in _STAGES)
        built = 0
        channels = stem
        self.stages = nn.ModuleList()
        pyramid_channels: list[int] = []
        for idx, (expand, out_c, repeats, stride, kernel) in enumerate(_STAGES):
            out_c = _round_channels(out_c * cfg.width)
            repeats = max(1, int(round(repeats * cfg.depth)))
            blocks: list[nn.Module] = []
            for r in range(repeats):
                blocks.append(
                    MBConv(
                        channels,
                        out_c,
                        expand_ratio=expand,
                        kernel_size=kernel,
                        stride=stride if r == 0 else 1,
                        drop_path=cfg.drop_path * built / max(1, total - 1),
                    )
                )
                channels = out_c
                built += 1
            self.stages.append(nn.Sequential(*blocks))
            if idx in _FEATURE_STAGES:
                pyramid_channels.append(out_c)

        self.extra = nn.ModuleList()
        c = pyramid_channels[-1]
        for _ in range(extra_levels):
            self.extra.append(
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(c, momentum=0.01, eps=1e-3),
                    Swish(),
                )
            )
            pyramid_channels.append(c)

        self.out_channels: tuple[int, ...] = tuple(pyramid_channels)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features: list[Tensor] = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in _FEATURE_STAGES:
                features.append(x)
        for extra in self.extra:
            x = extra(x)
            features.append(x)
        return features


def build_backbone(phi: int = 0, **kwargs: object) -> EfficientNetLite:
    """Build the backbone at compound-scaling coefficient ``phi``."""
    drop_path = float(kwargs.pop("drop_path", 0.0))  # type: ignore[arg-type]
    return EfficientNetLite(BackboneConfig.from_phi(phi, drop_path=drop_path), **kwargs)  # type: ignore[arg-type]


def pyramid_strides(n_levels: int = 5) -> Sequence[int]:
    """Down-sampling factor of each pyramid level: P3=8, P4=16, ... ."""
    return [2 ** (i + 3) for i in range(n_levels)]
