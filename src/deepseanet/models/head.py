"""Detection head: a shared classification subnet and box-regression subnet.

The head follows EfficientDet. Both subnets are applied to every pyramid level
with shared weights, each is a stack of depthwise-separable convolutions, and
the final layer predicts ``num_anchors * num_classes`` logits and
``num_anchors * 4`` box offsets per spatial location.

Anchor generation is included because the paper's related-work section spends
time on it: anchor aspect ratios matter a lot when objects have extreme shapes,
and a school of small fish is about as extreme as it gets on this dataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .activations import Swish
from .biskfpn import SeparableConvBlock

__all__ = ["HeadSubnet", "DetectionHead", "AnchorGenerator"]


class HeadSubnet(nn.Module):
    """``n_layers`` separable convs followed by a linear prediction layer.

    Batch-norm statistics are kept per pyramid level (a detail from the
    EfficientDet paper) while the convolution weights are shared, so a level
    with a very different activation scale does not drag the others.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        *,
        n_layers: int = 3,
        n_levels: int = 5,
        prior_prob: float | None = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList(
            SeparableConvBlock(channels, channels, activate=False, norm=False)
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleList(
                nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3) for _ in range(n_layers)
            )
            for _ in range(n_levels)
        )
        self.act = Swish()
        self.predict = SeparableConvBlock(channels, out_channels, activate=False, norm=False)

        if prior_prob is not None:
            # Focal-loss init: start with a low objectness prior so the huge
            # number of background anchors does not swamp training at step 0.
            bias = -math.log((1.0 - prior_prob) / prior_prob)
            nn.init.constant_(self.predict.pointwise.bias, bias)

    def forward(self, x: Tensor, level: int) -> Tensor:
        for conv, norm in zip(self.convs, self.norms[level]):
            x = self.act(norm(conv(x)))
        return self.predict(x)


class DetectionHead(nn.Module):
    """Classification and box-regression subnets over the whole pyramid.

    Returns flattened predictions concatenated across levels so they line up
    with :class:`AnchorGenerator` output:
    ``(cls_logits[B, A, num_classes], box_deltas[B, A, 4])``.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        *,
        num_anchors: int = 9,
        n_layers: int = 3,
        n_levels: int = 5,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.class_net = HeadSubnet(
            channels,
            num_anchors * num_classes,
            n_layers=n_layers,
            n_levels=n_levels,
            prior_prob=prior_prob,
        )
        self.box_net = HeadSubnet(channels, num_anchors * 4, n_layers=n_layers, n_levels=n_levels)

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        cls_out, box_out = [], []
        for level, feat in enumerate(features):
            b = feat.shape[0]
            c = self.class_net(feat, level)
            r = self.box_net(feat, level)
            # [B, A*K, H, W] -> [B, H*W*A, K]
            c = c.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            r = r.permute(0, 2, 3, 1).reshape(b, -1, 4)
            cls_out.append(c)
            box_out.append(r)
        return torch.cat(cls_out, dim=1), torch.cat(box_out, dim=1)


class AnchorGenerator(nn.Module):
    """Anchors in ``(x1, y1, x2, y2)`` pixel coordinates for each level.

    Args:
        strides: Down-sampling factor of each pyramid level.
        sizes: Base anchor edge length per level. Defaults to ``4 * stride``.
        aspect_ratios: Width-to-height ratios.
        scales: Multiplicative scales applied to each base size.
    """

    def __init__(
        self,
        strides: Sequence[int] = (8, 16, 32, 64, 128),
        sizes: Sequence[float] | None = None,
        aspect_ratios: Sequence[float] = (0.5, 1.0, 2.0),
        scales: Sequence[float] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ) -> None:
        super().__init__()
        self.strides = list(strides)
        self.sizes = list(sizes) if sizes is not None else [4.0 * s for s in self.strides]
        self.aspect_ratios = list(aspect_ratios)
        self.scales = list(scales)

    @property
    def num_anchors(self) -> int:
        return len(self.aspect_ratios) * len(self.scales)

    def _cell_anchors(self, size: float, device: torch.device, dtype: torch.dtype) -> Tensor:
        boxes = []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                area = (size * scale) ** 2
                w = math.sqrt(area / ratio)
                h = w * ratio
                boxes.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(boxes, device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != len(self.strides):
            raise ValueError(
                f"AnchorGenerator configured for {len(self.strides)} levels, "
                f"got {len(features)} feature maps"
            )
        device, dtype = features[0].device, features[0].dtype
        all_anchors = []
        for feat, stride, size in zip(features, self.strides, self.sizes):
            h, w = feat.shape[-2:]
            shift_x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
            shift_y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
            yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shifts = torch.stack([xx, yy, xx, yy], dim=-1).reshape(-1, 1, 4)
            cell = self._cell_anchors(size, device, dtype).reshape(1, -1, 4)
            all_anchors.append((shifts + cell).reshape(-1, 4))
        return torch.cat(all_anchors, dim=0)
