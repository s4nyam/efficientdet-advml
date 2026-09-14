"""The assembled detector: backbone -> neck -> head.

``DeepSeaNet`` is the paper's model with the neck swapped for BiSkFPN. Set
``neck="bifpn"`` to get the stock EfficientDet arrangement, which is the
comparison the paper draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from .backbone import build_backbone, pyramid_strides
from .biskfpn import build_neck
from .head import AnchorGenerator, DetectionHead

__all__ = ["DeepSeaNetConfig", "DeepSeaNet", "build_model"]


@dataclass
class DeepSeaNetConfig:
    """Configuration for :class:`DeepSeaNet`.

    Args:
        num_classes: Number of object classes (6 for Brackish).
        phi: Compound-scaling coefficient shared by backbone, neck and head.
        neck: ``"biskfpn"`` for the proposed neck, ``"bifpn"`` for the baseline.
        neck_channels: Channel width inside the neck. Defaults to the
            EfficientDet rule ``64 * 1.35**phi``.
        neck_layers: How many times the fusion pass is repeated.
        head_layers: Depth of the class and box subnets.
        aspect_ratios: Anchor aspect ratios.
        scales: Anchor scales.
        image_size: Square input resolution the model is trained at.
    """

    num_classes: int = 6
    phi: int = 0
    neck: Literal["bifpn", "biskfpn"] = "biskfpn"
    neck_channels: int | None = None
    neck_layers: int | None = None
    head_layers: int | None = None
    aspect_ratios: Sequence[float] = field(default_factory=lambda: (0.5, 1.0, 2.0))
    scales: Sequence[float] = field(default_factory=lambda: (1.0, 2 ** (1 / 3), 2 ** (2 / 3)))
    image_size: int = 512

    def resolved(self) -> DeepSeaNetConfig:
        """Fill in the EfficientDet scaling defaults for any unset field."""
        return DeepSeaNetConfig(
            num_classes=self.num_classes,
            phi=self.phi,
            neck=self.neck,
            neck_channels=self.neck_channels or int(64 * 1.35**self.phi),
            neck_layers=self.neck_layers or (3 + self.phi),
            head_layers=self.head_layers or (3 + self.phi // 3),
            aspect_ratios=tuple(self.aspect_ratios),
            scales=tuple(self.scales),
            image_size=self.image_size,
        )


class DeepSeaNet(nn.Module):
    """One-stage detector with an optional skip-connected feature-pyramid neck.

    Args:
        config: Model configuration. Defaults to the ``phi=0`` BiSkFPN variant.
        backbone: Optional pre-built backbone. Any module returning a list of
            feature maps, finest first, works — pass a pretrained EfficientNet
            here for real training.

    Example:
        >>> import torch
        >>> from deepseanet.models import DeepSeaNet, DeepSeaNetConfig
        >>> model = DeepSeaNet(DeepSeaNetConfig(num_classes=6, image_size=256))
        >>> cls_logits, box_deltas, anchors = model(torch.randn(1, 3, 256, 256))
        >>> cls_logits.shape[-1], box_deltas.shape[-1]
        (6, 4)
    """

    def __init__(
        self,
        config: DeepSeaNetConfig | None = None,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = (config or DeepSeaNetConfig()).resolved()
        self.config = cfg

        self.backbone: nn.Module = backbone or build_backbone(cfg.phi)
        if hasattr(self.backbone, "out_channels"):
            in_channels = list(self.backbone.out_channels)  # type: ignore[arg-type]
        else:
            raise ValueError("custom backbones must expose an `out_channels` attribute")

        assert cfg.neck_channels is not None and cfg.neck_layers is not None
        self.neck = build_neck(
            cfg.neck, in_channels, channels=cfg.neck_channels, n_layers=cfg.neck_layers
        )
        self.anchors = AnchorGenerator(
            strides=pyramid_strides(len(in_channels)),
            aspect_ratios=cfg.aspect_ratios,
            scales=cfg.scales,
        )
        assert cfg.head_layers is not None
        self.head = DetectionHead(
            cfg.neck_channels,
            cfg.num_classes,
            num_anchors=self.anchors.num_anchors,
            n_layers=cfg.head_layers,
            n_levels=len(in_channels),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the detector.

        Args:
            images: Batch of images, ``[B, 3, H, W]``, normalised to roughly
                zero mean and unit variance.

        Returns:
            ``(cls_logits, box_deltas, anchors)`` where ``cls_logits`` is
            ``[B, A, num_classes]``, ``box_deltas`` is ``[B, A, 4]`` and
            ``anchors`` is ``[A, 4]`` in ``(x1, y1, x2, y2)`` pixels.
        """
        self._check_input_size(images)
        features = self.backbone(images)
        features = self.neck(features)
        cls_logits, box_deltas = self.head(features)
        anchors = self.anchors(features)
        return cls_logits, box_deltas, anchors

    def _check_input_size(self, images: Tensor) -> None:
        """Reject inputs the pyramid cannot be built from.

        The coarsest level is down-sampled by ``2**(n_levels + 2)`` -- 128 for
        the default five levels. A size that is not a multiple of that produces
        off-by-one feature maps that silently misalign with the anchors, which
        is far harder to debug than an exception here.
        """
        stride = 2 ** (len(self.anchors.strides) + 2)
        h, w = images.shape[-2:]
        if h % stride or w % stride:
            raise ValueError(
                f"input {h}x{w} must be divisible by {stride} (the coarsest pyramid stride); "
                f"resize or pad first"
            )
        if min(h, w) < stride:
            raise ValueError(f"input {h}x{w} is smaller than the coarsest stride {stride}")

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count, for the model-size comparisons in the paper."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> dict[str, int]:
        """Parameters per component, useful when comparing BiFPN to BiSkFPN."""
        return {
            name: sum(p.numel() for p in module.parameters())
            for name, module in (
                ("backbone", self.backbone),
                ("neck", self.neck),
                ("head", self.head),
            )
        }


def build_model(
    num_classes: int = 6,
    phi: int = 0,
    neck: Literal["bifpn", "biskfpn"] = "biskfpn",
    **kwargs: object,
) -> DeepSeaNet:
    """Convenience constructor mirroring the config fields."""
    return DeepSeaNet(DeepSeaNetConfig(num_classes=num_classes, phi=phi, neck=neck, **kwargs))  # type: ignore[arg-type]
