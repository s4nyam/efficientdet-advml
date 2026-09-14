"""Bounding-box helpers: format conversion, IoU, encoding/decoding and NMS.

Two coordinate conventions appear throughout this project and mixing them up
is the single most common source of silently wrong results:

``xyxy``
    ``(x1, y1, x2, y2)`` absolute pixel corners. Used by COCO tooling,
    torchvision and this package's anchors.
``cxcywh``
    ``(x_center, y_center, width, height)``, normalised to ``[0, 1]`` by the
    image dimensions. This is the YOLO text format.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "cxcywh_to_xyxy",
    "xyxy_to_cxcywh",
    "box_area",
    "box_iou",
    "encode_boxes",
    "decode_boxes",
    "nms",
    "batched_nms",
]


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert ``(x1, y1, x2, y2)`` to ``(cx, cy, w, h)``."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Area of ``xyxy`` boxes, clamped at zero for degenerate boxes."""
    w = (boxes[..., 2] - boxes[..., 0]).clamp(min=0)
    h = (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
    return w * h


def box_iou(boxes1: Tensor, boxes2: Tensor, *, eps: float = 1e-7) -> Tensor:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` ``xyxy`` boxes.

    Returns:
        ``[N, M]`` matrix of intersection-over-union values.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def encode_boxes(
    boxes: Tensor,
    anchors: Tensor,
    *,
    weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> Tensor:
    """Encode ground-truth ``xyxy`` boxes as offsets relative to anchors.

    The standard R-CNN parameterisation: centre offsets are divided by the
    anchor size, sizes are log-ratios. ``weights`` rescale the four terms so
    they contribute comparably to the regression loss.
    """
    gt = xyxy_to_cxcywh(boxes)
    an = xyxy_to_cxcywh(anchors)
    wx, wy, ww, wh = weights
    dx = wx * (gt[..., 0] - an[..., 0]) / an[..., 2].clamp(min=1e-6)
    dy = wy * (gt[..., 1] - an[..., 1]) / an[..., 3].clamp(min=1e-6)
    dw = ww * torch.log(gt[..., 2].clamp(min=1e-6) / an[..., 2].clamp(min=1e-6))
    dh = wh * torch.log(gt[..., 3].clamp(min=1e-6) / an[..., 3].clamp(min=1e-6))
    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_boxes(
    deltas: Tensor,
    anchors: Tensor,
    *,
    weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
    max_ratio: float = 4.0,
) -> Tensor:
    """Inverse of :func:`encode_boxes`; returns ``xyxy`` boxes.

    ``max_ratio`` clamps the log-scale term so a wild prediction cannot produce
    an astronomically large box early in training.
    """
    an = xyxy_to_cxcywh(anchors)
    wx, wy, ww, wh = weights
    dx = deltas[..., 0] / wx
    dy = deltas[..., 1] / wy
    dw = (deltas[..., 2] / ww).clamp(max=max_ratio)
    dh = (deltas[..., 3] / wh).clamp(max=max_ratio)
    cx = dx * an[..., 2] + an[..., 0]
    cy = dy * an[..., 3] + an[..., 1]
    w = torch.exp(dw) * an[..., 2]
    h = torch.exp(dh) * an[..., 3]
    return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=-1))


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """Greedy non-maximum suppression.

    Args:
        boxes: ``[N, 4]`` ``xyxy`` boxes.
        scores: ``[N]`` confidence scores.
        iou_threshold: Boxes overlapping a kept box by more than this are
            discarded.

    Returns:
        Indices of kept boxes, ordered by descending score.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: Tensor, scores: Tensor, labels: Tensor, iou_threshold: float = 0.5
) -> Tensor:
    """Class-aware NMS: boxes of different classes never suppress each other.

    Implemented with the usual coordinate-offset trick so a single NMS pass
    handles every class at once.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max()
    offsets = labels.to(boxes.dtype) * (max_coord + 1)
    return nms(boxes + offsets[:, None], scores, iou_threshold)
