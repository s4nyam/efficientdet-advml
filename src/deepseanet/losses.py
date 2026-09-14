r"""Losses for the detection head.

The paper's head computes three things, and the total objective is their
weighted sum:

.. math:: L = L_{cls} + \alpha L_{box} + \beta L_{reg}

* ``L_cls`` — classification. The paper writes it as cross-entropy and calls
  the head "multi-focal loss"; focal loss is cross-entropy with a
  :math:`(1-p_t)^\gamma` modulating term, which is what a detector with tens of
  thousands of mostly-background anchors actually needs. Both are available
  here; :func:`focal_loss` reduces to cross-entropy at ``gamma=0``.
* ``L_box`` — smooth L1 between predicted and ground-truth box offsets.
* ``L_reg`` — an L2 penalty on the weights, equivalent to the weight decay of
  ``5e-4`` used in the committed runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "focal_loss",
    "smooth_l1_loss",
    "l2_regularization",
    "DetectionLoss",
]


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: Literal["none", "sum", "mean"] = "sum",
) -> Tensor:
    r"""Sigmoid focal loss for multi-label / multi-class-per-anchor prediction.

    Args:
        logits: Raw scores, ``[..., num_classes]``.
        targets: One-hot (or multi-hot) targets with the same shape as
            ``logits``.
        alpha: Weight applied to the positive class; ``0.25`` is the value in
            the paper.
        gamma: Focusing parameter; ``2.0`` in the paper. ``gamma=0`` and
            ``alpha=0.5`` recover plain binary cross-entropy up to a constant.
        reduction: ``"none"``, ``"sum"`` or ``"mean"``.

    Returns:
        The reduced loss.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}"
        )
    targets = targets.to(logits.dtype)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * (1.0 - p_t).pow(gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def smooth_l1_loss(
    pred: Tensor,
    target: Tensor,
    *,
    beta: float = 1.0,
    reduction: Literal["none", "sum", "mean"] = "sum",
) -> Tensor:
    r"""Smooth L1 (Huber) loss.

    .. math::
        \mathrm{smooth}_{L1}(x) =
        \begin{cases}
            0.5 x^2 / \beta & |x| < \beta \\
            |x| - 0.5\beta  & \text{otherwise}
        \end{cases}

    With ``beta=1`` this is the piecewise definition printed in the paper.
    """
    diff = (pred - target).abs()
    loss = torch.where(diff < beta, 0.5 * diff.pow(2) / beta, diff - 0.5 * beta)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def l2_regularization(parameters: Iterable[Tensor] | nn.Module, *, skip_1d: bool = True) -> Tensor:
    r"""Mean squared L2 norm of the network weights.

    Args:
        parameters: An iterable of tensors, or a module whose parameters are
            used.
        skip_1d: Exclude 1-D parameters (biases, batch-norm scale and shift),
            which is standard practice — decaying them hurts more than it helps.

    Returns:
        A scalar tensor. Multiply by ``beta`` and add to the total loss, or
        equivalently set ``weight_decay`` on the optimiser.
    """
    if isinstance(parameters, nn.Module):
        parameters = parameters.parameters()
    total = None
    count = 0
    for p in parameters:
        if not p.requires_grad:
            continue
        if skip_1d and p.ndim <= 1:
            continue
        sq = p.pow(2).sum()
        total = sq if total is None else total + sq
        count += 1
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total / count


class DetectionLoss(nn.Module):
    r"""Combined objective ``L = L_cls + alpha * L_box + beta * L_reg``.

    Args:
        alpha: Weight on the box regression term. The committed EfficientDet
            run logged ``alpha = 50``.
        beta: Weight on the L2 penalty. ``1.0`` in the committed run.
        focal_alpha: Positive-class weight inside the focal loss.
        focal_gamma: Focusing parameter inside the focal loss.
        box_beta: Transition point of the smooth-L1 loss.

    The classification and box terms are normalised by the number of positive
    anchors, which keeps the loss scale independent of how crowded a frame is.
    """

    def __init__(
        self,
        *,
        alpha: float = 50.0,
        beta: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.box_beta = box_beta

    def forward(
        self,
        cls_logits: Tensor,
        box_deltas: Tensor,
        cls_targets: Tensor,
        box_targets: Tensor,
        positive_mask: Tensor,
        *,
        model: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        """Compute the loss terms.

        Args:
            cls_logits: ``[B, A, num_classes]`` predicted class logits.
            box_deltas: ``[B, A, 4]`` predicted box offsets.
            cls_targets: ``[B, A, num_classes]`` one-hot targets. Anchors
                ignored by the matcher should be all-zero and excluded via
                ``positive_mask``.
            box_targets: ``[B, A, 4]`` regression targets.
            positive_mask: ``[B, A]`` boolean mask of foreground anchors.
            model: If given, an L2 penalty over its weights is added.

        Returns:
            Dict with ``cls``, ``box``, ``reg`` and ``total``.
        """
        n_pos = positive_mask.sum().clamp(min=1).to(cls_logits.dtype)

        cls = (
            focal_loss(
                cls_logits,
                cls_targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )
            / n_pos
        )

        if positive_mask.any():
            box = (
                smooth_l1_loss(
                    box_deltas[positive_mask],
                    box_targets[positive_mask],
                    beta=self.box_beta,
                    reduction="sum",
                )
                / n_pos
            )
        else:
            box = box_deltas.sum() * 0.0

        reg = (
            l2_regularization(model)
            if model is not None
            else torch.zeros((), device=cls_logits.device, dtype=cls_logits.dtype)
        )

        total = cls + self.alpha * box + self.beta * reg
        return {"cls": cls, "box": box, "reg": reg, "total": total}

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, gamma={self.focal_gamma}"
