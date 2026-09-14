r"""Universal adversarial perturbations and the curriculum used to train against them.

A universal adversarial perturbation (UAP, Moosavi-Dezfooli et al., CVPR 2017)
is a *single* image-agnostic pattern :math:`\delta` that, added to almost any
input, pushes a network towards wrong predictions while staying small enough to
be nearly invisible. That is what makes it interesting for underwater vision:
the paper's argument is that murky water behaves like a naturally occurring
perturbation, so a model hardened against a crafted universal pattern should
also cope better with silt, backscatter and sensor noise.

Two constructions are provided.

:func:`random_uap`
    The closed form printed in the paper,
    :math:`U = \xi \cdot \mathrm{sign}\left(\sum_i r_i \cdot \delta_i\right)`,
    which is a fixed random sign pattern at amplitude :math:`\xi`. It is cheap,
    model-independent, and good enough to use as a noise augmentation.

:func:`optimize_uap`
    The actual attack: gradient ascent on a batch of real frames, projected
    back onto the :math:`\ell_\infty` (or :math:`\ell_2`) ball after every
    step. This is what you want when measuring robustness, because a random
    pattern is a weak adversary and will flatter the model.

:func:`curriculum_epsilon` schedules the perturbation strength across training,
which is the "curriculum fashion" the paper describes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Literal

import torch
from torch import Tensor, nn

__all__ = [
    "random_uap",
    "optimize_uap",
    "apply_uap",
    "curriculum_epsilon",
    "project",
]


def project(delta: Tensor, epsilon: float, norm: Literal["linf", "l2"] = "linf") -> Tensor:
    """Project ``delta`` back onto the ``epsilon`` ball of the given norm."""
    if norm == "linf":
        return delta.clamp(-epsilon, epsilon)
    if norm == "l2":
        flat = delta.flatten()
        n = flat.norm(p=2)
        if float(n) <= epsilon:
            return delta
        return delta * (epsilon / (n + 1e-12))
    raise ValueError(f"unknown norm {norm!r}")


def random_uap(
    shape: tuple[int, int, int],
    epsilon: float = 8 / 255,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""The paper's closed-form UAP: a signed random pattern at amplitude ``epsilon``.

    Args:
        shape: ``(C, H, W)`` of the perturbation.
        epsilon: Amplitude :math:`\xi`, in the same units as the image. For
            images in ``[0, 1]``, ``8/255`` is the usual "imperceptible" budget.
        generator: Optional RNG, so the same universal pattern can be
            regenerated exactly.
        device: Device to build the tensor on.
        dtype: Tensor dtype.

    Returns:
        ``[C, H, W]`` perturbation with every element equal to
        ``+epsilon`` or ``-epsilon``.
    """
    r = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    d = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return epsilon * torch.sign(r * d)


def apply_uap(
    images: Tensor, delta: Tensor, *, clamp: tuple[float, float] | None = (0.0, 1.0)
) -> Tensor:
    """Add a universal perturbation to a batch and clamp back to a valid range.

    Args:
        images: ``[B, C, H, W]`` batch.
        delta: ``[C, H, W]`` or ``[1, C, H, W]`` perturbation. It is resized to
            the batch's spatial size if needed, so one pattern can be reused
            across resolutions.
        clamp: Valid pixel range, or ``None`` to skip clamping.

    Returns:
        The perturbed batch.
    """
    if delta.ndim == 3:
        delta = delta.unsqueeze(0)
    if delta.shape[-2:] != images.shape[-2:]:
        delta = torch.nn.functional.interpolate(
            delta, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
    out = images + delta.to(images.device, images.dtype)
    if clamp is not None:
        out = out.clamp(*clamp)
    return out


def optimize_uap(
    model: nn.Module,
    batches: Iterable[Tensor],
    loss_fn: Callable[[nn.Module, Tensor], Tensor],
    *,
    epsilon: float = 8 / 255,
    steps: int = 10,
    step_size: float | None = None,
    norm: Literal["linf", "l2"] = "linf",
    image_shape: tuple[int, int, int] | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Learn a universal perturbation by gradient ascent on a set of batches.

    The perturbation is shared across every image: one tensor accumulates
    gradients from all of them, so it has to find structure that hurts the
    model generally rather than exploiting a single frame.

    Args:
        model: The detector under attack. Kept in eval mode and frozen.
        batches: Iterable of image batches ``[B, C, H, W]``. It is consumed
            once per step, so pass a list (or something re-iterable) rather
            than a one-shot generator.
        loss_fn: ``(model, perturbed_images) -> scalar``. Ascending this drives
            the attack; a detection loss with fixed targets, or the negative
            max class score, both work.
        epsilon: Radius of the ball the perturbation is confined to.
        steps: Number of passes over ``batches``.
        step_size: Ascent step. Defaults to ``2.5 * epsilon / steps``, the
            standard PGD heuristic.
        norm: Which ball to project onto.
        image_shape: ``(C, H, W)`` of the perturbation. Inferred from the first
            batch when omitted.
        device: Device to run on. Inferred from ``model`` when omitted.

    Returns:
        ``[C, H, W]`` perturbation. Feed it to :func:`apply_uap`.
    """
    batches = list(batches)
    if not batches:
        raise ValueError("optimize_uap needs at least one batch")
    if device is None:
        device = next(model.parameters()).device
    if image_shape is None:
        image_shape = tuple(batches[0].shape[1:])  # type: ignore[assignment]
    if step_size is None:
        step_size = 2.5 * epsilon / max(1, steps)

    was_training = model.training
    model.eval()
    frozen = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    delta = torch.zeros(image_shape, device=device, requires_grad=True)
    try:
        for _ in range(steps):
            for images in batches:
                images = images.to(device)
                perturbed = apply_uap(images, delta, clamp=(0.0, 1.0))
                loss = loss_fn(model, perturbed)
                grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                with torch.no_grad():
                    if norm == "linf":
                        delta += step_size * grad.sign()
                    else:
                        delta += step_size * grad / (grad.norm(p=2) + 1e-12)
                    delta.copy_(project(delta, epsilon, norm))
    finally:
        for p, req in zip(model.parameters(), frozen):
            p.requires_grad_(req)
        model.train(was_training)

    return delta.detach()


def curriculum_epsilon(
    epoch: int,
    total_epochs: int,
    *,
    max_epsilon: float = 8 / 255,
    warmup_frac: float = 0.2,
    schedule: Literal["linear", "cosine", "step"] = "linear",
) -> float:
    """Perturbation strength for a given epoch of adversarial training.

    Training on the full-strength perturbation from epoch zero tends to stall:
    the model never gets a clean signal to learn the task from. The curriculum
    keeps the first ``warmup_frac`` of training clean, then ramps up.

    Args:
        epoch: Current epoch, zero-based.
        total_epochs: Total number of epochs.
        max_epsilon: Final perturbation strength.
        warmup_frac: Fraction of training kept perturbation-free.
        schedule: Ramp shape after warmup.

    Returns:
        The epsilon to use this epoch.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    warmup = warmup_frac * total_epochs
    if epoch < warmup:
        return 0.0
    t = (epoch - warmup) / max(1e-9, total_epochs - warmup)
    t = min(max(t, 0.0), 1.0)
    if schedule == "linear":
        factor = t
    elif schedule == "cosine":
        import math

        factor = 0.5 * (1 - math.cos(math.pi * t))
    elif schedule == "step":
        factor = 0.0 if t < 0.33 else (0.5 if t < 0.66 else 1.0)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return max_epsilon * factor
