r"""GradCAM++ class activation maps.

GradCAM++ (Chattopadhay et al., WACV 2018) explains a prediction by weighting
the feature maps of a chosen convolutional layer and summing them:

.. math::
    L^c_{\mathrm{GradCAM++}}(x, y) = \mathrm{ReLU}\left(\sum_k \alpha^c_k A_k(x, y)\right)

where :math:`A_k` is the ``k``-th feature map of the target layer and
:math:`\alpha^c_k` is a weight derived from the first, second and third partial
derivatives of the class score with respect to that map. For a score that is
exponential in the logit the higher derivatives collapse to powers of the
first-order gradient, which is the closed form implemented here and the one
used by standard implementations.

Why this module exists
----------------------
The paper's explainability section describes GradCAM++. The four CAM notebooks
committed in the original repository call ``EigenCAM`` from ``pytorch-grad-cam``,
which is a different method: it needs no gradients and no class score, it just
projects the activations onto their first principal component. Both are
provided here so the two can be compared directly on the same frames:

* :class:`GradCAMPlusPlus` - the method described in the paper.
* :class:`EigenCAM` - the method the notebooks ran.

Neither is tied to a particular architecture; give either one a module and a
target layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["CAMBase", "GradCAMPlusPlus", "EigenCAM", "overlay_cam"]


class CAMBase:
    """Shared plumbing: hook a layer, capture activations and gradients.

    Args:
        model: The network to explain. Put it in eval mode first.
        target_layer: The module whose output is used as :math:`A_k`. The last
            convolutional block of the neck is the usual choice for a detector:
            deeper is more semantic, shallower is sharper.

    Use as a context manager, or call :meth:`remove` when finished, so the
    forward and backward hooks do not leak.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module: nn.Module, _inp: object, out: Tensor) -> None:
        self.activations = out

    def _save_gradient(self, _module: nn.Module, _gin: object, gout: Sequence[Tensor]) -> None:
        self.gradients = gout[0]

    def remove(self) -> None:
        """Detach the hooks."""
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> CAMBase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()

    @staticmethod
    def _normalise(cam: Tensor) -> Tensor:
        """Scale each map in the batch to ``[0, 1]`` independently."""
        b = cam.shape[0]
        flat = cam.reshape(b, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        flat = (flat - lo) / (hi - lo + 1e-8)
        return flat.reshape(cam.shape)

    def _resize(self, cam: Tensor, size: tuple[int, int] | None) -> Tensor:
        if size is None:
            return cam
        return F.interpolate(
            cam.unsqueeze(1), size=size, mode="bilinear", align_corners=False
        ).squeeze(1)


class GradCAMPlusPlus(CAMBase):
    """Gradient-weighted class activation mapping, ++ variant.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        ...                       nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
        >>> cam = GradCAMPlusPlus(model.eval(), model[0])
        >>> heat = cam(torch.randn(1, 3, 32, 32), class_idx=2)
        >>> tuple(heat.shape)
        (1, 32, 32)
        >>> cam.remove()
    """

    def __call__(
        self,
        inputs: Tensor,
        *,
        class_idx: int | Tensor | None = None,
        score_fn: Callable[[object], Tensor] | None = None,
        upsample: bool = True,
        eps: float = 1e-8,
    ) -> Tensor:
        """Compute the heatmap.

        Args:
            inputs: ``[B, C, H, W]`` batch to explain.
            class_idx: Which class to explain. An ``int`` applies to the whole
                batch; a ``[B]`` tensor gives one class per image; ``None``
                uses each image's own argmax.
            score_fn: Maps the model output to a ``[B]`` scalar per image.
                Required when the model returns something other than a
                ``[B, num_classes]`` tensor. A detector returning
                ``(cls_logits, box_deltas, anchors)`` needs one, for example
                ``lambda out: out[0][..., k].amax(dim=1)``.
            upsample: Resize the map to the input's spatial size.
            eps: Numerical floor for the alpha denominator.

        Returns:
            ``[B, H, W]`` heatmap, normalised to ``[0, 1]``.
        """
        self.model.zero_grad(set_to_none=True)
        output = self.model(inputs)

        if score_fn is not None:
            score = score_fn(output)
        else:
            if not isinstance(output, Tensor) or output.ndim != 2:
                raise ValueError(
                    "model output is not [B, num_classes]; pass score_fn to reduce it to [B]"
                )
            if class_idx is None:
                idx = output.argmax(dim=1)
            elif isinstance(class_idx, int):
                idx = torch.full(
                    (output.shape[0],), class_idx, device=output.device, dtype=torch.long
                )
            else:
                idx = class_idx.to(output.device).long()
            score = output.gather(1, idx[:, None]).squeeze(1)

        score.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("target layer produced no activations or gradients")
        acts = self.activations.detach()
        grads = self.gradients.detach()

        grads2 = grads.pow(2)
        grads3 = grads2 * grads
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * grads2 + sum_acts * grads3
        denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom)
        alpha = grads2 / denom

        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1))
        cam = self._resize(cam, tuple(inputs.shape[-2:]) if upsample else None)
        return self._normalise(cam)


class EigenCAM(CAMBase):
    """Principal-component CAM - the method the committed notebooks actually ran.

    Projects the activation tensor of the target layer onto its first principal
    component. No gradients and no class score, so it shows where the layer is
    active rather than what drove a particular prediction. Included for a
    like-for-like comparison against :class:`GradCAMPlusPlus`.
    """

    def __call__(self, inputs: Tensor, *, upsample: bool = True) -> Tensor:
        """Compute the heatmap for ``[B, C, H, W]`` inputs."""
        with torch.no_grad():
            self.model(inputs)
        if self.activations is None:
            raise RuntimeError("target layer produced no activations")
        acts = self.activations.detach()
        b, k, h, w = acts.shape
        flat = acts.reshape(b, k, h * w)
        flat = flat - flat.mean(dim=2, keepdim=True)
        cams = []
        for i in range(b):
            _u, _s, vh = torch.linalg.svd(flat[i], full_matrices=False)
            cams.append(vh[0].reshape(h, w))
        cam = F.relu(torch.stack(cams, dim=0))
        cam = self._resize(cam, tuple(inputs.shape[-2:]) if upsample else None)
        return self._normalise(cam)


def overlay_cam(image: object, cam: Tensor, *, alpha: float = 0.5, colormap: int | None = None):
    """Blend a heatmap over an RGB image for display.

    Args:
        image: ``[H, W, 3]`` array - the original frame, in ``[0, 1]`` or
            ``[0, 255]``.
        cam: ``[H, W]`` heatmap in ``[0, 1]``.
        alpha: Blend weight of the heatmap.
        colormap: An OpenCV colormap id; defaults to ``COLORMAP_JET``.

    Returns:
        ``[H, W, 3]`` ``uint8`` array, ready for ``PIL.Image.fromarray``.

    Requires OpenCV, an optional extra: ``pip install 'deepseanet[viz]'``.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("overlay_cam needs opencv-python and numpy") from exc

    if colormap is None:
        colormap = cv2.COLORMAP_JET
    heat = cam.detach().cpu().numpy() if isinstance(cam, Tensor) else np.asarray(cam)
    heat = cv2.applyColorMap(np.uint8(255 * heat), colormap)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    blended = alpha * heat + (1 - alpha) * img
    return np.uint8(255 * blended / blended.max())
