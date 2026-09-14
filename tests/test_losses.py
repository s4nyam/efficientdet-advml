"""Loss terms."""

from __future__ import annotations

import math

import pytest
import torch

from deepseanet.losses import DetectionLoss, focal_loss, l2_regularization, smooth_l1_loss


def test_focal_loss_downweights_easy_examples():
    """A confident correct prediction should cost far less under focal loss."""
    logits = torch.tensor([[6.0]])
    target = torch.ones(1, 1)
    hard = focal_loss(torch.tensor([[0.0]]), target, gamma=2.0)
    easy = focal_loss(logits, target, gamma=2.0)
    assert easy < hard / 100


def test_focal_loss_with_gamma_zero_is_weighted_bce():
    logits = torch.randn(4, 3)
    target = (torch.rand(4, 3) > 0.5).float()
    got = focal_loss(logits, target, alpha=0.5, gamma=0.0, reduction="sum")
    expected = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="sum"
    )
    assert torch.allclose(got, expected, atol=1e-5)


def test_focal_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        focal_loss(torch.randn(2, 3), torch.randn(2, 4))


def test_smooth_l1_is_quadratic_near_zero_and_linear_far_out():
    small = smooth_l1_loss(torch.tensor([0.5]), torch.tensor([0.0]), beta=1.0, reduction="sum")
    large = smooth_l1_loss(torch.tensor([10.0]), torch.tensor([0.0]), beta=1.0, reduction="sum")
    assert math.isclose(float(small), 0.125, rel_tol=1e-6)
    assert math.isclose(float(large), 9.5, rel_tol=1e-6)


def test_l2_regularization_skips_biases_and_norms():
    layer = torch.nn.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(100.0)
    # 16 weights of 1.0 -> sum 16, averaged over 1 qualifying tensor.
    assert math.isclose(float(l2_regularization(layer)), 16.0, rel_tol=1e-6)


def test_detection_loss_combines_terms_with_alpha_and_beta():
    torch.manual_seed(0)
    b, a, k = 2, 20, 6
    crit = DetectionLoss(alpha=10.0, beta=0.0)
    mask = torch.zeros(b, a, dtype=torch.bool)
    mask[:, :4] = True
    cls_t = torch.zeros(b, a, k)
    cls_t[mask] = torch.eye(k)[torch.randint(0, k, (int(mask.sum()),))]
    out = crit(torch.randn(b, a, k), torch.randn(b, a, 4), cls_t, torch.randn(b, a, 4), mask)
    assert torch.allclose(out["total"], out["cls"] + 10.0 * out["box"], atol=1e-5)
    assert torch.isfinite(out["total"])


def test_detection_loss_survives_a_frame_with_no_objects():
    crit = DetectionLoss()
    b, a, k = 1, 8, 6
    out = crit(
        torch.randn(b, a, k),
        torch.randn(b, a, 4),
        torch.zeros(b, a, k),
        torch.zeros(b, a, 4),
        torch.zeros(b, a, dtype=torch.bool),
    )
    assert torch.isfinite(out["total"])
    assert float(out["box"]) == 0.0
