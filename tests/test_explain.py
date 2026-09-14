"""Class activation maps."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from deepseanet.explain import EigenCAM, GradCAMPlusPlus


@pytest.fixture
def classifier() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 4),
    ).eval()


def test_gradcampp_returns_a_normalised_map_at_input_resolution(classifier):
    with GradCAMPlusPlus(classifier, classifier[2]) as cam:
        heat = cam(torch.randn(2, 3, 32, 32), class_idx=1)
    assert heat.shape == (2, 32, 32)
    assert float(heat.min()) >= 0.0
    assert float(heat.max()) == pytest.approx(1.0, abs=1e-5)


def test_gradcampp_responds_to_the_chosen_class(classifier):
    x = torch.randn(1, 3, 32, 32)
    with GradCAMPlusPlus(classifier, classifier[2]) as cam:
        a = cam(x, class_idx=0)
        b = cam(x, class_idx=3)
    assert not torch.allclose(a, b, atol=1e-4)


def test_gradcampp_accepts_a_custom_score_function():
    """Detectors return tuples, so the score has to be reduced explicitly."""

    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Conv2d(3, 6, 3, padding=1)
            self.head = nn.Conv2d(6, 5, 1)

        def forward(self, x):
            feat = self.body(x)
            logits = self.head(feat).flatten(2).transpose(1, 2)
            return logits, torch.zeros_like(logits[..., :4]), torch.zeros(1, 4)

    model = Detector().eval()
    with GradCAMPlusPlus(model, model.body) as cam:
        heat = cam(torch.randn(1, 3, 16, 16), score_fn=lambda out: out[0][..., 2].amax(dim=1))
    assert heat.shape == (1, 16, 16)


def test_gradcampp_requires_a_score_fn_for_non_classifier_output():
    model = nn.Conv2d(3, 4, 3, padding=1).eval()
    with GradCAMPlusPlus(model, model) as cam, pytest.raises(ValueError, match="pass score_fn"):
        cam(torch.randn(1, 3, 8, 8))


def test_eigencam_needs_no_gradients(classifier):
    with EigenCAM(classifier, classifier[2]) as cam:
        heat = cam(torch.randn(1, 3, 16, 16))
    assert heat.shape == (1, 16, 16)
    assert float(heat.max()) == pytest.approx(1.0, abs=1e-5)


def test_hooks_are_removed_on_exit(classifier):
    cam = GradCAMPlusPlus(classifier, classifier[0])
    cam.remove()
    assert cam._handles == []
