"""End-to-end detector behaviour."""

from __future__ import annotations

import pytest
import torch

from deepseanet.models import DeepSeaNet, DeepSeaNetConfig, Swish, swish, swish_derivative
from deepseanet.models.head import AnchorGenerator


@pytest.mark.parametrize("neck", ["bifpn", "biskfpn"])
def test_detector_output_contract(neck):
    model = DeepSeaNet(DeepSeaNetConfig(num_classes=6, neck=neck, image_size=128)).eval()
    with torch.no_grad():
        cls_logits, box_deltas, anchors = model(torch.randn(2, 3, 128, 128))
    assert cls_logits.shape[0] == 2
    assert cls_logits.shape[-1] == 6
    assert box_deltas.shape[-1] == 4
    assert cls_logits.shape[1] == box_deltas.shape[1] == anchors.shape[0]


def test_detector_trains_one_step():
    model = DeepSeaNet(DeepSeaNetConfig(num_classes=3, image_size=128))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    cls_logits, box_deltas, _ = model(torch.randn(2, 3, 128, 128))
    loss = cls_logits.square().mean() + box_deltas.square().mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_config_fills_in_efficientdet_scaling_defaults():
    cfg = DeepSeaNetConfig(phi=2).resolved()
    assert cfg.neck_channels == int(64 * 1.35**2)
    assert cfg.neck_layers == 5
    assert cfg.head_layers is not None


def test_parameter_breakdown_sums_to_total():
    model = DeepSeaNet(DeepSeaNetConfig(num_classes=6, image_size=128))
    assert sum(model.parameter_breakdown().values()) == model.num_parameters(trainable_only=False)


def test_anchors_are_valid_boxes():
    gen = AnchorGenerator(strides=(8, 16), aspect_ratios=(0.5, 1.0, 2.0), scales=(1.0, 1.26))
    feats = [torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, 4, 4)]
    anchors = gen(feats)
    assert anchors.shape == (8 * 8 * 6 + 4 * 4 * 6, 4)
    assert (anchors[:, 2] > anchors[:, 0]).all()
    assert (anchors[:, 3] > anchors[:, 1]).all()


def test_detector_rejects_an_input_it_cannot_downsample():
    model = DeepSeaNet(DeepSeaNetConfig(num_classes=6, image_size=128)).eval()
    with pytest.raises(ValueError, match="divisible by 128"):
        model(torch.randn(1, 3, 100, 100))


def test_swish_matches_analytic_derivative():
    x = torch.linspace(-4, 4, 64, requires_grad=True)
    swish(x, 1.0).sum().backward()
    assert torch.allclose(x.grad, swish_derivative(x.detach(), 1.0), atol=1e-6)


def test_swish_beta_is_learnable_when_requested():
    act = Swish(beta=0.5, trainable=True)
    act(torch.randn(4)).sum().backward()
    assert act.beta.grad is not None
