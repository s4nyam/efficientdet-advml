"""BiFPN and BiSkFPN: shapes, gradients and the properties that matter."""

from __future__ import annotations

import pytest
import torch

from deepseanet.models.biskfpn import (
    BiFPN,
    BiSkFPN,
    BiSkFPNLayer,
    WeightedFusion,
    build_neck,
)


@pytest.mark.parametrize("kind", ["bifpn", "biskfpn"])
def test_neck_preserves_pyramid_geometry(pyramid, kind):
    neck = build_neck(kind, [f.shape[1] for f in pyramid], channels=32, n_layers=2)
    out = neck(pyramid)
    assert len(out) == len(pyramid)
    for got, ref in zip(out, pyramid):
        assert got.shape[0] == ref.shape[0]
        assert got.shape[1] == 32, "every level is projected to the neck width"
        assert got.shape[-2:] == ref.shape[-2:], "spatial size must be unchanged"


@pytest.mark.parametrize("kind", ["bifpn", "biskfpn"])
def test_neck_is_differentiable_everywhere(pyramid, kind):
    neck = build_neck(kind, [f.shape[1] for f in pyramid], channels=16, n_layers=1)
    sum(o.square().mean() for o in neck(pyramid)).backward()
    missing = [n for n, p in neck.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_biskfpn_has_more_capacity_than_bifpn(pyramid):
    shapes = [f.shape[1] for f in pyramid]
    bifpn = BiFPN(shapes, channels=32, n_layers=2)
    biskfpn = BiSkFPN(shapes, channels=32, n_layers=2)
    n_bi = sum(p.numel() for p in bifpn.parameters())
    n_sk = sum(p.numel() for p in biskfpn.parameters())
    assert n_sk > n_bi, "the deconv and skip branches add parameters"


def test_biskfpn_finest_level_depends_on_low_level_input(pyramid):
    """The skip path must actually carry information, not be decorative."""
    layer = BiSkFPNLayer(channels=8, n_levels=3).eval()
    feats = [torch.randn(1, 8, 16, 16), torch.randn(1, 8, 8, 8), torch.randn(1, 8, 4, 4)]
    base = layer(feats)

    perturbed = [feats[0] + 5.0, feats[1], feats[2]]
    changed = layer(perturbed)
    # Level 1's skip branch reads level 0, so it must respond.
    assert not torch.allclose(base[1], changed[1], atol=1e-5)


def test_weighted_fusion_normalises_to_a_convex_combination():
    fuse = WeightedFusion(3)
    ones = [torch.ones(1, 2, 2, 2) for _ in range(3)]
    out = fuse(ones)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-3)


def test_weighted_fusion_rejects_wrong_input_count():
    fuse = WeightedFusion(2)
    with pytest.raises(ValueError, match="expects 2 inputs"):
        fuse([torch.zeros(1, 1, 1, 1)])


def test_build_neck_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown neck"):
        build_neck("fpn", [8, 8], channels=4)  # type: ignore[arg-type]
