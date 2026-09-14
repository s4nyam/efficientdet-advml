"""Universal adversarial perturbations and the training curriculum."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from deepseanet.attacks import apply_uap, curriculum_epsilon, optimize_uap, project, random_uap


def test_random_uap_respects_the_amplitude_budget():
    delta = random_uap((3, 16, 16), epsilon=8 / 255)
    assert delta.shape == (3, 16, 16)
    assert torch.allclose(delta.abs(), torch.full_like(delta, 8 / 255), atol=1e-7)


def test_random_uap_is_reproducible_from_a_generator():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert torch.equal(random_uap((3, 8, 8), generator=g1), random_uap((3, 8, 8), generator=g2))


def test_apply_uap_clamps_to_the_valid_pixel_range():
    images = torch.full((2, 3, 8, 8), 0.99)
    out = apply_uap(images, torch.full((3, 8, 8), 0.5))
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0


def test_apply_uap_resizes_a_pattern_to_the_batch():
    out = apply_uap(torch.zeros(1, 3, 32, 32), torch.zeros(3, 8, 8))
    assert out.shape == (1, 3, 32, 32)


def test_project_linf_clips_and_l2_rescales():
    d = torch.full((3, 4, 4), 1.0)
    assert float(project(d, 0.1, "linf").abs().max()) == pytest.approx(0.1)
    assert float(project(d, 1.0, "l2").norm()) == pytest.approx(1.0, abs=1e-5)


def test_optimize_uap_finds_a_stronger_pattern_than_noise():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Flatten(), nn.Linear(4 * 64, 2)).eval()
    batches = [torch.rand(4, 3, 8, 8) for _ in range(2)]

    def loss_fn(m, x):
        return m(x).abs().mean()

    delta = optimize_uap(model, batches, loss_fn, epsilon=0.05, steps=5)
    assert delta.shape == (3, 8, 8)
    assert float(delta.abs().max()) <= 0.05 + 1e-6

    learned = sum(float(loss_fn(model, apply_uap(b, delta))) for b in batches)
    noise = sum(float(loss_fn(model, apply_uap(b, random_uap((3, 8, 8), 0.05)))) for b in batches)
    assert learned >= noise


def test_optimize_uap_restores_model_state():
    model = nn.Sequential(nn.Conv2d(3, 2, 3, padding=1), nn.Flatten(), nn.Linear(2 * 16, 2))
    model.train()
    optimize_uap(model, [torch.rand(2, 3, 4, 4)], lambda m, x: m(x).mean(), steps=1)
    assert model.training, "training mode must be restored"
    assert all(p.requires_grad for p in model.parameters()), "grads must be re-enabled"


def test_optimize_uap_rejects_empty_input():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="at least one batch"):
        optimize_uap(model, [], lambda m, x: m(x).mean())


def test_curriculum_stays_clean_during_warmup_then_ramps():
    assert curriculum_epsilon(0, 100, warmup_frac=0.2) == 0.0
    assert curriculum_epsilon(19, 100, warmup_frac=0.2) == 0.0
    mid = curriculum_epsilon(60, 100, max_epsilon=0.1, warmup_frac=0.2)
    assert 0.0 < mid < 0.1
    assert curriculum_epsilon(100, 100, max_epsilon=0.1, warmup_frac=0.2) == pytest.approx(0.1)


@pytest.mark.parametrize("schedule", ["linear", "cosine", "step"])
def test_curriculum_is_monotone(schedule):
    values = [curriculum_epsilon(e, 50, max_epsilon=0.1, schedule=schedule) for e in range(50)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
