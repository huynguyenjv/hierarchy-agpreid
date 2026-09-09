"""QC-N2: unit tests for the multi-granularity loss.

Written alongside the loss rather than after it. The case that matters most is
``test_uniform_beta_equals_mean_of_level_losses``: if the uniform-beta branch is
not exactly the average of the four per-level SupCon terms, then it is some
hybrid rather than a plain multi-scale baseline, and the whole view-aware
versus uniform contrast loses its meaning.
"""

from __future__ import annotations

import pytest
import torch

from reid_advance.hierarchy.granularity import spec_for_image
from reid_advance.hierarchy.losses import MultiGranularityLoss, supcon_level

EMBED = 16


def fake_batch(identities: int = 4, per_identity: int = 4, seed: int = 0):
    """Tokens, labels and views for a synthetic P x K batch, half each view."""
    torch.manual_seed(seed)
    spec = spec_for_image((256, 128))
    batch = identities * per_identity
    tokens = torch.randn(batch, spec.rows * spec.cols, EMBED)
    cls = torch.randn(batch, EMBED)
    labels = torch.arange(identities).repeat_interleave(per_identity)
    # Alternate views inside each identity so cross-platform positives exist.
    views = torch.tensor([i % 2 for i in range(per_identity)]).repeat(identities)
    return spec, tokens, cls, labels, views


# --------------------------------------------------------------------------
# supcon_level
# --------------------------------------------------------------------------

def test_supcon_is_lower_when_positives_are_already_together():
    """Sanity: the loss must reward the arrangement it is meant to encourage."""
    labels = torch.tensor([0, 0, 1, 1])
    close = torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0], [-0.99, 0.14]])
    close = torch.nn.functional.normalize(close, dim=1)
    scattered = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    assert supcon_level(close, labels) < supcon_level(scattered, labels)


def test_anchor_without_a_positive_is_skipped_not_nan():
    labels = torch.tensor([0, 1, 2, 3])          # every identity is a singleton
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    loss = supcon_level(embeddings, labels)
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0)


def test_pair_weight_of_zero_ignores_that_positive():
    labels = torch.tensor([0, 0, 0, 1])
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)

    weight = torch.ones(4, 4)
    weight[0, 1] = 0.0
    weight[1, 0] = 0.0

    assert supcon_level(embeddings, labels, weight) != supcon_level(embeddings, labels)


def test_uniform_pair_weight_matches_no_weight():
    labels = torch.tensor([0, 0, 1, 1])
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    plain = supcon_level(embeddings, labels)
    weighted = supcon_level(embeddings, labels, torch.full((4, 4), 0.25))
    # Normalisation divides the weights out, so a constant weight changes nothing.
    assert float(plain) == pytest.approx(float(weighted), abs=1e-5)


# --------------------------------------------------------------------------
# the control branch
# --------------------------------------------------------------------------

def test_uniform_beta_equals_mean_of_level_losses():
    """The case the whole comparison rests on.

    With coarse_bias=0 every level gets 0.25 for every pair, so L_MG must be
    exactly the sum of four equally-weighted SupCon terms - a plain multi-scale
    baseline, not a hybrid.
    """
    spec, tokens, cls, labels, views = fake_batch()
    loss_fn = MultiGranularityLoss(spec, EMBED, coarse_bias=0.0)
    loss_fn.eval()          # freeze BatchNorm so both paths see identical stats

    total, breakdown = loss_fn(tokens, labels, views, cls)

    embeddings = loss_fn.level_embeddings(tokens, cls)
    expected = sum(
        supcon_level(z, labels, torch.full((len(labels), len(labels)), 0.25))
        for z in embeddings
    )
    assert float(total) == pytest.approx(float(expected), rel=1e-5)
    assert len(breakdown) == spec.depth


def test_uniform_beta_is_view_blind():
    """Swapping which samples are aerial must not change the uniform branch."""
    spec, tokens, cls, labels, views = fake_batch()
    loss_fn = MultiGranularityLoss(spec, EMBED, coarse_bias=0.0)
    loss_fn.eval()

    a, _ = loss_fn(tokens, labels, views, cls)
    b, _ = loss_fn(tokens, labels, 1 - views, cls)
    assert float(a) == pytest.approx(float(b), rel=1e-6)


def test_view_aware_branch_is_not_view_blind():
    """And the view-aware branch must be, or it is testing nothing."""
    spec, tokens, cls, labels, views = fake_batch()
    loss_fn = MultiGranularityLoss(spec, EMBED, coarse_bias=2.0)
    loss_fn.eval()

    a, _ = loss_fn(tokens, labels, views, cls)
    all_same = torch.zeros_like(views)
    b, _ = loss_fn(tokens, labels, all_same, cls)
    assert float(a) != pytest.approx(float(b), rel=1e-6)


# --------------------------------------------------------------------------
# level weighting
# --------------------------------------------------------------------------

def test_beta_reaches_the_documented_values():
    """The numbers recorded in the plan, so a change to them is deliberate."""
    from reid_advance.hierarchy.granularity import level_weights_for_view

    cross = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = level_weights_for_view(4, cross, coarse_bias=2.0)

    assert weights[0, 0].tolist() == pytest.approx(
        [0.523, 0.268, 0.138, 0.071], abs=1e-3
    )
    assert weights[0, 1].tolist() == pytest.approx(
        [0.071, 0.138, 0.268, 0.523], abs=1e-3
    )


def test_beta_is_a_distribution_per_pair():
    from reid_advance.hierarchy.granularity import level_weights_for_view

    weights = level_weights_for_view(4, torch.rand(6, 6).round(), coarse_bias=2.0)
    assert torch.allclose(weights.sum(-1), torch.ones(6, 6), atol=1e-5)
    assert (weights >= 0).all()


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------

def test_gradient_reaches_every_level():
    spec, tokens, cls, labels, views = fake_batch()
    tokens.requires_grad_(True)
    loss_fn = MultiGranularityLoss(spec, EMBED, coarse_bias=2.0)

    total, _ = loss_fn(tokens, labels, views, cls)
    total.backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad.abs().sum() > 0


def test_finite_with_collapsed_input():
    """Raw ViT tokens are near-identical; the loss must not produce NaN even so."""
    spec, tokens, cls, labels, views = fake_batch()
    collapsed = torch.ones_like(tokens) * 50.0 + 0.01 * tokens
    loss_fn = MultiGranularityLoss(spec, EMBED, coarse_bias=2.0)

    total, breakdown = loss_fn(collapsed, labels, views, cls)
    assert torch.isfinite(total)
    assert all(v == v for v in breakdown.values())
