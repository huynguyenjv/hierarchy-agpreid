"""QC tests for the View Stability Score helpers (AI-03)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reid_advance.attributes import ATTRIBUTE_CLASS_COUNTS, ATTRIBUTE_IGNORE_INDEX, ATTRIBUTE_NAMES
from reid_advance.hierarchy.vss import (
    AttributeHeads,
    attribute_ce_loss,
    balanced_accuracy,
    score_attribute,
)


# --------------------------------------------------------------------------
# balanced accuracy - the metric the tree ordering is read from
# --------------------------------------------------------------------------

def test_balanced_accuracy_equals_accuracy_when_classes_are_balanced():
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    assert balanced_accuracy(predictions, targets) == pytest.approx(0.75)
    assert (predictions == targets).mean() == pytest.approx(0.75)


def test_balanced_accuracy_punishes_majority_class_guessing():
    """The reason VSS uses balanced accuracy rather than raw accuracy.

    Nine of ten samples are class 0, so always predicting 0 scores 90% raw but
    only 50% balanced - it never gets the minority class right.
    """
    targets = np.array([0] * 9 + [1])
    predictions = np.zeros(10, dtype=np.int64)
    assert (predictions == targets).mean() == pytest.approx(0.9)
    assert balanced_accuracy(predictions, targets) == pytest.approx(0.5)


def test_balanced_accuracy_ignores_classes_absent_from_targets():
    targets = np.array([0, 0, 2, 2])
    predictions = np.array([0, 0, 2, 2])
    assert balanced_accuracy(predictions, targets) == pytest.approx(1.0)


def test_balanced_accuracy_of_empty_input_is_zero():
    empty = np.array([], dtype=np.int64)
    assert balanced_accuracy(empty, empty) == 0.0


# --------------------------------------------------------------------------
# score assembly
# --------------------------------------------------------------------------

def test_retention_is_the_aerial_over_ground_ratio():
    score = score_attribute(
        "gender",
        ground_predictions=np.array([0, 1, 0, 1]),
        ground_targets=np.array([0, 1, 0, 1]),          # perfect on ground
        aerial_predictions=np.array([0, 1, 0, 0]),
        aerial_targets=np.array([0, 1, 0, 1]),          # 0.75 balanced on aerial
    )
    assert score.ground_balanced_accuracy == pytest.approx(1.0)
    assert score.aerial_balanced_accuracy == pytest.approx(0.75)
    assert score.retention == pytest.approx(0.75)
    assert score.vss == score.aerial_balanced_accuracy


def test_lift_over_majority_is_negative_when_the_model_beats_nothing():
    """A high raw accuracy on a skewed group must not look like a good level."""
    targets = np.array([0] * 9 + [1])
    score = score_attribute(
        "age",
        ground_predictions=targets, ground_targets=targets,
        aerial_predictions=np.zeros(10, dtype=np.int64), aerial_targets=targets,
    )
    assert score.aerial_accuracy == pytest.approx(0.9)     # looks great
    assert score.majority_baseline == pytest.approx(0.9)   # but so does guessing
    assert score.lift_over_majority == pytest.approx(0.0)
    assert score.vss == pytest.approx(0.5)                 # VSS is not fooled


def test_retention_of_a_useless_ground_model_is_zero_not_nan():
    empty = np.array([], dtype=np.int64)
    score = score_attribute("bag", empty, empty, empty, empty)
    assert score.retention == 0.0


# --------------------------------------------------------------------------
# heads and masked loss
# --------------------------------------------------------------------------

def test_heads_emit_one_logit_tensor_per_attribute_group():
    heads = AttributeHeads(embed_dim=32)
    logits = heads(torch.randn(4, 32))
    assert set(logits) == set(ATTRIBUTE_NAMES)
    for name, tensor in logits.items():
        assert tensor.shape == (4, ATTRIBUTE_CLASS_COUNTS[name])


def test_loss_skips_rows_marked_unknown():
    """Unknown rows must not contribute; they are absent labels, not a class."""
    torch.manual_seed(0)
    heads = AttributeHeads(embed_dim=16)
    logits = heads(torch.randn(4, 16))

    targets = torch.zeros(4, len(ATTRIBUTE_NAMES), dtype=torch.long)
    targets[:] = ATTRIBUTE_IGNORE_INDEX
    targets[0, 0] = 0
    targets[1, 0] = 1

    loss, valid = attribute_ce_loss(logits, targets)
    assert valid == 2
    assert torch.isfinite(loss)


def test_loss_is_finite_and_zero_when_everything_is_unknown():
    heads = AttributeHeads(embed_dim=16)
    logits = heads(torch.randn(4, 16))
    targets = torch.full((4, len(ATTRIBUTE_NAMES)), ATTRIBUTE_IGNORE_INDEX, dtype=torch.long)

    loss, valid = attribute_ce_loss(logits, targets)
    assert valid == 0
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0)


def test_loss_backpropagates_through_the_shared_bottleneck():
    heads = AttributeHeads(embed_dim=16)
    features = torch.randn(4, 16, requires_grad=True)
    targets = torch.full((4, len(ATTRIBUTE_NAMES)), ATTRIBUTE_IGNORE_INDEX, dtype=torch.long)
    targets[:, 0] = torch.tensor([0, 1, 0, 1])

    loss, _ = attribute_ce_loss(heads(features), targets)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
