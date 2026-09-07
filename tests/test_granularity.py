"""QC tests for the spatial granularity tree."""

from __future__ import annotations

import pytest
import torch

from reid_advance.hierarchy.granularity import (
    GRANULARITY_LEVELS,
    GranularitySpec,
    flatten_level,
    level_weights_for_view,
    pyramid_features,
    spec_for_image,
    split_tokens_by_level,
)


# --------------------------------------------------------------------------
# tree geometry
# --------------------------------------------------------------------------

def test_spec_for_the_project_input_size():
    spec = spec_for_image((256, 128))
    assert (spec.rows, spec.cols) == (16, 8)
    assert spec.levels == GRANULARITY_LEVELS
    assert [spec.rows_per_region(l) for l in range(spec.depth)] == [16, 8, 4, 2]


def test_levels_must_refine_each_other():
    """Each level has to be a strict subdivision, or it is not a tree.

    Both counts divide 16 cleanly, so this isolates the refinement rule: going
    from 4 regions back to 2 is a coarsening, not a subdivision.
    """
    with pytest.raises(ValueError, match="does not refine"):
        GranularitySpec(rows=16, cols=8, levels=(4, 2)).validate()


def test_levels_must_divide_the_row_count():
    with pytest.raises(ValueError, match="does not divide evenly"):
        GranularitySpec(rows=16, cols=8, levels=(1, 5)).validate()


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def test_split_produces_one_feature_per_region():
    spec = spec_for_image((256, 128))
    tokens = torch.randn(3, spec.rows * spec.cols, 384)
    for level in range(spec.depth):
        pooled = split_tokens_by_level(tokens, spec, level)
        assert pooled.shape == (3, spec.levels[level], 384)


def test_regions_are_contiguous_horizontal_bands():
    """Region r at level l must average exactly its own rows, in order."""
    spec = spec_for_image((256, 128))
    # Give every token the value of its row index so pooling is predictable.
    rows = torch.arange(spec.rows, dtype=torch.float32)
    tokens = rows.repeat_interleave(spec.cols).reshape(1, -1, 1)

    pooled = split_tokens_by_level(tokens, spec, level=1)  # 2 regions of 8 rows
    assert pooled[0, 0, 0] == pytest.approx(rows[:8].mean())
    assert pooled[0, 1, 0] == pytest.approx(rows[8:].mean())

    pooled = split_tokens_by_level(tokens, spec, level=2)  # 4 regions of 4 rows
    for region in range(4):
        expected = rows[region * 4:(region + 1) * 4].mean()
        assert pooled[0, region, 0] == pytest.approx(expected)


def test_coarse_level_is_the_average_of_the_finer_regions():
    """The refinement property, stated numerically."""
    spec = spec_for_image((256, 128))
    tokens = torch.randn(2, spec.rows * spec.cols, 16)
    half = split_tokens_by_level(tokens, spec, level=1)
    quarter = split_tokens_by_level(tokens, spec, level=2)
    # Two quarter-regions make up one half-region.
    assert torch.allclose(half[:, 0], quarter[:, 0:2].mean(1), atol=1e-5)
    assert torch.allclose(half[:, 1], quarter[:, 2:4].mean(1), atol=1e-5)


def test_wrong_token_count_is_rejected():
    spec = spec_for_image((256, 128))
    with pytest.raises(ValueError, match="expected 128 patch tokens"):
        split_tokens_by_level(torch.randn(2, 100, 384), spec, 0)


def test_pyramid_uses_the_cls_token_for_the_whole_body_level():
    spec = spec_for_image((256, 128))
    tokens = torch.randn(2, spec.rows * spec.cols, 8)
    cls = torch.randn(2, 8)
    features = pyramid_features(tokens, spec, cls_token=cls)
    assert len(features) == spec.depth
    assert torch.allclose(features[0].squeeze(1), cls)
    assert features[-1].shape == (2, spec.levels[-1], 8)


def test_flatten_normalises_each_region_and_the_result():
    features = torch.randn(4, 4, 16) * 10.0
    flattened = flatten_level(features)
    assert flattened.shape == (4, 64)
    assert torch.allclose(flattened.norm(dim=1), torch.ones(4), atol=1e-5)


# --------------------------------------------------------------------------
# view-aware level weighting - the mechanism the hypothesis predicts
# --------------------------------------------------------------------------

def test_cross_view_pairs_are_pushed_towards_coarse_levels():
    depth = 4
    cross = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = level_weights_for_view(depth, cross)

    cross_pair = weights[0, 0]
    same_pair = weights[0, 1]
    # A cross-view pair should lean on the coarsest level more than a same-view
    # pair does, and less on the finest.
    assert cross_pair[0] > same_pair[0]
    assert cross_pair[-1] < same_pair[-1]


def test_level_weights_are_a_distribution_over_levels():
    weights = level_weights_for_view(4, torch.rand(5, 5).round())
    assert weights.shape == (5, 5, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(5, 5), atol=1e-5)
    assert (weights >= 0).all()


def test_zero_bias_makes_every_level_equal():
    """With no bias the weighting must not prefer any granularity."""
    weights = level_weights_for_view(4, torch.ones(2, 2), coarse_bias=0.0)
    assert torch.allclose(weights, torch.full((2, 2, 4), 0.25), atol=1e-6)
