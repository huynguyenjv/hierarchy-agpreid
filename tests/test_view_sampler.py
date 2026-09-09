"""QC-N1: the sampler gate.

If a batch holds no identity with images from both platforms, the cross-view
term of the multi-granularity loss never fires. Training still runs and the
loss still falls, so the failure is invisible from the curves - it would only
surface as "the method did not work", days later and for the wrong reason.
These tests measure batch composition directly rather than trusting a log line.
"""

from __future__ import annotations

import os

import pytest

from reid_advance.cargo import AERIAL_CAMERAS, GROUND_CAMERAS, binary_view_of_camera
from reid_advance.hierarchy.sampler import ViewBalancedPKSampler

CARGO_ROOT = "D:/datasets/cargo"


def synthetic_items(identities: int, aerial_each: int, ground_each: int):
    """(path, pid, camera) triples with a known view split."""
    items = []
    for pid in range(identities):
        for i in range(aerial_each):
            items.append((f"Cam1_day_{pid}_{i}.jpg", pid, 1))
        for i in range(ground_each):
            items.append((f"Cam7_day_{pid}_{i}.jpg", pid, 7))
    return items


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_rejects_odd_instances_per_identity():
    """K must be even or an identity cannot be split evenly across platforms."""
    items = synthetic_items(8, 4, 4)
    with pytest.raises(ValueError, match="even"):
        ViewBalancedPKSampler(items, batch_size=35, instances_per_identity=7)


def test_rejects_batch_not_divisible_by_k():
    items = synthetic_items(8, 4, 4)
    with pytest.raises(ValueError, match="divisible"):
        ViewBalancedPKSampler(items, batch_size=30, instances_per_identity=4)


# --------------------------------------------------------------------------
# the property the loss depends on
# --------------------------------------------------------------------------

def test_every_batch_is_half_aerial_half_ground():
    items = synthetic_items(64, aerial_each=6, ground_each=6)
    sampler = ViewBalancedPKSampler(items, batch_size=32, instances_per_identity=8)
    for report in sampler.batch_composition(4):
        assert report["aerial_images"] == 16
        assert report["ground_images"] == 16


def test_every_batch_carries_cross_platform_pairs():
    """The gate. Zero here means the cross-view term is a no-op."""
    items = synthetic_items(64, aerial_each=6, ground_each=6)
    sampler = ViewBalancedPKSampler(items, batch_size=32, instances_per_identity=8)
    for report in sampler.batch_composition(4):
        assert report["cross_platform_pairs"] > 0
        # 4 identities x 4 aerial x 4 ground
        assert report["cross_platform_pairs"] == 64


def test_single_view_identities_are_padded_and_counted():
    """An identity with no aerial images must not silently break the split."""
    items = synthetic_items(16, aerial_each=4, ground_each=4)
    items += [(f"Cam7_day_99_{i}.jpg", 99, 7) for i in range(8)]  # ground only

    sampler = ViewBalancedPKSampler(items, batch_size=16, instances_per_identity=8)
    assert 99 not in sampler.dual_view_pids
    list(iter(sampler))
    assert sampler.last_fallback_count > 0, (
        "a single-view identity should register as a fallback, not pass silently"
    )


def test_batches_are_full_and_deterministic_per_epoch():
    """Same seed and epoch reproduce; a later epoch reshuffles.

    Note ``__iter__`` advances the epoch as a side effect, so a fresh sampler
    is needed for each comparison rather than reusing one already iterated.
    """
    items = synthetic_items(32, 4, 4)

    def draw(epoch: int) -> list[int]:
        sampler = ViewBalancedPKSampler(
            items, batch_size=32, instances_per_identity=8, seed=7
        )
        sampler.set_epoch(epoch)
        return list(iter(sampler))

    assert draw(0) == draw(0), "same seed and epoch must reproduce"
    assert draw(0) != draw(1), "epochs must reshuffle"


def test_indices_stay_in_range():
    items = synthetic_items(32, 4, 4)
    sampler = ViewBalancedPKSampler(items, batch_size=32, instances_per_identity=8)
    indices = list(iter(sampler))
    assert len(indices) == len(sampler)
    assert all(0 <= i < len(items) for i in indices)


# --------------------------------------------------------------------------
# against the real dataset
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isdir(CARGO_ROOT), reason="CARGO not downloaded")
def test_gate_on_real_cargo_batches():
    """The measurement that licensed writing the loss.

    Observed at P=16, K=8: every batch is 64 aerial / 64 ground, all 16
    identities carry both platforms, and each batch offers 256 cross-platform
    pairs.
    """
    from reid_advance.cargo import scan_cargo_images

    items = scan_cargo_images(os.path.join(CARGO_ROOT, "train"))
    sampler = ViewBalancedPKSampler(items, batch_size=128, instances_per_identity=8)

    for report in sampler.batch_composition(3):
        assert report["aerial_images"] == 64
        assert report["ground_images"] == 64
        assert report["dual_view_identities"] == 16
        assert report["cross_platform_pairs"] == 256


def test_view_mapping_matches_cargo_camera_ranges():
    assert all(binary_view_of_camera(c) == 1 for c in AERIAL_CAMERAS)
    assert all(binary_view_of_camera(c) == 0 for c in GROUND_CAMERAS)
