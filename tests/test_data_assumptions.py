"""QC-02: guard the dataset assumptions the whole topic rests on.

These are silent-failure risks: if any of them breaks, training still runs and
the loss still decreases, but the hierarchy stops meaning what the paper claims
it means.
"""

from __future__ import annotations

import os
from collections import defaultdict

import pytest
import torch

from reid_advance.data import parse_camera_id, track_key
from reid_advance.hierarchy import (
    CAMERA_TO_PLATFORM,
    UnknownCameraError,
    binary_view_of_path,
    binary_view_tensor,
    platform_of_camera,
    platform_of_path,
)
from reid_advance.identity import parse_identity_id

pytestmark = pytest.mark.dataset


# --------------------------------------------------------------------------
# view mapping (BA-01)
# --------------------------------------------------------------------------

def test_camera_platform_table_matches_protocol_evidence():
    """C0=aerial, C2=wearable, C3=CCTV, verified against the protocol files."""
    assert CAMERA_TO_PLATFORM == {0: "aerial", 2: "wearable", 3: "cctv"}


def test_ground_is_the_union_of_wearable_and_cctv():
    """beta_cross keys off this binary split, so pin the convention down."""
    assert binary_view_of_path("x/P0313T02220A0C0F11011.jpg") == 1
    assert binary_view_of_path("x/P0313T02220A0C2F11011.jpg") == 0
    assert binary_view_of_path("x/P0313T02220A0C3F11011.jpg") == 0


def test_unknown_camera_raises_instead_of_defaulting_to_ground():
    """Silently mapping an unknown camera to ground would make beta_cross a no-op."""
    with pytest.raises(UnknownCameraError):
        platform_of_camera(1)
    with pytest.raises(UnknownCameraError):
        binary_view_tensor(torch.tensor([0, 2, 1]))


def test_every_split_only_contains_known_cameras(data_root):
    for split in ("train_all", "query", "gallery"):
        directory = os.path.join(data_root, split)
        if not os.path.isdir(directory):
            pytest.skip(f"{split} not available")
        seen = set()
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    seen.add(parse_camera_id(name))
        assert seen <= set(CAMERA_TO_PLATFORM), (
            f"{split} has camera ids outside the known set: "
            f"{sorted(seen - set(CAMERA_TO_PLATFORM))}"
        )


# --------------------------------------------------------------------------
# attribute assumptions (BA-02)
# --------------------------------------------------------------------------

def test_every_train_identity_resolves_to_a_mat_row(train_items, track_attributes):
    unresolved = {
        pid for path, pid in train_items if track_key(path) not in track_attributes
    }
    assert not unresolved, (
        f"{len(unresolved)} training identities have no attribute row; "
        "the tree cannot assign them a branch"
    )


def test_attributes_are_constant_within_an_identity(train_items, track_attributes):
    """KB 2.2: attributes are annotated per identity, not per image.

    This is what makes a tree branch identical for the aerial and the ground
    images of one person. If it ever fails, majority-vote per identity before
    building the tree and say so in the paper.
    """
    vectors = defaultdict(set)
    for path, pid in train_items:
        vector = track_attributes.get(track_key(path))
        if vector is not None:
            vectors[pid].add(tuple(int(v) for v in vector.tolist()))

    unstable = {pid: found for pid, found in vectors.items() if len(found) > 1}
    assert not unstable, (
        f"{len(unstable)} identities carry more than one attribute vector, "
        "so a tree branch would not be view-invariant"
    )


def test_identities_hold_both_aerial_and_ground_images(train_items):
    """Without both views an identity can never form a cross-view positive pair."""
    platforms = defaultdict(set)
    for path, pid in train_items:
        platforms[pid].add(platform_of_path(path))

    usable = {
        pid for pid, found in platforms.items()
        if "aerial" in found and found & {"wearable", "cctv"}
    }
    ratio = len(usable) / len(platforms)
    assert ratio > 0.5, (
        f"only {ratio:.1%} of identities have both aerial and ground images; "
        "the view-balanced sampler cannot build cross-view pairs"
    )


# --------------------------------------------------------------------------
# identity scheme (docs/project.md)
# --------------------------------------------------------------------------

def test_identity_uses_the_composite_p_t_a_key():
    """P alone collapses distinct tracks; the project mandates P+T+A."""
    assert parse_identity_id("P0313T02220A0C0F11011.jpg") == int("0313" + "02220" + "0")


def test_train_split_has_the_expected_identity_count(train_items):
    identities = {pid for _, pid in train_items}
    assert len(identities) == 807, (
        f"expected 807 P+T+A training identities, found {len(identities)}"
    )


def test_track_key_maps_folder_to_mat_index():
    path = os.path.join("AG-ReID.v2", "train_all", "P0000T02140A0", "P0000T02140A0C0F1.jpg")
    assert track_key(path) == "0000021400"
