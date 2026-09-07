"""QC-02: unit tests for the shared view mapping (no dataset needed)."""

from __future__ import annotations

import pytest
import torch

from reid_advance.hierarchy import (
    AERIAL,
    GROUND,
    UnknownCameraError,
    binary_view_of_camera,
    binary_view_tensor,
    platform_of_camera,
    platform_tensor,
)
from reid_advance.pipelines.transreid import camera_to_view


@pytest.mark.parametrize(
    "camera_id,platform",
    [(0, "aerial"), (2, "wearable"), (3, "cctv")],
)
def test_platform_of_camera(camera_id, platform):
    assert platform_of_camera(camera_id) == platform


@pytest.mark.parametrize(
    "camera_id,view", [(0, AERIAL), (2, GROUND), (3, GROUND)]
)
def test_binary_view_collapses_both_ground_platforms(camera_id, view):
    assert binary_view_of_camera(camera_id) == view


def test_binary_view_tensor_matches_scalar_version():
    cameras = torch.tensor([0, 2, 3, 3, 0, 2])
    expected = torch.tensor([binary_view_of_camera(int(c)) for c in cameras])
    assert torch.equal(binary_view_tensor(cameras), expected)


def test_platform_tensor_matches_the_existing_pipeline_helper():
    """The SIE branch keeps its 3-way encoding; don't let the two drift."""
    cameras = torch.tensor([0, 2, 3, 0, 3, 2])
    assert torch.equal(platform_tensor(cameras), camera_to_view(cameras))


@pytest.mark.parametrize("camera_id", [1, 4, -1, 99])
def test_unknown_camera_ids_raise(camera_id):
    with pytest.raises(UnknownCameraError):
        platform_of_camera(camera_id)


def test_unknown_camera_in_tensor_reports_the_offenders():
    with pytest.raises(UnknownCameraError, match=r"\[1, 7\]"):
        binary_view_tensor(torch.tensor([0, 1, 3, 7]))
