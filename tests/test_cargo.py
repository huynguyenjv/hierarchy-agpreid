"""QC tests for the CARGO adapter."""

from __future__ import annotations

import os

import pytest
import torch

from reid_advance.cargo import (
    AERIAL,
    AERIAL_CAMERAS,
    GROUND,
    GROUND_CAMERAS,
    CargoParseError,
    binary_view_of_camera,
    binary_view_of_path,
    binary_view_tensor,
    parse_cargo_name,
    platform_of_camera,
    scan_cargo_images,
    summarise,
)

CARGO_ROOT = "D:/datasets/cargo"


# --------------------------------------------------------------------------
# filename parsing
# --------------------------------------------------------------------------

def test_parses_the_documented_filename_format():
    """camID_time_personID_index.jpg, e.g. Cam2_day_2519_320.jpg."""
    assert parse_cargo_name("Cam2_day_2519_320.jpg") == (2519, 2, "day")


def test_parses_night_captures_and_two_digit_cameras():
    assert parse_cargo_name("Cam13_night_47_9001.jpg") == (47, 13, "night")


def test_parsing_ignores_leading_directories():
    path = os.path.join("gallery", "Cam7", "Cam7_day_100_5.jpg")
    assert parse_cargo_name(path) == (100, 7, "day")


def test_a_non_cargo_filename_raises():
    """AG-ReID.v2 names must not silently parse as CARGO ones."""
    with pytest.raises(CargoParseError):
        parse_cargo_name("P0313T02220A0C0F11011.jpg")


# --------------------------------------------------------------------------
# view mapping - cameras 1-5 aerial, 6-13 ground
# --------------------------------------------------------------------------

@pytest.mark.parametrize("camera_id", sorted(AERIAL_CAMERAS))
def test_cameras_one_to_five_are_aerial(camera_id):
    assert binary_view_of_camera(camera_id) == AERIAL
    assert platform_of_camera(camera_id) == "aerial"


@pytest.mark.parametrize("camera_id", sorted(GROUND_CAMERAS))
def test_cameras_six_to_thirteen_are_ground(camera_id):
    assert binary_view_of_camera(camera_id) == GROUND
    assert platform_of_camera(camera_id) == "ground"


def test_view_of_path_matches_view_of_camera():
    assert binary_view_of_path("Cam3_day_10_1.jpg") == AERIAL
    assert binary_view_of_path("Cam9_day_10_1.jpg") == GROUND


@pytest.mark.parametrize("camera_id", [0, 14, 99, -1])
def test_cameras_outside_the_documented_range_raise(camera_id):
    """Defaulting an unknown camera to ground would silently corrupt any
    cross-view analysis, exactly as it would on AG-ReID.v2."""
    with pytest.raises(CargoParseError):
        binary_view_of_camera(camera_id)


def test_binary_view_tensor_matches_the_scalar_version():
    cameras = torch.tensor([1, 5, 6, 13, 3, 9])
    expected = torch.tensor([binary_view_of_camera(int(c)) for c in cameras])
    assert torch.equal(binary_view_tensor(cameras), expected)


def test_binary_view_tensor_reports_unknown_cameras():
    with pytest.raises(CargoParseError, match=r"\[0, 14\]"):
        binary_view_tensor(torch.tensor([1, 0, 7, 14]))


# --------------------------------------------------------------------------
# on-disk checks
# --------------------------------------------------------------------------

pytestmark_dataset = pytest.mark.skipif(
    not os.path.isdir(CARGO_ROOT), reason="CARGO not downloaded"
)


@pytestmark_dataset
def test_train_split_has_the_documented_size():
    items = scan_cargo_images(os.path.join(CARGO_ROOT, "train"))
    summary = summarise(items)
    assert summary["images"] == 51451
    assert summary["identities"] == 2500


@pytestmark_dataset
def test_all_thirteen_cameras_are_present_in_train():
    summary = summarise(scan_cargo_images(os.path.join(CARGO_ROOT, "train")))
    assert set(summary["images_per_camera"]) == AERIAL_CAMERAS | GROUND_CAMERAS


@pytestmark_dataset
def test_a_same_view_control_is_constructible_on_both_platforms():
    """The property AG-ReID.v2 lacked.

    AG-ReID.v2 has one aerial camera, so aerial-to-aerial retrieval is
    impossible there and its same-view control had to be ground-only. CARGO
    must support both, or the stricter comparison this dataset was fetched for
    does not exist.
    """
    summary = summarise(scan_cargo_images(os.path.join(CARGO_ROOT, "gallery")))
    assert summary["identities_with_2plus_aerial_cameras"] > 100
    assert summary["identities_with_2plus_ground_cameras"] > 100


@pytestmark_dataset
def test_gallery_identities_span_both_platforms():
    summary = summarise(scan_cargo_images(os.path.join(CARGO_ROOT, "gallery")))
    ratio = summary["identities_with_both_platforms"] / summary["identities"]
    assert ratio > 0.5, (
        f"only {ratio:.1%} of gallery identities appear on both platforms; "
        "cross-platform pairs would be too sparse to score"
    )
