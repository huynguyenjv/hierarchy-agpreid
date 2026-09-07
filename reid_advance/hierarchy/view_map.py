"""Platform/view labels for AG-ReID.v2 (BA-01).

The aerial-ground gap is what the hierarchy is meant to bridge, so the sampler,
the CV-HWC loss and the hierarchy-aware metrics all need the same notion of
"which platform did this image come from".  Keeping it in one module stops the
three from drifting apart.

Camera assignment was verified against the four official protocol files:

- ``exp1_aerial_to_cctv``     contains only cameras {0, 3}, queries are C0
- ``exp2_aerial_to_wearable`` contains only cameras {0, 2}, queries are C0
- ``exp4_cctv_to_aerial``     queries are C3
- ``exp5_wearable_to_aerial`` queries are C2

so C0=aerial, C2=wearable, C3=CCTV.  See ``docs/view_mapping.md``.
"""

from __future__ import annotations

from typing import Literal

import torch

from ..data import parse_camera_id


Platform = Literal["aerial", "wearable", "cctv"]

#: The only camera ids present in AG-ReID.v2 train_all/query/gallery.
CAMERA_TO_PLATFORM: dict[int, Platform] = {
    0: "aerial",
    2: "wearable",
    3: "cctv",
}

#: Binary view used by the cross-view reweighting term of CV-HWC.
#: Both ground platforms collapse into one class: the contribution is about the
#: aerial-ground gap, not about telling CCTV from wearable.
AERIAL = 1
GROUND = 0

GROUND_PLATFORMS: frozenset[Platform] = frozenset({"wearable", "cctv"})


class UnknownCameraError(ValueError):
    """Raised when an image carries a camera id outside the known set."""


def platform_of_camera(camera_id: int) -> Platform:
    try:
        return CAMERA_TO_PLATFORM[int(camera_id)]
    except KeyError as error:
        raise UnknownCameraError(
            f"camera id {camera_id} is not one of {sorted(CAMERA_TO_PLATFORM)}; "
            "AG-ReID.v2 should only contain C0/C2/C3"
        ) from error


def platform_of_path(path: str) -> Platform:
    return platform_of_camera(parse_camera_id(path))


def binary_view_of_camera(camera_id: int) -> int:
    """0 = ground (wearable or CCTV), 1 = aerial."""
    return AERIAL if platform_of_camera(camera_id) == "aerial" else GROUND


def binary_view_of_path(path: str) -> int:
    return binary_view_of_camera(parse_camera_id(path))


def binary_view_tensor(camera_ids: torch.Tensor) -> torch.Tensor:
    """Vectorised ``binary_view_of_camera`` for a batch of camera ids.

    Unknown ids raise rather than silently mapping to ground, because a wrong
    view label makes the cross-view term of CV-HWC a no-op without any visible
    failure.
    """
    known = torch.zeros_like(camera_ids, dtype=torch.bool)
    for camera_id in CAMERA_TO_PLATFORM:
        known |= camera_ids == camera_id
    if not bool(known.all()):
        offenders = sorted({int(value) for value in camera_ids[~known].tolist()})
        raise UnknownCameraError(
            f"camera ids {offenders} are not one of {sorted(CAMERA_TO_PLATFORM)}"
        )
    return (camera_ids == 0).long()


def platform_tensor(camera_ids: torch.Tensor) -> torch.Tensor:
    """Three-way platform label: 0 aerial, 1 wearable, 2 cctv.

    Matches ``pipelines.transreid.camera_to_view`` so SIE keeps its existing
    behaviour; use ``binary_view_tensor`` for the CV-HWC cross-view term.
    """
    result = torch.zeros_like(camera_ids, dtype=torch.long)
    result[camera_ids == 2] = 1
    result[camera_ids == 3] = 2
    return result
