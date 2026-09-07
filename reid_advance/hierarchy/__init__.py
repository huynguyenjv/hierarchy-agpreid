"""Hierarchy-aware representation learning for aerial-ground ReID (Topic 1)."""

from .view_map import (
    AERIAL,
    CAMERA_TO_PLATFORM,
    GROUND,
    UnknownCameraError,
    binary_view_of_camera,
    binary_view_of_path,
    binary_view_tensor,
    platform_of_camera,
    platform_of_path,
    platform_tensor,
)

__all__ = [
    "AERIAL",
    "CAMERA_TO_PLATFORM",
    "GROUND",
    "UnknownCameraError",
    "binary_view_of_camera",
    "binary_view_of_path",
    "binary_view_tensor",
    "platform_of_camera",
    "platform_of_path",
    "platform_tensor",
]
