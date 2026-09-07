"""CARGO dataset adapter, kept deliberately parallel to the AG-ReID.v2 path.

CARGO exists here for one purpose: to tell whether the flat camera-pair matrix
measured on AG-ReID.v2 is a property of that dataset or of aerial-ground ReID
after ordinary supervision.  That question is only answerable if both datasets
are measured the *same* way, so this module reuses the existing TransReIDSmall
backbone, identity + triplet losses, PK sampler and ``evaluate_rank`` rather
than importing CARGO's own fast-reid pipeline.

Filenames are ``camID_time_personID_index.jpg``, e.g. ``Cam2_day_2519_320.jpg``.
Cameras 1-5 are aerial (UAV), cameras 6-13 are ground (fixed CCTV).

Unlike AG-ReID.v2, whose aerial side is a single camera, CARGO has five aerial
and eight ground cameras.  A same-view control is therefore constructible on
*both* platforms here, which AG-ReID.v2 could not support.
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

#: ``Cam<id>_<time>_<pid>_<index>.jpg``
FILENAME_PATTERN = re.compile(
    r"Cam(?P<cam>\d+)_(?P<time>[A-Za-z]+)_(?P<pid>\d+)_(?P<index>\d+)",
    re.IGNORECASE,
)

AERIAL_CAMERAS = frozenset(range(1, 6))    # Cam1-Cam5, UAV
GROUND_CAMERAS = frozenset(range(6, 14))   # Cam6-Cam13, fixed CCTV

AERIAL = 1
GROUND = 0


class CargoParseError(ValueError):
    """Raised when a filename does not follow the CARGO convention."""


def parse_cargo_name(path: str) -> tuple[int, int, str]:
    """Return ``(pid, camera_id, time_of_day)`` for one CARGO image."""
    match = FILENAME_PATTERN.search(os.path.basename(os.fspath(path)))
    if not match:
        raise CargoParseError(f"Cannot parse a CARGO filename from {path}")
    return (
        int(match.group("pid")),
        int(match.group("cam")),
        match.group("time").lower(),
    )


def parse_identity_id(path: str) -> int:
    return parse_cargo_name(path)[0]


def parse_camera_id(path: str) -> int:
    return parse_cargo_name(path)[1]


def binary_view_of_camera(camera_id: int) -> int:
    """1 = aerial (Cam1-5), 0 = ground (Cam6-13)."""
    if camera_id in AERIAL_CAMERAS:
        return AERIAL
    if camera_id in GROUND_CAMERAS:
        return GROUND
    raise CargoParseError(
        f"camera {camera_id} is outside CARGO's documented range 1-13"
    )


def binary_view_of_path(path: str) -> int:
    return binary_view_of_camera(parse_camera_id(path))


def platform_of_camera(camera_id: int) -> str:
    return "aerial" if binary_view_of_camera(camera_id) == AERIAL else "ground"


def binary_view_tensor(camera_ids: torch.Tensor) -> torch.Tensor:
    """Vectorised view lookup that raises on unknown cameras.

    Mirrors ``hierarchy.view_map.binary_view_tensor``: an unrecognised camera
    must fail loudly rather than silently defaulting to ground, because a wrong
    view label turns any cross-view analysis into a no-op without a symptom.
    """
    known = torch.zeros_like(camera_ids, dtype=torch.bool)
    for camera_id in AERIAL_CAMERAS | GROUND_CAMERAS:
        known |= camera_ids == camera_id
    if not bool(known.all()):
        offenders = sorted({int(v) for v in camera_ids[~known].tolist()})
        raise CargoParseError(f"cameras {offenders} are outside CARGO's range 1-13")
    aerial = torch.zeros_like(camera_ids, dtype=torch.bool)
    for camera_id in AERIAL_CAMERAS:
        aerial |= camera_ids == camera_id
    return aerial.long()


def scan_cargo_images(folder: str) -> list[tuple[str, int, int]]:
    """All ``(path, pid, camera_id)`` under a CARGO split directory."""
    paths = sorted(
        glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)
    )
    items, invalid = [], []
    for path in paths:
        try:
            pid, camera_id, _ = parse_cargo_name(path)
        except CargoParseError:
            invalid.append(path)
            continue
        items.append((path, pid, camera_id))
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} files below {folder} do not follow the CARGO "
            f"naming convention; first: {invalid[0]}"
        )
    if not items:
        raise FileNotFoundError(f"No CARGO images found below {folder}")
    return items


def summarise(items: list[tuple[str, int, int]]) -> dict:
    """Counts needed to check a same-view control is even constructible."""
    per_camera: dict[int, int] = defaultdict(int)
    cameras_per_identity: dict[int, set[int]] = defaultdict(set)
    for _, pid, camera_id in items:
        per_camera[camera_id] += 1
        cameras_per_identity[pid].add(camera_id)

    both_platforms = 0
    multi_aerial = 0
    multi_ground = 0
    for cameras in cameras_per_identity.values():
        aerial = cameras & AERIAL_CAMERAS
        ground = cameras & GROUND_CAMERAS
        if aerial and ground:
            both_platforms += 1
        if len(aerial) >= 2:
            multi_aerial += 1
        if len(ground) >= 2:
            multi_ground += 1

    return {
        "images": len(items),
        "identities": len(cameras_per_identity),
        "images_per_camera": dict(sorted(per_camera.items())),
        "identities_with_both_platforms": both_platforms,
        "identities_with_2plus_aerial_cameras": multi_aerial,
        "identities_with_2plus_ground_cameras": multi_ground,
    }


class CargoDataset(Dataset):
    """Training split with the same augmentation recipe as TransReIDDataset."""

    def __init__(self, root: str, image_size, norm_mean, norm_std,
                 flip_probability: float = 0.5, padding: int = 10,
                 random_erasing_probability: float = 0.5,
                 pid_to_class: dict[int, int] | None = None):
        self.items = scan_cargo_images(root)
        if pid_to_class is None:
            unique = sorted({pid for _, pid, _ in self.items})
            self.pid2idx = {pid: index for index, pid in enumerate(unique)}
        else:
            self.pid2idx = dict(pid_to_class)
            self.items = [item for item in self.items if item[1] in self.pid2idx]
        self.num_classes = len(self.pid2idx)
        self.labels = [self.pid2idx[pid] for _, pid, _ in self.items]
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=flip_probability),
            T.Pad(padding),
            T.RandomCrop(image_size),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
            T.RandomErasing(p=random_erasing_probability, scale=(0.02, 0.2),
                            value=norm_mean),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        path, pid, camera_id = self.items[index]
        image = self.transform(Image.open(path).convert("RGB"))
        # Camera ids are passed straight through for SIE; the trainer maps them
        # to the embedding's index space.
        return image, self.pid2idx[pid], camera_id


class CargoEvalDataset(Dataset):
    """Evaluation split, no augmentation."""

    def __init__(self, root: str, image_size, norm_mean, norm_std):
        self.items = scan_cargo_images(root)
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        path, pid, camera_id = self.items[index]
        return self.transform(Image.open(path).convert("RGB")), pid, camera_id
