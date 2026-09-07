import os
import re
import glob
import random
from collections import defaultdict
from typing import Tuple
import scipy.io
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from .attributes import ATTRIBUTE_GROUPS, ATTRIBUTE_IGNORE_INDEX
from .config import BaseConfig
from .identity import parse_identity_id, validate_identity_count

def _scan_images(folder: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.jpg"), recursive=True))
    regex = re.compile(pattern)
    items = []
    invalid = []
    for p in paths:
        try:
            items.append((p, parse_identity_id(p, regex)))
        except ValueError:
            invalid.append(p)
    if invalid:
        raise RuntimeError(
            f"Could not parse P+T+A identity for {len(invalid)} images below "
            f"{folder}; first invalid path: {invalid[0]}"
        )
    if not items:
        raise FileNotFoundError(f"No AG-ReID.v2 JPG images found below {folder}")
    return items


def select_identity_budget(paths, pattern: str, fraction: float, seed: int, minimum: int):
    """Select complete PIDs deterministically for a limited-label experiment.

    The mapping is shared by the supervised PersonViT fine-tune and the
    semi-supervised proposal so a 25% run exposes exactly the same identities
    to both stages.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("labeled_fraction must be in (0, 1]")
    regex = re.compile(pattern)
    pids = set()
    for path in paths:
        pids.add(parse_identity_id(path, regex))
    if not pids:
        raise RuntimeError("No training PIDs could be parsed for the label budget")

    ordered = sorted(pids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    count = max(minimum, round(len(ordered) * fraction))
    count = min(count, len(ordered))
    selected = sorted(ordered[:count])
    return {pid: index for index, pid in enumerate(selected)}, len(ordered)


def parse_camera_id(path: str) -> int:
    match = re.search(r"C(?P<cam>\d+)F", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse camera id from {path}")
    return int(match.group("cam"))


def track_key(path: str) -> str:
    """P0000T02140A0 -> MAT image_index 0000021400."""
    folder = os.path.basename(os.path.dirname(path))
    return "".join(character for character in folder if character.isdigit())


def _mat_string(value) -> str:
    value = np.asarray(value)
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    return str(value).strip()


def load_track_attributes(mat_path: str, ignore_unknown: bool):
    mat_data = scipy.io.loadmat(mat_path)
    if "qut_attribute" not in mat_data:
        raise KeyError(f"qut_attribute is missing from {mat_path}")
    train_data = mat_data["qut_attribute"][0, 0]["train"][0, 0]
    fields = set(train_data.dtype.names or ())
    required = {"image_index"}
    required.update(field for group in ATTRIBUTE_GROUPS.values() for field in group)
    missing = sorted(required - fields)
    if missing:
        raise KeyError(f"Missing attribute fields in MAT file: {missing}")

    mapping = {}
    for row, raw_key in enumerate(train_data["image_index"].reshape(-1)):
        targets = []
        for group_fields in ATTRIBUTE_GROUPS.values():
            values = np.asarray([
                int(train_data[field].reshape(-1)[row]) for field in group_fields
            ])
            positives = np.flatnonzero(values == 2)
            target = int(positives[0]) if len(positives) == 1 else ATTRIBUTE_IGNORE_INDEX
            if ignore_unknown and target == len(group_fields) - 1:
                target = ATTRIBUTE_IGNORE_INDEX
            targets.append(target)
        mapping[_mat_string(raw_key)] = torch.tensor(targets, dtype=torch.long)
    return mapping

class SupervisedReIDDataset(Dataset):
    """Official Bag-of-Tricks augmentation recipe for the BNNeck baseline."""
    def __init__(self, cfg: BaseConfig, pid_to_class=None):
        items = [
            item
            for item in _scan_images(
                os.path.join(cfg.data_root, cfg.train_dir), cfg.filename_pattern
            )
            if item[1] != -1
        ]
        if pid_to_class is None:
            unique_pids = sorted({pid for _, pid in items})
            self.pid2idx = {pid: idx for idx, pid in enumerate(unique_pids)}
        else:
            self.pid2idx = dict(pid_to_class)
            items = [item for item in items if item[1] in self.pid2idx]
            if not items:
                raise RuntimeError("The selected identity budget contains no images")
        self.items = items
        self.num_classes = len(self.pid2idx)
        if pid_to_class is None and cfg.strict_dataset_integrity:
            validate_identity_count(
                self.num_classes,
                cfg.expected_train_identities,
                "AG-ReID.v2 train_all",
            )
        self.labels = [self.pid2idx[pid] for _, pid in self.items]
        self.transform = T.Compose([
            T.Resize(cfg.image_size),
            T.RandomHorizontalFlip(p=cfg.flip_probability),
            T.Pad(cfg.padding),
            T.RandomCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
            T.RandomErasing(
                p=cfg.random_erasing_probability,
                value=cfg.norm_mean,
            ),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, pid = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), self.pid2idx[pid]


class TransReIDDataset(Dataset):
    """Supervised samples with identity and camera/view metadata for SIE."""

    def __init__(self, cfg: BaseConfig):
        self.items = _scan_images(
            os.path.join(cfg.data_root, cfg.train_dir), cfg.filename_pattern
        )
        unique_pids = sorted({pid for _, pid in self.items})
        self.pid2idx = {pid: idx for idx, pid in enumerate(unique_pids)}
        self.num_classes = len(unique_pids)
        if cfg.strict_dataset_integrity:
            validate_identity_count(
                self.num_classes,
                cfg.expected_train_identities,
                "AG-ReID.v2 TransReID train_all",
            )
        self.labels = [self.pid2idx[pid] for _, pid in self.items]
        self.transform = T.Compose([
            T.Resize(cfg.image_size, interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=getattr(cfg, "flip_probability", 0.5)),
            T.Pad(getattr(cfg, "padding", 10)),
            T.RandomCrop(cfg.image_size),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
            T.RandomErasing(
                p=getattr(cfg, "random_erasing_probability", 0.5),
                scale=(0.02, 0.2),
                value=cfg.norm_mean,
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, pid = self.items[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, self.pid2idx[pid], parse_camera_id(path)


class UnlabeledReIDDataset:
    """AG-ReID images grouped only by opaque single-camera track keys.

    Identity tokens in filenames are never parsed. A track is the combination
    of its parent folder and camera, which is obtainable from a single-camera
    tracker without cross-camera identity annotation.
    """

    def __init__(self, cfg: BaseConfig):
        root = os.path.join(cfg.data_root, cfg.train_dir)
        paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
        if not paths:
            raise FileNotFoundError(f"No training images found below {root}")

        raw_items = []
        track_keys = set()
        for path in paths:
            relative_folder = os.path.relpath(os.path.dirname(path), root)
            key = f"{relative_folder}|C{parse_camera_id(path)}"
            raw_items.append((path, key))
            track_keys.add(key)
        self.track_keys = sorted(track_keys)
        track_to_index = {key: index for index, key in enumerate(self.track_keys)}
        self.items = [(path, track_to_index[key]) for path, key in raw_items]
        self.indices_by_track = defaultdict(list)
        for index, (_, track_index) in enumerate(self.items):
            self.indices_by_track[track_index].append(index)

        if getattr(cfg, "preserve_person_aspect_augmentation", False):
            # ``image_size`` follows torchvision's (height, width) convention.
            # Preserve the complete portrait crop before adding only small
            # translation/occlusion perturbations, as in standard ReID recipes.
            self.train_transform = T.Compose([
                T.Resize(cfg.image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.Pad(getattr(cfg, "augmentation_padding", 10)),
                T.RandomCrop(cfg.image_size),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
                T.RandomErasing(
                    p=getattr(cfg, "random_erasing_probability", 0.5),
                    scale=(0.02, 0.2),
                    value="random",
                ),
            ])
        else:
            self.train_transform = T.Compose([
                T.RandomResizedCrop(
                    cfg.image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
                T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ])
        self.cluster_transform = build_eval_transform(
            cfg.image_size, cfg.norm_mean, cfg.norm_std
        )

    @property
    def num_tracks(self):
        return len(self.track_keys)

    def load_train_image(self, index):
        path, _ = self.items[index]
        return self.train_transform(Image.open(path).convert("RGB"))

    def load_cluster_image(self, index):
        path, _ = self.items[index]
        return self.cluster_transform(Image.open(path).convert("RGB"))


class TrackPrototypeDataset(Dataset):
    """Deterministic, bounded samples used to form one prototype per track."""

    def __init__(self, dataset: UnlabeledReIDDataset, samples_per_track: int):
        self.dataset = dataset
        self.indices = []
        for track_index in range(dataset.num_tracks):
            candidates = dataset.indices_by_track[track_index]
            count = min(max(1, samples_per_track), len(candidates))
            if count == 1:
                chosen = [candidates[len(candidates) // 2]]
            else:
                positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
                chosen = [candidates[position] for position in positions]
            self.indices.extend(chosen)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        _, track_index = self.dataset.items[source_index]
        return self.dataset.load_cluster_image(source_index), track_index


class PseudoLabeledDataset(Dataset):
    """Training images whose single-camera track received a DBSCAN cluster."""

    def __init__(self, dataset: UnlabeledReIDDataset, track_labels, samples_per_track):
        self.dataset = dataset
        self.indices = []
        self.labels = []
        for track_index in range(dataset.num_tracks):
            pseudo_label = int(track_labels[track_index])
            if pseudo_label < 0:
                continue
            candidates = dataset.indices_by_track[track_index]
            count = min(max(1, samples_per_track), len(candidates))
            positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
            for position in positions:
                self.indices.append(candidates[position])
                self.labels.append(pseudo_label)
        if not self.indices:
            raise RuntimeError("DBSCAN produced no usable pseudo-labeled images")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset.load_train_image(self.indices[index]), self.labels[index]


class RandomIdentitySampler(Sampler):
    """P x K sampler required by batch-hard triplet learning."""

    def __init__(self, labels, batch_size: int, instances_per_identity: int, seed=42):
        if batch_size % instances_per_identity != 0:
            raise ValueError("batch_size must be divisible by instances_per_identity")
        self.labels = list(labels)
        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.identities_per_batch = batch_size // instances_per_identity
        self.seed = seed
        self.epoch = 0
        self.index_by_pid = defaultdict(list)
        for index, pid in enumerate(self.labels):
            self.index_by_pid[pid].append(index)
        self.pids = sorted(self.index_by_pid)
        self.length = (
            sum(
                max(len(indices), instances_per_identity)
                // instances_per_identity
                * instances_per_identity
                for indices in self.index_by_pid.values()
            )
            // batch_size
            * batch_size
        )

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        chunks = {}
        for pid, source_indices in self.index_by_pid.items():
            indices = list(source_indices)
            if len(indices) < self.instances_per_identity:
                indices.extend(rng.choices(indices, k=self.instances_per_identity - len(indices)))
            rng.shuffle(indices)
            remainder = len(indices) % self.instances_per_identity
            if remainder:
                indices.extend(rng.choices(indices, k=self.instances_per_identity - remainder))
            chunks[pid] = [
                indices[i : i + self.instances_per_identity]
                for i in range(0, len(indices), self.instances_per_identity)
            ]

        available = [pid for pid in self.pids if chunks[pid]]
        final_indices = []
        target_batches = self.length // self.batch_size
        for _ in range(target_batches):
            selected = list(available)
            if len(selected) >= self.identities_per_batch:
                selected = rng.sample(selected, self.identities_per_batch)
            else:
                candidates = [pid for pid in self.pids if pid not in selected]
                selected.extend(
                    rng.sample(candidates, self.identities_per_batch - len(selected))
                )
            for pid in selected:
                if chunks[pid]:
                    final_indices.extend(chunks[pid].pop(0))
                else:
                    source = self.index_by_pid[pid]
                    final_indices.extend(
                        rng.choices(source, k=self.instances_per_identity)
                    )
                if pid in available and not chunks[pid]:
                    available.remove(pid)
        self.epoch += 1
        return iter(final_indices)

class TemporalTrackletDataset(Dataset):
    """Chronological frame windows for the LeWM-inspired ReID proposal."""

    def __init__(self, cfg: BaseConfig):
        root = os.path.join(cfg.data_root, cfg.train_dir)
        mat_path = os.path.join(cfg.data_root, cfg.attr_file)
        self.track_to_attr = load_track_attributes(mat_path, cfg.ignore_unknown_attributes)
        self.default_attr = torch.full(
            (len(ATTRIBUTE_GROUPS),), ATTRIBUTE_IGNORE_INDEX, dtype=torch.long
        )
        self.windows = []
        all_pids = set()
        for folder in sorted(glob.glob(os.path.join(root, "P*T*A*"))):
            try:
                pid = parse_identity_id(folder)
            except ValueError:
                continue
            all_pids.add(pid)
            paths_by_camera = defaultdict(list)
            for path in glob.glob(os.path.join(folder, "*.jpg")):
                paths_by_camera[parse_camera_id(path)].append(path)
            for paths in paths_by_camera.values():
                paths.sort(key=self._frame_number)
                if len(paths) < cfg.sequence_length:
                    continue
                max_start = len(paths) - cfg.sequence_length
                for start in range(0, max_start + 1, cfg.sequence_stride):
                    sequence = paths[start : start + cfg.sequence_length]
                    self.windows.append((sequence, pid, False))

        unique_pids = sorted(all_pids)
        self.pid2idx = {pid: index for index, pid in enumerate(unique_pids)}
        self.num_classes = len(unique_pids)
        if cfg.strict_dataset_integrity:
            validate_identity_count(
                self.num_classes,
                cfg.expected_train_identities,
                "AG-ReID.v2 temporal train_all",
            )
        self._assign_labeled_windows(cfg.labeled_fraction, cfg.labeled_split_seed)
        self.transform = T.Compose([
            T.RandomResizedCrop(
                cfg.image_size,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.25),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
        ])

    @staticmethod
    def _frame_number(path):
        match = re.search(r"F(?P<frame>\d+)", os.path.basename(path))
        return int(match.group("frame")) if match else 0

    def _assign_labeled_windows(self, fraction, seed):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("labeled_fraction must be in (0, 1]")
        by_pid = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.windows):
            by_pid[pid].append(index)
        rng = random.Random(seed)
        labeled = set()
        for indices in by_pid.values():
            rng.shuffle(indices)
            count = max(1, round(len(indices) * fraction))
            labeled.update(indices[:count])
        self.windows = [
            (paths, pid, index in labeled)
            for index, (paths, pid, _) in enumerate(self.windows)
        ]
        self.labeled_window_count = len(labeled)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        paths, pid, is_labeled = self.windows[index]
        frames = torch.stack([
            self.transform(Image.open(path).convert("RGB")) for path in paths
        ])
        label = self.pid2idx[pid] if is_labeled else ATTRIBUTE_IGNORE_INDEX
        attributes = self.track_to_attr.get(track_key(paths[0]), self.default_attr)
        return frames, label, attributes

class EvalReIDDataset(Dataset):
    def __init__(self, data_root: str, txt_file: str, split: str, image_size: Tuple[int, int],
                 norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        self.items = []
        camera_regex = re.compile(r"C(?P<cam>\d+)F")
        
        with open(txt_file, 'r') as f:
            lines = f.read().splitlines()
            
        for line in lines:
            if not line.startswith(split):
                continue
            path = os.path.join(data_root, line)
            camera_match = camera_regex.search(line)
            if not camera_match:
                raise ValueError(f"Cannot parse camera id from protocol entry: {line}")
            pid = parse_identity_id(line)
            cam = int(camera_match.group("cam"))
            self.items.append((path, pid, cam))
        if not self.items:
            raise RuntimeError(f"Protocol contains no {split} samples: {txt_file}")
        self.transform = build_eval_transform(image_size, norm_mean, norm_std)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, pid, camid = self.items[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, pid, camid
    
def build_eval_transform(image_size: Tuple[int, int],
                         norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                         norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> T.Compose:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ])
