"""Shared fixtures and seed locking (QC-01)."""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AG-ReID.v2")
MAT_PATH = os.path.join(DATA_ROOT, "qut_attribute_v8.mat")

SEED = 42


@pytest.fixture(autouse=True)
def _lock_seeds():
    """Every test starts from the same RNG state, in all three libraries."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    yield


def requires_dataset(*paths: str):
    """Skip marker for tests that read the real AG-ReID.v2 files."""
    missing = [p for p in paths if not os.path.exists(p)]
    return pytest.mark.skipif(
        bool(missing), reason=f"dataset files missing: {missing}"
    )


@pytest.fixture(scope="session")
def data_root() -> str:
    return DATA_ROOT


@pytest.fixture(scope="session")
def mat_path() -> str:
    return MAT_PATH


@pytest.fixture(scope="session")
def train_items():
    """(path, pid) pairs for the train_all split, parsed with the P+T+A scheme."""
    from reid_advance.data import _scan_images
    from reid_advance.identity import FILENAME_PATTERN

    split = os.path.join(DATA_ROOT, "train_all")
    if not os.path.isdir(split):
        pytest.skip("AG-ReID.v2/train_all not available")
    return _scan_images(split, FILENAME_PATTERN)


@pytest.fixture(scope="session")
def track_attributes():
    """Identity-level attribute vectors keyed by MAT image_index."""
    from reid_advance.data import load_track_attributes

    if not os.path.exists(MAT_PATH):
        pytest.skip("qut_attribute_v8.mat not available")
    return load_track_attributes(MAT_PATH, ignore_unknown=False)
