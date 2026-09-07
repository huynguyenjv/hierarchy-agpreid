"""Validate the AG-ReID.v2 MAT attribute schema and one-hot rows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reid_advance.attributes import ATTRIBUTE_GROUPS  # noqa: E402


def main():
    mat_path = ROOT / "AG-ReID.v2" / "qut_attribute_v8.mat"
    root = scipy.io.loadmat(mat_path)["qut_attribute"][0, 0]
    train = root["train"][0, 0]
    rows = train["image_index"].size
    print(f"MAT rows: {rows}")
    for name, fields in ATTRIBUTE_GROUPS.items():
        values = np.stack([train[field].reshape(-1) for field in fields], axis=1)
        positive_count = (values == 2).sum(axis=1)
        invalid = int((positive_count != 1).sum())
        unknown = int((values[:, -1] == 2).sum())
        print(f"{name:10s} classes={len(fields):2d} invalid={invalid:3d} unknown={unknown:3d}")


if __name__ == "__main__":
    main()
