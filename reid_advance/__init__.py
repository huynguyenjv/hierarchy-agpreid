"""AG-ReID.v2 supervised baselines and semi-supervised temporal ReID."""

import os

# TensorBoard may discover TensorFlow even though the training code uses
# PyTorch. Silence TensorFlow INFO/oneDNN startup messages in the parent and in
# Windows DataLoader workers before tensorboard is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from .config import (
    BNNeckConfig,
    ProposedConfig,
    ProposedV2Config,
    ProposedV3Config,
    ProposedV3FineTuneConfig,
    TransReIDConfig,
)

__all__ = [
    "BNNeckConfig",
    "TransReIDConfig",
    "ProposedConfig",
    "ProposedV2Config",
    "ProposedV3Config",
    "ProposedV3FineTuneConfig",
]
