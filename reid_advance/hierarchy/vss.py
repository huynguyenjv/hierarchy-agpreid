"""View Stability Score: how well each soft attribute survives the climb (AI-03).

An attribute only works as a tree level if it is recognisable from *both*
platforms. Gender read off a 15-120m drone shot is still gender; "wearing
glasses" is not. VSS quantifies that: train one shared ViT-S with 15 attribute
heads on ground images, then measure per-attribute accuracy on held-out aerial
images.

    VSS(attr) = accuracy on aerial images of held-out identities

Two design points that make the number meaningful:

1. **Identity-disjoint split.** Attributes are annotated per identity, so if the
   same identity appeared in train and test the model could recognise the person
   and look the attribute up instead of reading it off the image. VSS would then
   measure re-identification, not attribute legibility. Train and test identities
   are therefore disjoint.
2. **Ground-only training.** The drop from ground to aerial accuracy is exactly
   the quantity of interest, so aerial images are never trained on.

Because the classes are heavily imbalanced (KB 6.1 criterion 2), plain accuracy
flatters a group whose majority class dominates. Balanced accuracy is reported
alongside it and is the one to trust when ordering tree levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..attributes import ATTRIBUTE_CLASS_COUNTS, ATTRIBUTE_IGNORE_INDEX, ATTRIBUTE_NAMES


@dataclass
class AttributeScore:
    """Per-attribute view stability, on ground and aerial held-out images."""

    name: str
    ground_accuracy: float
    aerial_accuracy: float
    ground_balanced_accuracy: float
    aerial_balanced_accuracy: float
    majority_baseline: float
    support_ground: int
    support_aerial: int
    per_class_aerial: dict[int, float] = field(default_factory=dict)

    @property
    def vss(self) -> float:
        """The headline number: balanced accuracy on aerial images."""
        return self.aerial_balanced_accuracy

    @property
    def retention(self) -> float:
        """Fraction of ground performance that survives the view change."""
        if self.ground_balanced_accuracy <= 0:
            return 0.0
        return self.aerial_balanced_accuracy / self.ground_balanced_accuracy

    @property
    def lift_over_majority(self) -> float:
        """Aerial accuracy above always predicting the most frequent class.

        A group can score a high raw accuracy purely because one class covers
        95% of identities; only the lift shows the model learned anything.
        """
        return self.aerial_accuracy - self.majority_baseline


class AttributeHeads(nn.Module):
    """One shared backbone feature, one linear head per attribute group."""

    def __init__(self, embed_dim: int, names: tuple[str, ...] = ATTRIBUTE_NAMES):
        super().__init__()
        self.names = tuple(names)
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, ATTRIBUTE_CLASS_COUNTS[name])
            for name in self.names
        })

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = self.bottleneck(features)
        return {name: head(normalized) for name, head in self.heads.items()}


def attribute_ce_loss(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    names: tuple[str, ...] = ATTRIBUTE_NAMES,
) -> tuple[torch.Tensor, int]:
    """Mean CE across attribute groups, skipping rows labelled unknown.

    ``targets`` is (B, 15) with ``ATTRIBUTE_IGNORE_INDEX`` marking unknown.
    Groups with no valid row in the batch contribute nothing rather than a NaN.
    """
    losses = []
    valid_count = 0
    for index, name in enumerate(names):
        target = targets[:, index]
        valid = target != ATTRIBUTE_IGNORE_INDEX
        if not bool(valid.any()):
            continue
        losses.append(nn.functional.cross_entropy(logits[name][valid], target[valid]))
        valid_count += int(valid.sum())
    if not losses:
        zero = sum(value.sum() * 0.0 for value in logits.values())
        return zero, 0
    return torch.stack(losses).mean(), valid_count


def balanced_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean per-class recall, ignoring classes absent from the targets."""
    recalls = []
    for label in np.unique(targets):
        mask = targets == label
        recalls.append(float((predictions[mask] == label).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def score_attribute(
    name: str,
    ground_predictions: np.ndarray,
    ground_targets: np.ndarray,
    aerial_predictions: np.ndarray,
    aerial_targets: np.ndarray,
) -> AttributeScore:
    """Assemble the per-attribute report from held-out predictions."""
    majority = 0.0
    if aerial_targets.size:
        counts = np.bincount(aerial_targets)
        majority = float(counts.max() / counts.sum())

    per_class = {}
    for label in np.unique(aerial_targets):
        mask = aerial_targets == label
        per_class[int(label)] = float((aerial_predictions[mask] == label).mean())

    return AttributeScore(
        name=name,
        ground_accuracy=float((ground_predictions == ground_targets).mean())
        if ground_targets.size else 0.0,
        aerial_accuracy=float((aerial_predictions == aerial_targets).mean())
        if aerial_targets.size else 0.0,
        ground_balanced_accuracy=balanced_accuracy(ground_predictions, ground_targets),
        aerial_balanced_accuracy=balanced_accuracy(aerial_predictions, aerial_targets),
        majority_baseline=majority,
        support_ground=int(ground_targets.size),
        support_aerial=int(aerial_targets.size),
        per_class_aerial=per_class,
    )


#: Attributes scoring below this are not usable as a tree level (KB 6.1).
#: The knowledge base proposes 65% raw accuracy; balanced accuracy is stricter
#: and is what the threshold is applied to here.
VSS_THRESHOLD = 0.65
