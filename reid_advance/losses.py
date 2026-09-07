"""Shared ReID objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attributes import ATTRIBUTE_IGNORE_INDEX


def masked_attribute_loss(logits1, logits2, targets):
    """Average CE over available attributes and safely skip empty groups."""
    losses = []
    valid_labels = 0
    for index, name in enumerate(logits1):
        target = targets[:, index]
        valid = target != ATTRIBUTE_IGNORE_INDEX
        if not torch.any(valid):
            continue
        loss1 = F.cross_entropy(logits1[name][valid], target[valid])
        loss2 = F.cross_entropy(logits2[name][valid], target[valid])
        losses.append(0.5 * (loss1 + loss2))
        valid_labels += int(valid.sum().item())
    if not losses:
        return sum(value.sum() * 0.0 for value in logits1.values()), 0
    return torch.stack(losses).mean(), valid_labels


class TripletLoss(nn.Module):
    """Batch-hard triplet loss used by both supervised and semi-supervised runs."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # Pairwise distance is intentionally computed in fp32. Under AMP,
        # reduction ops can produce fp32 while embeddings remain fp16; the
        # in-place addmm_ then rejects the mixed dtypes. Keeping this block in
        # fp32 also avoids precision loss when distances are close to zero.
        features = inputs.float()
        number = features.size(0)
        distance = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(number, number)
        distance = distance + distance.t()
        distance.addmm_(features, features.t(), beta=1, alpha=-2)
        distance = distance.clamp(min=1e-12).sqrt()

        same_identity = targets.expand(number, number).eq(
            targets.expand(number, number).t()
        )
        hardest_positive = torch.cat([
            distance[index][same_identity[index]].max().unsqueeze(0)
            for index in range(number)
        ])
        hardest_negative = torch.cat([
            distance[index][~same_identity[index]].min().unsqueeze(0)
            for index in range(number)
        ])
        target = torch.ones_like(hardest_negative)
        return self.ranking_loss(hardest_negative, hardest_positive, target)
