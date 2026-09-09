"""Multi-granularity contrastive loss with view-conditioned level weighting.

The hypothesis this exists to test: an aerial-ground pair and a same-platform
pair should not be matched at the same spatial granularity. Climbing costs local
detail but leaves whole-body structure, so a cross-platform positive is best
pulled together on the coarse levels, while a same-platform positive can be
matched on fine stripes. The loss therefore runs a supervised contrastive term
at every level of the granularity pyramid and weights the levels per pair,
according to whether that pair crosses platforms.

    L_MG = sum_l  beta(l, cross_ij) * SupCon_l

Three properties make the result readable:

**Fixed budget.** beta is a softmax over levels, so every pair spends the same
total weight however it is distributed. If the view-aware branch beats the
uniform-beta branch, the gain cannot be attributed to simply optimising a larger
loss - both spend the same.

**Decorrelated features.** ``SupCon_l`` runs on ``decorrelate_level`` output,
never on raw tokens. Raw ViT tokens sit at cosine 0.993 +/- 0.001 whatever level
they are pooled at, which at tau=0.07 leaves no usable gradient; the loss would
fall steadily and teach nothing.

**Comparable spreads.** After decorrelation the four levels have near identical
similarity spread (0.089-0.092) despite dimensionality running 384 to 3072, so
beta acts on granularity rather than compensating for scale.

The uniform-beta control is this same loss with ``coarse_bias=0``: still four
levels, same parameters, same compute, same schedule - only the view
conditioning is removed. It is the branch that separates a genuine granularity
effect from plain multi-scale features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .granularity import GranularitySpec, decorrelate_level, level_weights_for_view, pyramid_features


def supcon_level(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    pair_weight: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss for one granularity level.

    ``pair_weight`` is a (B, B) weight applied to positive pairs only; the
    denominator always spans every other sample, so reweighting redistributes
    attention among positives without changing what they are contrasted
    against.

    Anchors with no positive in the batch contribute nothing rather than a NaN.
    """
    batch = embeddings.shape[0]
    device = embeddings.device

    similarity = embeddings @ embeddings.t() / temperature
    eye = torch.eye(batch, dtype=torch.bool, device=device)

    positives = (labels[:, None] == labels[None, :]) & ~eye
    # log-softmax over everything except the anchor itself
    logits = similarity.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # The diagonal is -inf, and 0 * -inf is NaN rather than 0, so the entries
    # the weights are about to discard have to be neutralised first. Masking
    # after multiplying is too late.
    log_prob = log_prob.masked_fill(eye, 0.0)

    weights = positives.float()
    if pair_weight is not None:
        weights = weights * pair_weight

    total = weights.sum(dim=1)
    valid = total > 0
    if not bool(valid.any()):
        return embeddings.sum() * 0.0

    per_anchor = -(weights * log_prob).sum(dim=1)[valid] / total[valid]
    return per_anchor.mean()


class MultiGranularityLoss(nn.Module):
    """``L_MG`` over the granularity pyramid, with per-level BatchNorm.

    The norms are owned by this module so their running statistics persist
    across steps, mirroring the BNNeck the identity head already uses.
    """

    def __init__(
        self,
        spec: GranularitySpec,
        embed_dim: int,
        coarse_bias: float = 2.0,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.spec = spec
        self.coarse_bias = coarse_bias
        self.temperature = temperature
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(spec.levels[level] * embed_dim)
            for level in range(spec.depth)
        ])

    def level_embeddings(
        self, patch_tokens: torch.Tensor, cls_token: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        features = pyramid_features(patch_tokens, self.spec, cls_token=cls_token)
        return [
            decorrelate_level(feature, self.norms[level])
            for level, feature in enumerate(features)
        ]

    def forward(
        self,
        patch_tokens: torch.Tensor,
        labels: torch.Tensor,
        views: torch.Tensor,
        cls_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns the loss and a per-level breakdown for logging."""
        embeddings = self.level_embeddings(patch_tokens, cls_token)

        cross_view = (views[:, None] != views[None, :]).float()
        weights = level_weights_for_view(
            self.spec.depth, cross_view, self.coarse_bias
        )

        total = embeddings[0].sum() * 0.0
        breakdown = {}
        for level, z in enumerate(embeddings):
            level_loss = supcon_level(
                z, labels, weights[..., level], self.temperature
            )
            total = total + level_loss
            breakdown[f"L{level}"] = float(level_loss.detach())
        return total, breakdown
