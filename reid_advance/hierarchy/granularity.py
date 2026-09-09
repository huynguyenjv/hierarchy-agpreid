"""Spatial granularity hierarchy for aerial-ground ReID.

The attribute experiment (AI-03, ``docs/vss_table.md``) found that only gender
survives as a usable semantic level, and that retention from ground to aerial
sits near 1.0 for almost every attribute.  So the view gap is not attributes
being destroyed by altitude.  This module follows the other reading: the gap is
about **spatial granularity**.

An aerial crop keeps whole-body structure - silhouette, limb proportions, gait
posture - while losing local detail.  A ground crop keeps texture, logos and
face while its global shape is foreshortened by the low camera.  A representation
forced into one flat granularity has to trade one against the other; organising
it coarse-to-fine lets cross-view matching happen at the level both views can
actually share.

The tree needs no annotation at all.  It comes from the patch grid: a 256x128
input at patch 16 gives a 16x8 token grid, and splitting the rows gives nested
horizontal stripes that follow the body.

    L0  whole body   1 region   16 rows   CLS token
    L1  half body    2 regions   8 rows   upper / lower
    L2  quarter      4 regions   4 rows   head / torso / hips / legs
    L3  stripe       8 regions   2 rows   fine detail

Level l is a strict refinement of level l-1, which is what makes it a tree and
what lets a coarse level act as the fallback when the fine one stops
transferring across views.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


#: Number of horizontal regions at each level of the granularity tree.
#: Each entry must divide the patch-grid row count and refine the previous one.
GRANULARITY_LEVELS: tuple[int, ...] = (1, 2, 4, 8)

LEVEL_NAMES: tuple[str, ...] = ("whole", "half", "quarter", "stripe")


@dataclass(frozen=True)
class GranularitySpec:
    """Geometry of the granularity tree for one input resolution."""

    rows: int
    cols: int
    levels: tuple[int, ...] = GRANULARITY_LEVELS

    @property
    def depth(self) -> int:
        return len(self.levels)

    def rows_per_region(self, level: int) -> int:
        return self.rows // self.levels[level]

    def validate(self) -> None:
        previous = 0
        for index, count in enumerate(self.levels):
            if self.rows % count:
                raise ValueError(
                    f"level {index} splits {self.rows} rows into {count} regions, "
                    "which does not divide evenly"
                )
            if previous and count % previous:
                raise ValueError(
                    f"level {index} ({count} regions) does not refine the previous "
                    f"level ({previous} regions); the levels must form a tree"
                )
            previous = count


def spec_for_image(image_size: tuple[int, int], patch_size: int = 16,
                   levels: tuple[int, ...] = GRANULARITY_LEVELS) -> GranularitySpec:
    height, width = image_size
    spec = GranularitySpec(height // patch_size, width // patch_size, levels)
    spec.validate()
    return spec


def split_tokens_by_level(
    patch_tokens: torch.Tensor, spec: GranularitySpec, level: int
) -> torch.Tensor:
    """Average patch tokens inside each horizontal region of one level.

    ``patch_tokens`` is (B, rows*cols, D) in row-major order, excluding CLS.
    Returns (B, regions, D).
    """
    batch, count, dim = patch_tokens.shape
    expected = spec.rows * spec.cols
    if count != expected:
        raise ValueError(
            f"expected {expected} patch tokens for a {spec.rows}x{spec.cols} grid, "
            f"got {count}"
        )
    regions = spec.levels[level]
    grid = patch_tokens.reshape(batch, spec.rows, spec.cols, dim)
    # Group whole rows so every region is a contiguous horizontal band.
    grid = grid.reshape(batch, regions, spec.rows // regions * spec.cols, dim)
    return grid.mean(dim=2)


def pyramid_features(
    patch_tokens: torch.Tensor,
    spec: GranularitySpec,
    cls_token: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Features at every level of the tree, coarse to fine.

    Level 0 uses the CLS token when available: it is the model's own global
    summary and a better whole-body descriptor than averaging every patch.
    """
    features = []
    for level in range(spec.depth):
        if level == 0 and cls_token is not None and spec.levels[0] == 1:
            features.append(cls_token.unsqueeze(1))
        else:
            features.append(split_tokens_by_level(patch_tokens, spec, level))
    return features


def flatten_level(level_features: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """(B, regions, D) -> (B, regions*D), with each region normalised first.

    Normalising per region keeps one high-norm body part from dominating the
    concatenated descriptor.

    .. warning::
       This alone is **not** enough to build a contrastive loss on. Raw ViT
       tokens carry a large shared component - measured on a real batch, every
       pair sits at cosine 0.993 +/- 0.001 whichever level it is pooled at, and
       token norms are around 49. A SupCon term over these would see near
       identical logits at tau=0.07, produce almost no gradient, and train
       quietly to no effect. Retrieval works only because the ID head puts the
       features through BNNeck first (its output sits at 0.0025 +/- 0.088).

       Use :func:`decorrelate_level` before any similarity is taken. This
       function is left as-is because the frozen-feature probes already
       published used it and their numbers must stay reproducible.
    """
    if normalize:
        level_features = F.normalize(level_features, dim=-1)
    flattened = level_features.flatten(1)
    return F.normalize(flattened, dim=-1) if normalize else flattened


def decorrelate_level(
    level_features: torch.Tensor, norm: torch.nn.Module | None = None
) -> torch.Tensor:
    """Strip the shared component so a similarity has something to measure.

    ``norm`` is a per-level ``BatchNorm1d``, mirroring the BNNeck the identity
    head already relies on; pass the module so its statistics are learned and
    shared across steps. Without one, the batch mean is subtracted directly,
    which is enough for analysis but not for training.

    Measured effect on one real batch of 128, at every level:

        before   cosine 0.993 +/- 0.001   (nothing to discriminate)
        after    cosine 0.000 +/- 0.09    (usable)

    The resulting spread is near identical across levels (0.089 to 0.092), so a
    per-level weighting acts on granularity rather than compensating for the
    levels' different dimensionalities (384 up to 3072).
    """
    flattened = level_features.flatten(1)
    if norm is not None:
        flattened = norm(flattened)
    else:
        flattened = flattened - flattened.mean(dim=0, keepdim=True)
    return F.normalize(flattened, dim=-1)


def level_weights_for_view(
    depth: int,
    cross_view: torch.Tensor,
    coarse_bias: float = 2.0,
) -> torch.Tensor:
    """Per-level weights that shift cross-view pairs towards coarse levels.

    This is the granularity analogue of the cross-view reweighting term: a pair
    photographed from the same platform can be matched on fine detail, while an
    aerial-ground pair should be pushed together using the level both views
    still agree on.

    ``cross_view`` is a (B, B) float mask, 1 where the two samples come from
    different platforms.  Returns (B, B, depth) weights summing to 1 per pair.
    """
    device = cross_view.device
    # Ramp from coarse (level 0) to fine (level depth-1).
    ramp = torch.linspace(0.0, 1.0, depth, device=device)
    same_view_logits = ramp * coarse_bias           # favour fine detail
    cross_view_logits = (1.0 - ramp) * coarse_bias  # favour coarse structure

    same = torch.softmax(same_view_logits, dim=0)
    cross = torch.softmax(cross_view_logits, dim=0)

    mask = cross_view.unsqueeze(-1)
    return mask * cross + (1.0 - mask) * same
