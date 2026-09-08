# Aerial-ground gap as a function of model capability

The plan was to compare both datasets at the same absolute mAP so that "CARGO has a gap, AG-ReID.v2 does not" could not be re-read as "the CARGO model is simply weaker". That comparison turned out to be impossible: CARGO tops out at 46.6% under this recipe while AG-ReID.v2 sits above 70%. mAP on CARGO went 42.1 -> 45.5 -> 46.9 -> 46.6 across epochs 10/20/25/30 while the train loss kept falling, so this is the ceiling of the recipe, not under-training. Same recipe, two different ceilings.

The underlying question survives without matched mAP. Measuring the gap at several checkpoints across each dataset's own capability range asks the same thing more directly: does more capability dissolve the gap, or not? A curve also resists the objection that any single matched operating point was chosen arbitrarily.

## CARGO

| epoch | overall mAP | aerial-ground | same-platform | gap | ground-ground | gap (ground only) |
|---|---|---|---|---|---|---|
| 5 | 21.61% | 33.63% | 47.53% | **+13.89%** | 51.90% | **+18.27%** |
| 10 | 42.06% | 54.67% | 68.82% | **+14.14%** | 72.65% | **+17.98%** |
| 15 | 41.88% | 56.72% | 71.82% | **+15.10%** | 75.12% | **+18.40%** |
| 20 | 45.50% | 60.59% | 74.58% | **+13.99%** | 77.70% | **+17.11%** |
| 25 | 46.88% | 61.42% | 75.42% | **+13.99%** | 78.58% | **+17.16%** |
| 30 | 46.56% | 61.47% | 75.64% | **+14.17%** | 78.67% | **+17.20%** |

- `all_same_platform`: slope +0.009 per unit of mAP, change +0.27% over mAP 21.61%-46.88%. HOLDS - the gap keeps its size across the capability range, so it is intrinsic rather than a training artifact

- `ground_only`: slope -0.040 per unit of mAP, change -1.07% over mAP 21.61%-46.88%. HOLDS - the gap keeps its size across the capability range, so it is intrinsic rather than a training artifact

## AG-ReID.v2

| epoch | overall mAP | aerial-ground | same-platform | gap | ground-ground | gap (ground only) |
|---|---|---|---|---|---|---|
| 5 | 58.08% | 55.56% | 56.72% | **+1.16%** | 56.72% | **+1.16%** |
| 10 | 65.19% | 60.43% | 61.08% | **+0.65%** | 61.08% | **+0.65%** |
| 15 | 68.21% | 63.20% | 62.99% | **-0.21%** | 62.99% | **-0.21%** |
| 20 | 70.35% | 64.48% | 63.90% | **-0.59%** | 63.90% | **-0.59%** |

- `all_same_platform`: slope -0.141 per unit of mAP, change -1.74% over mAP 58.08%-70.35%. DISSOLVES - the gap starts positive and ends negative; supervision removes it entirely

- `ground_only`: slope -0.141 per unit of mAP, change -1.74% over mAP 58.08%-70.35%. DISSOLVES - the gap starts positive and ends negative; supervision removes it entirely

## Reading the two curves together

If AG-ReID.v2's gap falls towards zero as its model strengthens while CARGO's holds flat across its own range, then the two datasets differ in kind and not merely in difficulty: ordinary supervision is enough to erase the view gap on one and not on the other. That is the causal claim, and it does not require the two models to sit at the same mAP.

If instead both curves fall and CARGO's has simply not fallen yet, the honest reading is that the gap is a function of supervision on both, and CARGO is only further from the point where it vanishes.

## Two definitions of the gap

The datasets are not built alike: AG-ReID.v2 has one aerial camera and therefore no aerial-aerial row, while CARGO has five. Comparing CARGO's full same-platform mean against AG-ReID.v2's would compare different quantities, so both are reported.

- **gap** - aerial-ground against all same-platform pairs, pair-weighted. Uses everything a dataset offers.
- **gap (ground only)** - aerial-ground against ground-ground alone. The one axis both datasets can produce, so this is the like-for-like comparison.

If the two disagree about the shape of a curve, the ground-only definition governs any cross-dataset claim.
