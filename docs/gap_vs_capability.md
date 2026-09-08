# Aerial-ground gap as a function of model capability

The plan was to compare both datasets at the same absolute mAP so that "CARGO has a gap, AG-ReID.v2 does not" could not be re-read as "the CARGO model is simply weaker". That comparison turned out to be impossible: CARGO tops out at 46.6% under this recipe while AG-ReID.v2 sits above 70%. mAP on CARGO went 42.1 -> 45.5 -> 46.9 -> 46.6 across epochs 10/20/25/30 while the train loss kept falling, so this is the ceiling of the recipe, not under-training. Same recipe, two different ceilings.

The underlying question survives without matched mAP. Measuring the gap at several checkpoints across each dataset's own capability range asks the same thing more directly: does more capability dissolve the gap, or not? A curve also resists the objection that any single matched operating point was chosen arbitrarily.

## CARGO

| epoch | overall mAP | aerial-ground | same-platform | gap |
|---|---|---|---|---|
| 5 | 21.61% | 33.63% | 47.53% | **+13.89%** |
| 10 | 42.06% | 54.67% | 68.82% | **+14.14%** |
| 15 | 41.88% | 56.72% | 71.82% | **+15.10%** |
| 20 | 45.50% | 60.59% | 74.58% | **+13.99%** |
| 25 | 46.88% | 61.42% | 75.42% | **+13.99%** |
| 30 | 46.56% | 61.47% | 75.64% | **+14.17%** |

Slope +0.009 gap per unit of mAP, end-to-end change +0.27% over a capability range of 21.61%-46.88%.

**FLAT against capability - the gap is intrinsic, not under-training**

## Reading the two curves together

If AG-ReID.v2's gap falls towards zero as its model strengthens while CARGO's holds flat across its own range, then the two datasets differ in kind and not merely in difficulty: ordinary supervision is enough to erase the view gap on one and not on the other. That is the causal claim, and it does not require the two models to sit at the same mAP.

If instead both curves fall and CARGO's has simply not fallen yet, the honest reading is that the gap is a function of supervision on both, and CARGO is only further from the point where it vanishes.
