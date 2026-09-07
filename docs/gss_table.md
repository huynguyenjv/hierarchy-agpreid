# Granularity Stability Score (Table 1)

Features read off a frozen `outputs/transreid/best_model.pth` at each level of the spatial granularity tree. No retraining: this only asks where in the pyramid the existing representation already transfers across views.

- patch grid 16x8, levels [1, 2, 4, 8] (whole, half, quarter, stripe)
- cross-view mAP uses the official protocol; same-view mAP restricts the gallery to the query's own platform

## exp1_aerial_to_cctv.txt

| level | regions | rows | cross-view mAP | cross-view R1 | same-view mAP | gap |
|---|---|---|---|---|---|---|
| L0 whole | 1 | 16 | 71.13% | 81.11% | 86.56% | +15.44% |
| L1 half | 2 | 8 | 71.59% | 81.24% | 86.41% | +14.82% |
| L2 quarter | 4 | 4 | 71.50% | 81.37% | 86.47% | +14.96% |
| L3 stripe | 8 | 2 | 71.41% | 81.28% | 86.44% | +15.03% |

Level-to-level distance-matrix correlation:

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| **L0** | 1.000 | 0.964 | 0.964 | 0.963 |
| **L1** | 0.964 | 1.000 | 1.000 | 1.000 |
| **L2** | 0.964 | 1.000 | 1.000 | 1.000 |
| **L3** | 0.963 | 1.000 | 1.000 | 1.000 |

Cross-view peaks at **L1** with a spread of only 0.46% across levels, and the levels correlate at 0.963 or above.

**Inconclusive.** The probe cannot decide the hypothesis: see the caveat below.

## exp4_cctv_to_aerial.txt

| level | regions | rows | cross-view mAP | cross-view R1 | same-view mAP | gap |
|---|---|---|---|---|---|---|
| L0 whole | 1 | 16 | 70.16% | 79.29% | 92.82% | +22.66% |
| L1 half | 2 | 8 | 70.04% | 78.58% | 92.68% | +22.64% |
| L2 quarter | 4 | 4 | 70.03% | 78.58% | 92.69% | +22.66% |
| L3 stripe | 8 | 2 | 70.03% | 78.63% | 92.65% | +22.61% |

Level-to-level distance-matrix correlation:

| | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| **L0** | 1.000 | 0.960 | 0.960 | 0.960 |
| **L1** | 0.960 | 1.000 | 1.000 | 1.000 |
| **L2** | 0.960 | 1.000 | 1.000 | 1.000 |
| **L3** | 0.960 | 1.000 | 1.000 | 1.000 |

Cross-view peaks at **L0** with a spread of only 0.13% across levels, and the levels correlate at 0.960 or above.

**Inconclusive.** The probe cannot decide the hypothesis: see the caveat below.

## What this probe can and cannot show

The hypothesis predicts the two mAP columns peak at different levels: coarse for cross-view, fine for same-view. A crossing would mean the optimal matching granularity depends on the view pair, which is what a granularity hierarchy exists to exploit.

But the correlation table above is the number to read first. Every level here pools the *same* patch tokens from a backbone trained with a single global objective, so the levels are near-duplicates of each other and their mAP differences are noise. A frozen-feature probe can therefore not falsify the hypothesis - it can only show whether granularity is *already* differentiated for free, and it is not.

Deciding the question requires levels that are trained to differ, i.e. a per-level objective. That is the experiment the multi-granularity loss is for; this table is the baseline it has to beat.
