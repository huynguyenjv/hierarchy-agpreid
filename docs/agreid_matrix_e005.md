# Camera-pair retrieval matrix

Locates the view gap instead of assuming it. Every ordered pair of cameras that exists in AG-ReID.v2 is scored separately, one query per identity.

Checkpoint `outputs/transreid_curve/epoch_005.pth`, epoch 5, trained on `train_all`.

That training split contains all three cameras, so this model has already received cross-view supervision through its identity and triplet losses. Read the matrix as a statement about this trained model, not about the dataset in the abstract.

| query | gallery | platforms | mAP | Rank-1 | #IDs |
|---|---|---|---|---|---|
| C2 wearable | C0 aerial | aerial-ground | 54.15% | 66.86% | 519 |
| C3 cctv | C0 aerial | aerial-ground | 54.76% | 66.10% | 534 |
| C0 aerial | C3 cctv | aerial-ground | 56.00% | 69.66% | 534 |
| C3 cctv | C2 wearable | ground-ground | 56.46% | 68.57% | 245 |
| C2 wearable | C3 cctv | ground-ground | 56.98% | 71.84% | 245 |
| C0 aerial | C2 wearable | aerial-ground | 57.34% | 71.48% | 519 |

- spread across pairs: **3.20%**
- aerial<->ground mean: 55.56%
- ground<->ground mean: 56.72%

**Verdict: FLAT - no pair is meaningfully harder than another, so there is no view gap left for a loss to close with this checkpoint.**
