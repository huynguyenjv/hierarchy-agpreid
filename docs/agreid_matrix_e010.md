# Camera-pair retrieval matrix

Locates the view gap instead of assuming it. Every ordered pair of cameras that exists in AG-ReID.v2 is scored separately, one query per identity.

Checkpoint `outputs/transreid_curve/epoch_010.pth`, epoch 10, trained on `train_all`.

That training split contains all three cameras, so this model has already received cross-view supervision through its identity and triplet losses. Read the matrix as a statement about this trained model, not about the dataset in the abstract.

| query | gallery | platforms | mAP | Rank-1 | #IDs |
|---|---|---|---|---|---|
| C2 wearable | C0 aerial | aerial-ground | 58.06% | 70.33% | 519 |
| C3 cctv | C0 aerial | aerial-ground | 59.54% | 72.66% | 534 |
| C3 cctv | C2 wearable | ground-ground | 60.19% | 73.47% | 245 |
| C0 aerial | C2 wearable | aerial-ground | 61.51% | 73.60% | 519 |
| C2 wearable | C3 cctv | ground-ground | 61.96% | 74.29% | 245 |
| C0 aerial | C3 cctv | aerial-ground | 62.59% | 77.53% | 534 |

- spread across pairs: **4.53%**
- aerial<->ground mean: 60.43%
- ground<->ground mean: 61.08%

**Verdict: FLAT - no pair is meaningfully harder than another, so there is no view gap left for a loss to close with this checkpoint.**
