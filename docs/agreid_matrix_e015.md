# Camera-pair retrieval matrix

Locates the view gap instead of assuming it. Every ordered pair of cameras that exists in AG-ReID.v2 is scored separately, one query per identity.

Checkpoint `outputs/transreid_curve/epoch_015.pth`, epoch 15, trained on `train_all`.

That training split contains all three cameras, so this model has already received cross-view supervision through its identity and triplet losses. Read the matrix as a statement about this trained model, not about the dataset in the abstract.

| query | gallery | platforms | mAP | Rank-1 | #IDs |
|---|---|---|---|---|---|
| C2 wearable | C0 aerial | aerial-ground | 60.64% | 71.48% | 519 |
| C3 cctv | C2 wearable | ground-ground | 62.00% | 73.88% | 245 |
| C3 cctv | C0 aerial | aerial-ground | 62.10% | 74.53% | 534 |
| C2 wearable | C3 cctv | ground-ground | 63.99% | 76.33% | 245 |
| C0 aerial | C2 wearable | aerial-ground | 64.98% | 78.61% | 519 |
| C0 aerial | C3 cctv | aerial-ground | 65.09% | 77.72% | 534 |

- spread across pairs: **4.45%**
- aerial<->ground mean: 63.20%
- ground<->ground mean: 62.99%

**Verdict: FLAT - no pair is meaningfully harder than another, so there is no view gap left for a loss to close with this checkpoint.**
