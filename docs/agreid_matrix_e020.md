# Camera-pair retrieval matrix

Locates the view gap instead of assuming it. Every ordered pair of cameras that exists in AG-ReID.v2 is scored separately, one query per identity.

Checkpoint `outputs/transreid_curve/epoch_020.pth`, epoch 20, trained on `train_all`.

That training split contains all three cameras, so this model has already received cross-view supervision through its identity and triplet losses. Read the matrix as a statement about this trained model, not about the dataset in the abstract.

| query | gallery | platforms | mAP | Rank-1 | #IDs |
|---|---|---|---|---|---|
| C2 wearable | C0 aerial | aerial-ground | 61.62% | 73.80% | 519 |
| C3 cctv | C2 wearable | ground-ground | 63.32% | 74.29% | 245 |
| C2 wearable | C3 cctv | ground-ground | 64.48% | 78.37% | 245 |
| C3 cctv | C0 aerial | aerial-ground | 65.01% | 75.66% | 534 |
| C0 aerial | C2 wearable | aerial-ground | 65.58% | 78.61% | 519 |
| C0 aerial | C3 cctv | aerial-ground | 65.73% | 78.46% | 534 |

- spread across pairs: **4.10%**
- aerial<->ground mean: 64.48%
- ground<->ground mean: 63.90%

**Verdict: FLAT - no pair is meaningfully harder than another, so there is no view gap left for a loss to close with this checkpoint.**
