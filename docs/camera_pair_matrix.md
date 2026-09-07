# Camera-pair retrieval matrix

Locates the view gap instead of assuming it. Every ordered pair of cameras that exists in AG-ReID.v2 is scored separately, one query per identity.

Checkpoint `outputs/transreid/best_model.pth`, epoch 125, trained on `train_all`.

That training split contains all three cameras, so this model has already received cross-view supervision through its identity and triplet losses. Read the matrix as a statement about this trained model, not about the dataset in the abstract.

| query | gallery | platforms | mAP | Rank-1 | #IDs |
|---|---|---|---|---|---|
| C3 cctv | C0 aerial | aerial-ground | 71.07% | 79.78% | 534 |
| C2 wearable | C0 aerial | aerial-ground | 71.19% | 80.92% | 519 |
| C3 cctv | C2 wearable | ground-ground | 71.26% | 80.00% | 245 |
| C2 wearable | C3 cctv | ground-ground | 71.92% | 82.86% | 245 |
| C0 aerial | C3 cctv | aerial-ground | 73.89% | 86.14% | 534 |
| C0 aerial | C2 wearable | aerial-ground | 76.56% | 84.39% | 519 |

- spread across pairs: **5.49%**
- aerial<->ground mean: 73.18%
- ground<->ground mean: 71.59%

**Verdict: UNEVEN but not along the aerial-ground axis; hardest pair is C3->C0.**
