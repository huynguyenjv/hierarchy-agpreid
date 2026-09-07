# Camera-pair retrieval matrix

Locates the view gap instead of assuming it. Every ordered pair of cameras that exists in AG-ReID.v2 is scored separately, one query per identity.

**Randomly initialised backbone** (contrast run).

| query | gallery | platforms | mAP | Rank-1 | #IDs |
|---|---|---|---|---|---|
| C2 wearable | C0 aerial | aerial-ground | 0.93% | 1.73% | 519 |
| C3 cctv | C0 aerial | aerial-ground | 1.15% | 3.18% | 534 |
| C3 cctv | C2 wearable | ground-ground | 1.26% | 4.08% | 245 |
| C2 wearable | C3 cctv | ground-ground | 1.33% | 3.27% | 245 |
| C0 aerial | C2 wearable | aerial-ground | 1.37% | 3.66% | 519 |
| C0 aerial | C3 cctv | aerial-ground | 1.58% | 4.12% | 534 |

- spread across pairs: **0.64%**
- aerial<->ground mean: 1.26%
- ground<->ground mean: 1.30%

**Verdict: FLAT - no pair is meaningfully harder than another, so there is no view gap left for a loss to close with this checkpoint.**
