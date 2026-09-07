# AI-03 - View Stability Score (Table 1)

One `vit_small_patch16_224.augreg_in21k_ft_in1k` backbone with 15 attribute heads, trained for 12 epochs on **ground images only**, evaluated on **identity-disjoint** held-out images.

- identities: 564 train / 243 test, no overlap
- images: 21,593 train-ground, 8,720 test-ground, 5,989 test-aerial
- VSS = balanced accuracy on aerial images (threshold 0.65)

Balanced accuracy is used instead of raw accuracy because several groups are dominated by one class; raw accuracy would reward a model that always predicts the majority. `lift` shows aerial raw accuracy minus that majority baseline.

| attribute | VSS (aerial bal.) | ground bal. | retention | aerial raw | majority | lift | usable as level |
|---|---|---|---|---|---|---|---|
| `gender` | **0.742** | 0.836 | 0.89 | 0.745 | 0.508 | +0.237 | yes |
| `moustache` | **0.502** | 0.503 | 1.00 | 0.908 | 0.913 | -0.006 | no |
| `beard` | **0.500** | 0.508 | 0.98 | 0.921 | 0.923 | -0.002 | no |
| `head` | **0.478** | 0.448 | 1.07 | 0.893 | 0.543 | +0.350 | no |
| `lower` | **0.460** | 0.536 | 0.86 | 0.531 | 0.364 | +0.167 | no |
| `hairstyle` | **0.412** | 0.424 | 0.97 | 0.556 | 0.505 | +0.051 | no |
| `weight` | **0.401** | 0.396 | 1.01 | 0.464 | 0.500 | -0.036 | no |
| `height` | **0.390** | 0.417 | 0.93 | 0.376 | 0.381 | -0.005 | no |
| `age` | **0.365** | 0.491 | 0.74 | 0.955 | 0.973 | -0.019 | no |
| `feet` | **0.358** | 0.384 | 0.93 | 0.753 | 0.713 | +0.040 | no |
| `glasses` | **0.342** | 0.378 | 0.90 | 0.578 | 0.657 | -0.079 | no |
| `ethnic` | **0.331** | 0.353 | 0.94 | 0.627 | 0.668 | -0.042 | no |
| `upper` | **0.266** | 0.302 | 0.88 | 0.599 | 0.535 | +0.064 | no |
| `bag` | **0.263** | 0.290 | 0.90 | 0.502 | 0.386 | +0.115 | no |
| `haircolor` | **0.246** | 0.256 | 0.96 | 0.552 | 0.459 | +0.093 | no |

**Reading the table.** `retention` is aerial divided by ground balanced accuracy: it isolates how much of the signal the climb destroys, independently of how hard the attribute is in the first place. An attribute belongs near the root when both VSS and retention are high (KB 6.2).
