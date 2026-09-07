# AI-00 memory profile

- GPU: NVIDIA GeForce RTX 3060 (12.00 GiB)
- torch 2.5.1+cu121
- backbone: vit_small_patch16_224.augreg_in21k_ft_in1k (ViT-S/16)
- image size: (256, 128), K=8, AMP on
- train identities: 807
- CV-HWC overhead projected for L=4

| batch | P | JPM | peak alloc | peak reserved | +CV-HWC | projected | headroom |
|---|---|---|---|---|---|---|---|
| 32 | 4 | 1 | 1.04 | 1.13 | 0.000 | 1.13 | 10.87 |
| 64 | 8 | 1 | 1.74 | 1.92 | 0.000 | 1.92 | 10.08 |
| 96 | 12 | 1 | 2.43 | 2.66 | 0.000 | 2.66 | 9.34 |
| 128 | 16 | 1 | 3.10 | 3.35 | 0.001 | 3.35 | 8.65 |
| 192 | 24 | 1 | 4.46 | 4.77 | 0.002 | 4.77 | 7.23 |
| 256 | 32 | 1 | 5.83 | 6.19 | 0.003 | 6.19 | 5.81 |
| 384 | 48 | 1 | 8.60 | 9.10 | 0.007 | 9.11 | 2.89 |
| 32 | 4 | 0 | 0.85 | 1.00 | 0.000 | 1.00 | 11.00 |
| 64 | 8 | 0 | 1.55 | 1.75 | 0.000 | 1.75 | 10.25 |
| 96 | 12 | 0 | 2.40 | 2.62 | 0.000 | 2.62 | 9.38 |
| 128 | 16 | 0 | 2.91 | 3.22 | 0.001 | 3.22 | 8.78 |
| 192 | 24 | 0 | 4.27 | 4.64 | 0.002 | 4.64 | 7.36 |
| 256 | 32 | 0 | 5.64 | 6.02 | 0.003 | 6.02 | 5.98 |
| 384 | 48 | 0 | 8.40 | 8.86 | 0.007 | 8.87 | 3.13 |

`projected` adds the estimated CV-HWC pairwise-matrix cost to the measured backbone peak.

## Conclusion

Batch 128 (P=16 x K=8) uses only 3.35 GiB of 12 GiB, so VRAM does not constrain
the design: L=4 is affordable and batch 256 stays available as headroom if the
contrastive loss turns out to need more negatives. The CV-HWC pairwise matrices
are negligible at these batch sizes.
