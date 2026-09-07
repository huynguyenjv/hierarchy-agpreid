# BA-02/BA-03 attribute audit

- MAT: `AG-ReID.v2/qut_attribute_v8.mat` (807 rows)
- Split: `train_all` (51,530 images, 807 identities, P+T+A scheme)

## 1. Coverage and identity-level stability

| check | value |
|---|---|
| identities resolved to a MAT row | 807 / 807 (100.0%) |
| images without a MAT row | 0 |
| distinct unmatched track keys | 0 |
| identities with >1 attribute vector | 0 |
| identities with both aerial and ground images | 807 (100.0%) |

**Identity-level stability assumption: HOLDS.** Every image of an identity shares one attribute vector, so a branch of the tree is identical for the aerial and the ground images of that person - this is what lets the hierarchy act as a view-invariant bridge (KB 2.2).

The last row also matters for the sampler: only identities holding both aerial and ground images can ever produce a cross-view positive pair, which is what the beta_cross term of CV-HWC reweights.

## 2. Per-group statistics (identity level)

| group | classes | unknown % | entropy (bits) | largest class % | distribution |
|---|---|---|---|---|---|
| gender | 3 | 0.6% | 1.00 | 53.4% | [428, 374] |
| age | 4 | 0.2% | 0.31 | 95.4% | [768, 20, 17] |
| height | 5 | 0.1% | 1.64 | 37.3% | [10, 195, 300, 301] |
| weight | 4 | 0.0% | 1.37 | 54.5% | [270, 440, 97] |
| ethnic | 5 | 1.1% | 1.16 | 71.6% | [571, 26, 172, 29] |
| haircolor | 7 | 2.7% | 1.28 | 48.5% | [368, 381, 16, 0, 17, 3] |
| hairstyle | 6 | 1.2% | 1.90 | 49.3% | [28, 393, 73, 143, 160] |
| beard | 3 | 0.9% | 0.33 | 94.0% | [48, 752] |
| moustache | 3 | 0.9% | 0.38 | 92.7% | [58, 741] |
| glasses | 4 | 5.8% | 1.20 | 69.1% | [100, 134, 523] |
| head | 5 | 85.1% | 0.56 | 90.7% | [107, 7, 1, 3] |
| upper | 13 | 0.0% | 2.10 | 58.2% | [470, 90, 19, 8, 21, 2, 48, 14, 110, 7, 10, 8] |
| lower | 10 | 0.4% | 2.32 | 32.8% | [122, 43, 264, 256, 40, 0, 62, 14, 3] |
| feet | 7 | 0.1% | 1.38 | 69.9% | [563, 128, 15, 33, 67, 0] |
| bag | 9 | 3.2% | 1.98 | 47.4% | [157, 370, 94, 2, 1, 8, 12, 137] |

A group is a poor tree level when the unknown rate is high, the entropy is near zero, or one class covers almost everything (KB 6.1 criterion 2).

## 3. Mutual information between groups (bits)

| | gender | age | height | weight | ethnic | haircolor | hairstyle | beard | moustache | glasses | head | upper | lower | feet | bag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **gender** | 1.04 | 0.01 | 0.10 | 0.00 | 0.01 | 0.01 | 0.55 | 0.05 | 0.07 | 0.02 | 0.03 | 0.28 | 0.26 | 0.04 | 0.13 |
| **age** | 0.01 | 0.34 | 0.02 | 0.00 | 0.02 | 0.13 | 0.03 | 0.01 | 0.02 | 0.03 | 0.02 | 0.03 | 0.02 | 0.01 | 0.03 |
| **height** | 0.10 | 0.02 | 1.65 | 0.07 | 0.07 | 0.04 | 0.08 | 0.02 | 0.03 | 0.04 | 0.02 | 0.07 | 0.07 | 0.04 | 0.07 |
| **weight** | 0.00 | 0.00 | 0.07 | 1.37 | 0.03 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.04 | 0.03 | 0.02 | 0.02 |
| **ethnic** | 0.01 | 0.02 | 0.07 | 0.03 | 1.23 | 0.32 | 0.03 | 0.03 | 0.03 | 0.08 | 0.02 | 0.05 | 0.05 | 0.03 | 0.03 |
| **haircolor** | 0.01 | 0.13 | 0.04 | 0.01 | 0.32 | 1.43 | 0.10 | 0.02 | 0.02 | 0.04 | 0.03 | 0.05 | 0.03 | 0.02 | 0.04 |
| **hairstyle** | 0.55 | 0.03 | 0.08 | 0.02 | 0.03 | 0.10 | 1.97 | 0.05 | 0.05 | 0.03 | 0.07 | 0.22 | 0.21 | 0.06 | 0.13 |
| **beard** | 0.05 | 0.01 | 0.02 | 0.01 | 0.03 | 0.02 | 0.05 | 0.40 | 0.27 | 0.04 | 0.01 | 0.03 | 0.02 | 0.01 | 0.03 |
| **moustache** | 0.07 | 0.02 | 0.03 | 0.01 | 0.03 | 0.02 | 0.05 | 0.27 | 0.46 | 0.05 | 0.02 | 0.04 | 0.02 | 0.01 | 0.03 |
| **glasses** | 0.02 | 0.03 | 0.04 | 0.01 | 0.08 | 0.04 | 0.03 | 0.04 | 0.05 | 1.48 | 0.04 | 0.03 | 0.03 | 0.02 | 0.03 |
| **head** | 0.03 | 0.02 | 0.02 | 0.01 | 0.02 | 0.03 | 0.07 | 0.01 | 0.02 | 0.04 | 0.71 | 0.05 | 0.04 | 0.02 | 0.04 |
| **upper** | 0.28 | 0.03 | 0.07 | 0.04 | 0.05 | 0.05 | 0.22 | 0.03 | 0.04 | 0.03 | 0.05 | 2.10 | 0.58 | 0.21 | 0.15 |
| **lower** | 0.26 | 0.02 | 0.07 | 0.03 | 0.05 | 0.03 | 0.21 | 0.02 | 0.02 | 0.03 | 0.04 | 0.58 | 2.35 | 0.22 | 0.13 |
| **feet** | 0.04 | 0.01 | 0.04 | 0.02 | 0.03 | 0.02 | 0.06 | 0.01 | 0.01 | 0.02 | 0.02 | 0.21 | 0.22 | 1.39 | 0.06 |
| **bag** | 0.13 | 0.03 | 0.07 | 0.02 | 0.03 | 0.04 | 0.13 | 0.03 | 0.03 | 0.03 | 0.04 | 0.15 | 0.13 | 0.06 | 2.12 |

Top redundant pairs:

- `upper` x `lower`: 0.579
- `gender` x `hairstyle`: 0.555
- `ethnic` x `haircolor`: 0.317
- `gender` x `upper`: 0.278
- `beard` x `moustache`: 0.273
- `gender` x `lower`: 0.257
- `lower` x `feet`: 0.221
- `hairstyle` x `upper`: 0.216

Avoid stacking two high-MI groups as consecutive levels; the second one would barely subdivide the first (KB 6.1 criterion 3).
