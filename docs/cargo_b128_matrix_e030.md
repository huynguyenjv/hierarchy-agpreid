# CARGO camera-pair retrieval matrix

Checkpoint `outputs/cargo_b128/epoch_030.pth` (epoch 30), trained with the same recipe as the AG-ReID.v2 baseline: TransReID-S ViT-S/16, identity + triplet loss, PK sampler, both platforms in the training split. Same recipe is what makes the two matrices comparable.

Scored on the `gallery` split, one query image per identity per pair.

## By pair kind

| kind | pairs | mean mAP | min | max |
|---|---|---|---|---|
| aerial-ground | 80 | 55.33% | 30.95% | 68.56% |
| aerial-aerial | 20 | 60.69% | 39.01% | 73.84% |
| ground-ground | 56 | 75.17% | 59.22% | 87.06% |

AG-ReID.v2 could not report an `aerial-aerial` row at all: it has a single aerial camera, so aerial same-view retrieval does not exist there. CARGO's five aerial cameras make this the stricter test.

## Shared identities per camera pair

Read this before the mAP table. A pair sharing few identities has almost no true matches to retrieve, so its mAP would measure an empty gallery rather than view difficulty - the same failure mode that produced the discarded AG-ReID.v2 same-view figures. Cells below 100 shared identities are excluded from every aggregate above.

| | Cam1 | Cam2 | Cam3 | Cam4 | Cam5 | Cam6 | Cam7 | Cam8 | Cam9 | Cam10 | Cam11 | Cam12 | Cam13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Cam1** | - | 1402 | 1392 | 1397 | 1409 | 1786 | 1796 | 1799 | 1805 | 1779 | 1778 | 1713 | 1824 |
| **Cam2** | 1402 | - | 1239 | 879 | 745 | 1288 | 1267 | 1223 | 1276 | 1277 | 1281 | 1184 | 1253 |
| **Cam3** | 1392 | 1239 | - | 1173 | 900 | 1261 | 1270 | 1197 | 1253 | 1234 | 1287 | 1161 | 1261 |
| **Cam4** | 1397 | 879 | 1173 | - | 1261 | 1243 | 1276 | 1262 | 1219 | 1226 | 1275 | 1190 | 1262 |
| **Cam5** | 1409 | 745 | 900 | 1261 | - | 1240 | 1292 | 1292 | 1224 | 1248 | 1271 | 1217 | 1269 |
| **Cam6** | 1786 | 1288 | 1261 | 1243 | 1240 | - | 1625 | 1467 | 1465 | 1463 | 1467 | 1398 | 1614 |
| **Cam7** | 1796 | 1267 | 1270 | 1276 | 1292 | 1625 | - | 1594 | 1488 | 1479 | 1482 | 1423 | 1506 |
| **Cam8** | 1799 | 1223 | 1197 | 1262 | 1292 | 1467 | 1594 | - | 1629 | 1467 | 1466 | 1410 | 1494 |
| **Cam9** | 1805 | 1276 | 1253 | 1219 | 1224 | 1465 | 1488 | 1629 | - | 1602 | 1467 | 1405 | 1491 |
| **Cam10** | 1779 | 1277 | 1234 | 1226 | 1248 | 1463 | 1479 | 1467 | 1602 | - | 1609 | 1401 | 1479 |
| **Cam11** | 1778 | 1281 | 1287 | 1275 | 1271 | 1467 | 1482 | 1466 | 1467 | 1609 | - | 1535 | 1484 |
| **Cam12** | 1713 | 1184 | 1161 | 1190 | 1217 | 1398 | 1423 | 1410 | 1405 | 1401 | 1535 | - | 1551 |
| **Cam13** | 1824 | 1253 | 1261 | 1262 | 1269 | 1614 | 1506 | 1494 | 1491 | 1479 | 1484 | 1551 | - |

Range: 745 to 1824 identities per pair. Every pair clears the floor, so no cell in the mAP table is an empty-gallery artifact.

## Hardest and easiest pairs

| query | gallery | kind | mAP | Rank-1 | #IDs scored | shared IDs |
|---|---|---|---|---|---|---|
| Cam9 | Cam1 | aerial-ground | 30.95% | 33.33% | 600 | 1805 |
| Cam1 | Cam9 | aerial-ground | 33.63% | 32.83% | 600 | 1805 |
| Cam11 | Cam1 | aerial-ground | 35.59% | 38.83% | 600 | 1778 |
| Cam13 | Cam1 | aerial-ground | 38.41% | 43.83% | 600 | 1824 |
| Cam8 | Cam1 | aerial-ground | 38.93% | 44.17% | 600 | 1799 |
| Cam5 | Cam1 | aerial-aerial | 39.01% | 43.00% | 600 | 1409 |
| Cam12 | Cam1 | aerial-ground | 39.40% | 44.00% | 600 | 1713 |
| Cam6 | Cam1 | aerial-ground | 40.37% | 45.00% | 600 | 1786 |
| Cam13 | Cam12 | ground-ground | 84.98% | 85.17% | 600 | 1551 |
| Cam12 | Cam10 | ground-ground | 85.18% | 87.17% | 600 | 1401 |
| Cam8 | Cam6 | ground-ground | 85.43% | 88.83% | 600 | 1467 |
| Cam10 | Cam12 | ground-ground | 87.06% | 87.00% | 600 | 1401 |

- spread across all 156 pairs: **56.11%**
- aerial-ground mean 55.33%, same-platform mean 71.36%, gap **+16.03%**

**Verdict: CARGO RETAINS an aerial-ground gap: same-platform retrieval beats cross-platform by 16.03%.**
