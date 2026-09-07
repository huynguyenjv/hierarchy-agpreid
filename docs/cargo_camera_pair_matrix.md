# CARGO camera-pair retrieval matrix

Checkpoint `outputs/cargo/epoch_030.pth` (epoch 30), trained with the same recipe as the AG-ReID.v2 baseline: TransReID-S ViT-S/16, identity + triplet loss, PK sampler, both platforms in the training split. Same recipe is what makes the two matrices comparable.

Scored on the `gallery` split, one query image per identity per pair.

## By pair kind

| kind | pairs | mean mAP | min | max |
|---|---|---|---|---|
| aerial-ground | 80 | 61.47% | 37.70% | 75.61% |
| aerial-aerial | 20 | 67.14% | 45.44% | 79.20% |
| ground-ground | 56 | 78.67% | 62.02% | 88.97% |

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
| Cam9 | Cam1 | aerial-ground | 37.70% | 41.00% | 600 | 1805 |
| Cam1 | Cam9 | aerial-ground | 41.32% | 40.17% | 600 | 1805 |
| Cam11 | Cam1 | aerial-ground | 42.96% | 48.67% | 600 | 1778 |
| Cam8 | Cam1 | aerial-ground | 45.40% | 52.83% | 600 | 1799 |
| Cam5 | Cam1 | aerial-aerial | 45.44% | 50.67% | 600 | 1409 |
| Cam13 | Cam1 | aerial-ground | 45.75% | 51.17% | 600 | 1824 |
| Cam1 | Cam11 | aerial-ground | 46.06% | 43.67% | 600 | 1778 |
| Cam12 | Cam1 | aerial-ground | 46.58% | 52.67% | 600 | 1713 |
| Cam10 | Cam8 | ground-ground | 87.16% | 93.00% | 600 | 1467 |
| Cam8 | Cam6 | ground-ground | 87.37% | 90.50% | 600 | 1467 |
| Cam12 | Cam10 | ground-ground | 88.05% | 89.50% | 600 | 1401 |
| Cam10 | Cam12 | ground-ground | 88.97% | 89.83% | 600 | 1401 |

- spread across all 156 pairs: **51.27%**
- aerial-ground mean 61.47%, same-platform mean 75.64%, gap **+14.17%**

**Verdict: CARGO RETAINS an aerial-ground gap: same-platform retrieval beats cross-platform by 14.17%.**
