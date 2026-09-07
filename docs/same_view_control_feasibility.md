# Why a same-view control does not exist on AG-ReID.v2

*A result, not a logistics note. This is the evidence behind one of the
analysis claims, and it is also the reason CARGO was added.*

## The count

Identities whose images span enough cameras to support a same-platform,
cross-camera retrieval query:

| | AG-ReID.v2 | CARGO (gallery) |
|---|---|---|
| identities with **≥2 aerial cameras** | **0** | **2,351** |
| identities with **≥2 ground cameras** | 245 *(only after pooling all four protocols)* | **2,351** |
| identities with **both platforms** | 807 | 2,351 |
| aerial cameras in the dataset | 1 (C0) | 5 (Cam1–Cam5) |
| ground cameras in the dataset | 2 (C2 wearable, C3 CCTV) | 8 (Cam6–Cam13) |

Reproduce with `tools/camera_pair_matrix.py` and
`reid_advance.cargo.summarise`; asserted in
`tests/test_data_assumptions.py` and `tests/test_cargo.py`.

## What the zero means

Standard ReID evaluation discards gallery hits that share the query's camera
(`evaluation.py:76`), because retrieving another frame from the same camera is a
far easier problem than re-identification. A *same-view* score must therefore
still be *cross-camera*: two different cameras on the same platform.

AG-ReID.v2 has exactly one aerial camera. **Aerial-to-aerial retrieval is not
merely hard on this dataset — it does not exist.** No construction recovers it,
because the required image pairs were never captured.

The ground side is only marginally better. Each split of each official protocol
contains exactly one camera (`exp1` gallery is all C3, `exp4` gallery all C0),
so no identity has two same-platform cameras *within* a protocol. A ground
same-view control exists only after pooling images across all four protocol
files, which no published evaluation on this dataset does.

## Why this matters beyond bookkeeping

Any reported "same-view baseline" on AG-ReID.v2 must have done something
non-standard, because the standard construction yields nothing. This project
made exactly that mistake before catching it: an early version of the GSS probe
offset gallery camera ids by +100 to stop the same-camera filter from firing,
which silently converted the control into *within-camera* retrieval and
produced same-view mAP of 86.6% / 92.8% and an apparent +22.7% aerial-ground
gap. With the filter restored the same-view figure is 70.01% and the gap is
slightly **negative**.

That discarded number is kept here deliberately. It is the cheapest available
demonstration of how a same-view control on this dataset goes wrong, and the
error is invisible in aggregate metrics — the training curve, the loss and the
official protocol mAP all look entirely normal while it is happening.

## Why CARGO

Not because AG-ReID.v2 turned out to be disappointing, but because a claim about
where the view gap lives cannot rest on a dataset that structurally cannot
measure one of its own axes.

CARGO has five aerial and eight ground cameras, and 2,351 gallery identities
appear on two or more cameras of *each* platform. It therefore supports all
three comparisons:

- aerial ↔ ground (the axis the topic assumed was the hard one)
- aerial ↔ aerial (**impossible on AG-ReID.v2**)
- ground ↔ ground

If the flat camera-pair matrix measured on AG-ReID.v2 also appears on CARGO,
the finding holds across two datasets built under different capture regimes —
and it holds on an axis AG-ReID.v2 could not test at all.
