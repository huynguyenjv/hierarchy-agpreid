# AG-ReID.v2 Baselines and Semi-Supervised Temporal ReID

This repository contains two supervised baselines, a zero-label SSL baseline,
and two independent semi-supervised proposals for aerial-ground person
re-identification on AG-ReID.v2.

## Dataset layout

```text
AG-ReID.v2/
├── train_all/
├── query/
├── gallery/
├── qut_attribute_v8.mat
├── exp1_aerial_to_cctv.txt
├── exp2_aerial_to_wearable.txt
├── exp4_cctv_to_aerial.txt
└── exp5_wearable_to_aerial.txt
```

The default protocol is aerial-to-CCTV. Change `eval_txt_file` in the relevant
config profile to run another protocol.

### Dataset identity integrity

AG-ReID.v2 identity labels are the composite `P + T + A` fields. `P` alone is
only unique inside the current video and must never be used as the classifier
label. Every training and evaluation lane uses the shared parser in
`reid_advance/identity.py` and fails fast unless the official local dataset
audits pass:

- `train_all`: 51,530 images and 807 identities;
- a 25% identity budget: 202 of 807 identities with seed 42;
- protocol 1 query/gallery: 534 identities in each split and no train/test PID
  overlap.

Supervised checkpoints created before this parser fix have no
`identity_scheme` metadata and are intentionally rejected by staged training
and `tools/compare_experiments.py`. Keep them only as archived debugging
artifacts; rerun BNNeck, TransReID, PersonViT fine-tuning, and Proposed V1-V3.

## Project structure

```text
ReID_Advance/
├── AG-ReID.v2/               # Dataset and official protocols
├── reid_advance/
│   ├── config.py             # Three independent experiment profiles
│   ├── data.py               # Datasets, attributes and P×K sampler
│   ├── evaluation.py         # mAP and CMC
│   ├── losses.py             # Triplet and masked attribute losses
│   ├── runtime.py            # Shared loaders, scheduler and evaluation
│   └── pipelines/
│       ├── bnneck.py
│       ├── transreid.py
│       └── proposed.py
├── docs/project.md           # Research design and objective
├── tools/inspect_attributes.py
├── legacy/                   # Archived VICReg/reranking experiments
├── outputs/
└── run.py                    # Single launcher
```

## Independent experiment lanes

Each member edits only their dataclass profile in `reid_advance/config.py` and uses a
separate command and output directory.

| Member | Method | Config | Command | Output |
|---|---|---|---|---|
| 1 | ViT-S/16 + BNNeck | `BNNeckConfig` | `python run.py bnneck` | `outputs/bnneck/` |
| Stage 1 | PersonViT-S/16 supervised fine-tune (25% PID labels) | `PersonViTFineTuneConfig` | `python run.py personvit_ft` | `outputs/personvit_ft_25/` |
| 2 | TransReID-S | `TransReIDConfig` | `python run.py transreid` | `outputs/transreid/` |
| SSL baseline | Local UntransReID adaptation | `UntransReIDConfig` | `python run.py ssl` | `outputs/untransreid/` |
| 3a | Original LeWM-inspired proposal | `ProposedConfig` | `python run.py proposed` | `outputs/proposed_lewm/` |
| 3b | Cluster-enhanced BNNeck proposal | `ProposedV2Config` | `python run.py proposed_v2` | `outputs/proposed_v2/` |
| 3c | EMA + JPM + re-ranked clustering refinement | `ProposedV3Config` | `python run.py proposed_v3` | `outputs/proposed_v3/` |
| 3d | Low-LR V3 continuation | `ProposedV3FineTuneConfig` | `python run.py proposed_v3_ft` | `outputs/proposed_v3_finetune/` |

Each lane keeps exactly one checkpoint named `best_model.pth` in its own output
directory. Later improvements overwrite that file; epoch-by-epoch checkpoints
are not retained. Files inside each `tb/` directory are TensorBoard logs, not
model checkpoints.

Temporary overrides do not require editing source files:

```powershell
python run.py transreid --epochs 30 --batch-size 4
python run.py ssl --epochs 30 --batch-size 16 --dbscan-eps 0.35
python run.py ssl --ssl-checkpoint pretrained/personvit_vits16.pth
python run.py proposed --labeled-fraction 0.10
python run.py personvit_ft --labeled-fraction 0.25
python run.py proposed_v2 --labeled-fraction 0.25
python run.py proposed_v3 --batch-size 32
python run.py proposed_v3_ft
python run.py bnneck --protocol exp2_aerial_to_wearable.txt
```

### BNNeck baseline

This lane adapts the BNNeck idea to an ImageNet-pretrained ViT-S/16 at
`256x128`. It keeps the Bag-of-Tricks PK sampler, padding/crop/flip/random
erasing, label-smoothed identity CE, batch-hard triplet loss and normalized
post-BN retrieval feature. Its ViT initialization, AdamW optimizer and warm-up
cosine schedule match the TransReID-S lane, so the difference is architectural:
plain global ViT+BNNeck versus TransReID's SIE/optional JPM.

This is an adapted ViT-BNNeck baseline, not an exact reproduction of the 2019
BNNeck repository, whose official backbone is ResNet-50. The AG-ReID.v2 paper's
77.03% mAP is also a plain ViT result, not a published ViT-BNNeck checkpoint.

### PersonViT supervised fine-tune (Stage 1, no DBSCAN)

`python run.py personvit_ft` implements the downstream recipe stated in the
[PersonViT paper](https://arxiv.org/abs/2408.05398) and released in the
[official repository](https://github.com/hustvl/PersonViT): vanilla ViT-S/16,
final CLS-token aggregation, BNNeck, identity cross entropy, and batch-hard
triplet loss. It has no DBSCAN, SIE, JPM, temporal, attribute, or proposed loss.
Input/normalization are `256x128` and mean/std `0.5`; training uses P x K
sampling, SGD, a 20-epoch warm-up, and the paper's `4e-4 * batch_size / 64`
learning-rate scaling. The local default batch is 32 instead of the paper's 64.

The default deterministic PID budget is 25%. Its single checkpoint stores the
exact labeled PID list and is evaluated with normalized raw CLS features, which
matches the released `TEST.NECK_FEAT: before` setting. This is both a clean
supervised-25% baseline and the legal Stage-1 initialization for Proposed V2.
A 100%-label checkpoint must never initialize a 25%-label proposed run.

This is a controlled AG-ReID.v2 label-budget adaptation, not a claim that the
original paper used 25% downstream labels. The paper's 25% ablation refers to
the amount of unlabeled LUPerson pretraining data; its benchmark fine-tuning is
supervised. Here the downloaded LUPerson-pretrained teacher is kept intact and
only the AG downstream identity budget is reduced to 25%.

### TransReID-S baseline

The local profile uses ViT-Small, side-information embeddings (SIE), P×K
sampling, AMP, and gradient accumulation. An optional local-part branch inspired
by JPM is implemented but disabled by default for low-memory GPUs. Enable it
with `TransReIDConfig.transreid_jpm=True`.

The local ViT-Small result must be reported as an adapted TransReID-S baseline;
it is not an exact reproduction of published ViT-Base numbers.

### UntransReID SSL baseline

The SSL lane uses no target identity labels. It clusters bounded single-camera
track prototypes with DBSCAN, trains a cluster-level contrastive memory, and
adds UntransReID-style masked-patch consistency. Clustering approximately 2,000
track prototypes instead of all 51,000 image features keeps the run practical
on one local GPU. Training samples at most eight frames per single-camera track
per epoch (about 16,000 images before DBSCAN outlier filtering). Parent-folder
identity tokens are never parsed; each track key is only an opaque folder plus
a camera id.

For a paper-style initialization, download a ViT-S/16 PersonViT or
TransReID-SSL checkpoint and pass `--ssl-checkpoint`. Without one, the code
falls back to generic DINO self-supervised pretraining from timm and must be reported as an adapted
UntransReID-style baseline, not an exact reproduction. Query/gallery labels are
read only once for final mAP and are not used to select the checkpoint; the one
saved checkpoint is selected by a label-free clustering proxy.

For the official PersonViT ViT-S/16 folder, download only
`checkpoint0220.pth`. The authors' fine-tuning command selects epoch 220; files
0240 and 0260 are later snapshots of the same pretraining run. The loader reads
the teacher backbone and automatically uses PersonViT's 0.5/0.5 normalization.

### Proposed LeWM-ReID method

The proposed method adapts
[LeWorldModel (arXiv:2603.19312)](https://arxiv.org/abs/2603.19312) to temporal
person ReID. It is not a reproduction: AG-ReID.v2 contains video frames but no
robot actions, so the original action-conditioned predictor is replaced with an
action-free causal predictor over chronological frames from one camera
tracklet.

The two LeWorldModel ingredients retained are:

- next-embedding prediction without stop-gradient or EMA;
- SIGReg Gaussian latent regularization for anti-collapse.

Identity CE and triplet losses are applied only to windows selected by
`ProposedConfig.labeled_fraction`. Known attributes provide weak auxiliary
supervision; missing and unknown values are masked.

Run at least these label budgets:

```text
0.10, 0.25, 0.50, 1.00
```

The main proposed-method result is the mAP-versus-label-budget curve.

The current `proposed.py` remains unchanged, so its earlier experiments remain
comparable.

### Proposed V2: PersonViT + BNNeck + semi-supervision

`python run.py proposed_v2` is a separate implementation in
`reid_advance/pipelines/proposed_v2.py`. It first constructs the PersonViT
ViT-S/16 teacher from `pretrained/checkpoint0240.pth`, then loads the same-budget
Stage-1 checkpoint from `outputs/personvit_ft_25/best_model.pth`. It refuses a
checkpoint whose labeled PID list differs from the Proposed split, and combines:

- BNNeck retrieval features and labeled identity CE/triplet;
- DBSCAN pseudo-identities and ClusterNCE on the remaining identities;
- adjacent-frame temporal prediction within each single-camera track;
- masked coarse-attribute supervision;
- a lightweight camera-view component that is removed from the retrieval
  representation.

The default label budget is 25% of complete training identities. DBSCAN uses a
conservative `eps=0.20`: an empirical sweep on the downloaded checkpoint showed
that `0.35` collapsed roughly 2,000 tracks into only about two dozen clusters.
The model is re-clustered every five epochs to keep a local run feasible. Only
one `outputs/proposed_v2/best_model.pth` is retained, selected by a label-free
coverage-times-positive-silhouette score; official query/gallery mAP is run
only after training.

Run the stages in order:

```powershell
python run.py personvit_ft --labeled-fraction 0.25
python run.py proposed_v2 --labeled-fraction 0.25
python run.py proposed_v3
```

### Proposed V3: V2 refinement with EMA, JPM and k-reciprocal clustering

`python run.py proposed_v3` is a separate second-stage pipeline. By default it
loads `outputs/proposed_v2/best_model.pth`, then adds four horizontal local-patch
branches, uses an EMA teacher for clustering, and replaces raw-cosine DBSCAN
with a k-reciprocal Jaccard/cosine blended distance. Two labeled-only warm-up
epochs allow the new local heads to settle before pseudo-label training starts.
At every clustering step it sweeps multiple re-ranked radii and a known-safe
raw-cosine fallback, then accepts only the highest-scoring candidate whose
silhouette is at least `0.10`. The same guard is mandatory at epoch zero, so an
unsafe initial partition cannot poison the run.

On the checked V2 epoch-25 checkpoint, the automatic initial search selected
raw cosine at `eps=0.24`: 189 clusters, 30.74% coverage and 0.3957 silhouette.
This replaces the earlier unsafe fixed `eps=0.50` behavior, which produced
77.59% coverage with negative silhouette.

V3 keeps only `outputs/proposed_v3/best_model.pth`, independently of V2. It is
selected without query/gallery labels. The local machine completed V3 with the
default batch of 32 pairs, giving eight identities per P×K batch and a stronger
batch-hard triplet signal.

```powershell
python run.py proposed_v3
python run.py proposed_v3 --init-checkpoint outputs/proposed_v2/best_model.pth
python run.py proposed_v3 --dbscan-eps-start 0.50 --dbscan-eps-end 0.60
```

This is an experimental refinement, not a guarantee of 75--80% mAP. Its main
diagnostic is whether re-ranked clustering raises coverage while silhouette
remains positive.

### V3 low-LR fine-tuning and image geometry

`python run.py proposed_v3_ft` loads all 203 tensors from the completed V3
teacher checkpoint, writes to a separate output directory, and runs 15
additional epochs with backbone/head learning rates of `1e-6`/`1e-5`. It is a
low-LR second stage rather than an exact optimizer-state resume; the existing
V3 checkpoint is never overwritten.

The downloaded PersonViT checkpoint declares `image_size=[256,128]` and patch
size 16. Torchvision and timm both interpret the tuple as `(height, width)`, so
the model correctly receives `3x256x128` tensors and a `16x8` patch grid. A
1,000-image sample from each AG-ReID.v2 split had average width/height around
0.44--0.45, close to the target ratio 0.50.

The earlier generic SSL augmentation was the actual geometry problem: its
`RandomResizedCrop` sampled near-square ratios of 0.75--1.33 and could remove
large portions of a portrait person crop. V3 and V3-FT now use the standard
ReID sequence `Resize(256,128) -> Flip -> Pad(10) -> RandomCrop(256,128)`,
followed by color jitter, normalization and random erasing. V2 is unchanged so its
reported result remains reproducible.

## Local GPU defaults

- BNNeck: batch 32 (8 identities x 4 images) with ViT-S/16.
- TransReID: real batch 8, gradient accumulation 8, effective batch 64.
- Proposed: real batch 8 tracklets × 4 frames, gradient accumulation 4.
- Proposed V2: two real batches of 8 adjacent-frame pairs per round (labeled
  and pseudo), gradient accumulation 2.
- Proposed V3: batch 32 pairs, four local patch parts and a parameter-only EMA
  teacher. Reduce to 16 only if a future configuration runs out of memory.
- Reduce `batch_size` first if CUDA runs out of memory.
- SIGReg uses 256 projections locally; the LeWorldModel paper default is 1024.

## Evaluation

All methods use L2-normalized features, cosine distance, mAP, and CMC Rank-k.
Gallery samples with the same identity and camera as the query are removed as
junk. Keep the protocol file, normalization, and split seed identical across
methods.

### Compare every available checkpoint

The comparison suite evaluates rather than retrains. It always includes the
untouched PersonViT teacher checkpoint, discovers every
`outputs/**/best_model.pth`, reconstructs the corresponding source model, and
recomputes Rank-1/mAP under one protocol. Missing BNNeck/TransReID outputs are
simply absent until their checkpoints are copied into the standard folders.

```powershell
# Preview what will be evaluated.
python tools/compare_experiments.py --list

# Full Aerial-to-CCTV comparison with flip TTA for every model.
python tools/compare_experiments.py

# Only the raw, target-untuned PersonViT baseline.
python tools/compare_experiments.py --only personvit_base

# Add a teammate checkpoint stored elsewhere.
python tools/compare_experiments.py `
  --checkpoint teammate_transreid=path/to/best_model.pth
```

The latest comparison reports are written to:

```text
outputs/comparison/metrics.csv
outputs/comparison/metrics.md
```

The current auto-detection covers raw PersonViT, supervised PersonViT
fine-tuning, BNNeck, TransReID, UntransReID, LeWM V1, Proposed V2, Proposed
V3/V3-FT, and the archived DINOv2+VICReg checkpoint. All models use the same
protocol and flip-TTA setting, while retaining the input size and normalization
stored with their own checkpoint.

## Main references

- Luo et al., *A Strong Baseline and Batch Normalization Neck for Deep Person
  Re-identification*, 2019.
- He et al., *TransReID: Transformer-based Object Re-Identification*, ICCV 2021.
- Ye et al., *Transformer for Object Re-Identification: A Survey* (includes
  UntransReID), IJCV 2024.
- Dai et al., *Cluster Contrast for Unsupervised Person Re-Identification*,
  ACCV 2022.
- Maes et al., *LeWorldModel: Stable End-to-End Joint-Embedding Predictive
  Architecture from Pixels*, arXiv 2026.
