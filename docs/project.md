# Project experiment design

> Dataset integrity: all reported experiments parse an AG-ReID.v2 identity as
> the official composite `P+T+A` key (807 train identities), never `P` alone.
> Checkpoints without the `agreid_v2_p+t+a_v1` audit marker are legacy and are
> excluded from comparisons.

## Research question

How much identity-label annotation can be removed from aerial-ground person
ReID while retaining the retrieval quality of a fully supervised baseline?

## Dataset and protocol

- Dataset: AG-ReID.v2.
- Default evaluation: aerial query to CCTV gallery
  (`exp1_aerial_to_cctv.txt`).
- Metrics: mAP and CMC Rank-1/5/10.
- Query/gallery identity labels are used only for final evaluation.

## Independent baselines

1. `python run.py bnneck`: ImageNet-pretrained ViT-S/16 with BNNeck,
   label-smoothed identity CE, triplet loss and PK sampling.
2. `python run.py transreid`: local TransReID-S baseline with SIE, PK sampling,
   optional JPM, CE and triplet loss.

Both supervised baselines use all training identity labels and quantify the
upper-bound performance and annotation cost.

## Zero-label SSL baseline

`python run.py ssl` launches a local adaptation of UntransReID. It uses zero
AG-ReID.v2 training identity labels and combines DBSCAN pseudo-identities,
cluster-level contrastive memory, and masked-patch consistency. DBSCAN operates
on bounded single-camera track prototypes for local feasibility. The official
query/gallery identity labels are used only for the final retrieval metrics.

The checkpoint is selected by a label-free proxy combining DBSCAN coverage and
silhouette score, not by test mAP. With no `--ssl-checkpoint`, generic DINO SSL
pretraining from timm is used and the experiment must be named
"UntransReID-style".

## Proposed method

`python run.py proposed` launches a LeWorldModel-inspired, semi-supervised
temporal ReID adaptation. It does not claim to reproduce LeWorldModel because
AG-ReID.v2 does not contain action vectors or control trajectories.

For chronological frames from one camera tracklet:

\[
z_t = f_\theta(x_t), \qquad
\hat z_{t+1} = g_\phi(z_{\le t})
\]

The objective is:

\[
L = L_{pred} + \lambda_{sig}L_{SIGReg}
  + \lambda_{id}L_{ID}^{labeled}
  + \lambda_{tri}L_{triplet}^{labeled}
  + \lambda_{attr}L_{attribute}.
\]

- `L_pred`: next-embedding MSE without stop-gradient or EMA.
- `L_SIGReg`: random-projection Epps-Pulley regularization that encourages an
  isotropic Gaussian latent distribution and prevents collapse.
- Identity CE and triplet losses are applied only to the fraction selected by
  `ProposedConfig.labeled_fraction`.
- Known soft attributes remain an auxiliary weak-supervision signal; missing
  and unknown attributes are masked.

Run label budgets of 10%, 25%, 50%, and 100%. The primary result is the mAP vs.
label-budget curve, not a direct claim of fully supervised SOTA.

## Proposed V2: cluster-enhanced BNNeck method

Before V2, `python run.py personvit_ft` runs the paper-faithful PersonViT
downstream baseline on the same deterministic 25% PID budget. It is vanilla
ViT-S/16 CLS aggregation with BNNeck, CE, and triplet loss, using no DBSCAN or
proposal-specific component. The checkpoint stores the exact labeled PID list.
V2 verifies that list before loading the fine-tuned encoder/BNNeck/classifier;
this prevents a full-label supervised checkpoint from leaking information into
a 25%-label semi-supervised claim.

The 25% setting is this project's AG identity-label budget. It must not be
confused with the PersonViT paper's 25% ablation, which varies the amount of
unlabeled LUPerson pretraining data rather than downstream identity labels.

`python run.py proposed_v2` implements the cluster-enhanced design in a new
pipeline; it does not replace or modify the original LeWM experiment. Its
components are:

- PersonViT/TransReID-SSL: ReID-specific SSL initialization, reused rather than
  reproduced locally.
- labeled CE/triplet: identity supervision only for the selected label budget.
- DBSCAN + ClusterNCE: pseudo-identity supervision for otherwise unlabeled
  single-camera tracks.
- temporal consistency: adjacent frames in one track should retain identity
  despite pose and blur changes.
- attribute guidance: known coarse attributes are an auxiliary weak signal.

The implemented objective is

\[
L = L_{cluster}^{unlabeled}
  + \lambda_{id}L_{ID}^{labeled}
  + \lambda_{tri}L_{triplet}^{labeled}
  + \lambda_{temp}L_{temporal}
  + \lambda_{attr}L_{attribute}.
\]

It additionally learns a three-way aerial/wearable/CCTV view component and
penalizes its cosine overlap with the identity component. Retrieval uses the
L2-normalized post-BN identity feature only. The ViT-S/16 backbone is first
initialized from `pretrained/checkpoint0240.pth`, then the same-budget supervised
state is loaded from `outputs/personvit_ft_25/best_model.pth`; new
temporal/view/attribute heads are trained from scratch. The default split
exposes complete labels for 25% of training identities and treats every other
training identity as unlabeled.

Checkpoint selection never uses official query/gallery identities. A
conservative label-free score favours pseudo-label coverage only when DBSCAN
clusters have positive silhouette separation. Final AG-ReID.v2 mAP and Rank-1
are evaluated once from the single saved best checkpoint.

## Proposed V3: high-coverage second stage

`python run.py proposed_v3` preserves V2 as an independent result and refines
its best checkpoint. It targets V2's measured bottleneck--only about 18% of
tracks received DBSCAN supervision--with:

- four JPM-style horizontal patch features combined with the global feature;
- an EMA teacher used for deterministic track prototypes;
- k-reciprocal Jaccard/cosine distance before DBSCAN;
- a short labeled-only local-head warm-up;
- automatic re-ranked/raw-cosine epsilon search with a silhouette safety floor
  applied from epoch zero onward.

The V3 objective adds local identity classification to V2:

\[
L_{V3} = L_{V2}
  + \lambda_{local}L_{ID}^{local}.
\]

V3 is initialized from V2 but has its own output directory and single best
checkpoint. Query/gallery labels remain inaccessible until the final official
evaluation, so any gain cannot come from selecting epochs by test mAP.

### V3 continuation and input geometry

`python run.py proposed_v3_ft` performs a separate 15-epoch low-LR continuation
from `outputs/proposed_v3/best_model.pth`. It uses `1e-6` for the backbone and
`1e-5` for task/local heads, and saves only to
`outputs/proposed_v3_finetune/best_model.pth`.

PersonViT pretraining and the local model both use `(height,width)=(256,128)`
with a `16x8` ViT patch grid. V3 replaces the unsuitable near-square random
resized crop with resize/pad/random-crop augmentation, preserving the complete
portrait body while retaining small translation and occlusion perturbations.

## Unified checkpoint comparison

`python tools/compare_experiments.py` evaluates the untouched PersonViT
backbone and all discovered experiment checkpoints under the same official
protocol. It does not reuse mAP stored in supervised checkpoints and does not
train any model. The generated CSV/Markdown table is the source for the final
raw-SSL, supervised, proposed and fine-tuned comparison reported in the paper.
