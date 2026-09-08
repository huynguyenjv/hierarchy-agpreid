"""Independent experiment profiles for all project lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .identity import FILENAME_PATTERN, IDENTITY_SCHEME


@dataclass
class BaseConfig:
    data_root: str = "AG-ReID.v2"
    train_dir: str = "train_all"
    query_dir: str = "query"
    gallery_dir: str = "gallery"
    filename_pattern: str = FILENAME_PATTERN
    identity_scheme: str = IDENTITY_SCHEME
    strict_dataset_integrity: bool = True
    expected_train_identities: int = 807
    # None selects the official expected count from eval_txt_file (534 for
    # aerial/CCTV and 519 for aerial/wearable protocols).
    expected_protocol_identities: int | None = None
    attr_file: str = "qut_attribute_v8.mat"
    eval_txt_file: str = "exp1_aerial_to_cctv.txt"

    image_size: Tuple[int, int] = (224, 224)
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    pretrained: bool = True
    ignore_unknown_attributes: bool = True

    batch_size: int = 8
    instances_per_identity: int = 4
    num_workers: int = 4
    epochs: int = 150
    weight_decay: float = 0.05
    grad_clip: float = 5.0
    warmup_epochs: int = 10
    validation_fraction: float = 0.1
    split_seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    checkpoint_name: str = "best_model.pth"
    eval_batch_size: int = 64
    flip_tta: bool = True


@dataclass
class BNNeckConfig(BaseConfig):
    """ViT-S/16 + BNNeck supervised baseline for local AG-ReID.v2 runs."""

    output_dir: str = "./outputs/bnneck"
    checkpoint_name: str = "best_model.pth"
    image_size: Tuple[int, int] = (256, 128)
    batch_size: int = 32
    instances_per_identity: int = 4
    epochs: int = 120
    encoder_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    embed_dim: int = 384
    learning_rate: float = 3e-5
    head_lr_multiplier: float = 10.0
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    flip_probability: float = 0.5
    padding: int = 10
    random_erasing_probability: float = 0.5
    flip_tta: bool = True
    warmup_epochs: int = 10
    eval_interval: int = 5


@dataclass
class PersonViTFineTuneConfig(BaseConfig):
    """Paper-faithful PersonViT-S/16 BOT fine-tune on a PID label budget."""

    output_dir: str = "./outputs/personvit_ft_25"
    checkpoint_name: str = "best_model.pth"
    image_size: Tuple[int, int] = (256, 128)
    norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    batch_size: int = 32
    instances_per_identity: int = 4
    num_workers: int = 2
    epochs: int = 120
    warmup_epochs: int = 20
    encoder_name: str = "vit_small_patch16_224.dino"
    embed_dim: int = 384
    pretrained: bool = False
    ssl_pretrained_path: str = "pretrained/checkpoint0240.pth"

    # The released PersonViT recipe uses SGD and scales 4e-4 from batch 64.
    reference_learning_rate: float = 4e-4
    reference_batch_size: int = 64
    momentum: float = 0.9
    weight_decay: float = 1e-4
    triplet_margin: float = 0.3
    label_smoothing: float = 0.0

    labeled_fraction: float = 0.25
    labeled_split_seed: int = 42
    flip_probability: float = 0.5
    padding: int = 10
    random_erasing_probability: float = 0.5
    flip_tta: bool = True


@dataclass
class TransReIDConfig(BaseConfig):
    """Member 2: local-friendly supervised TransReID-S baseline."""

    output_dir: str = "./outputs/transreid"
    checkpoint_name: str = "best_model.pth"
    batch_size: int = 32
    grad_accum_steps: int = 2
    image_size: Tuple[int, int] = (256, 128)
    encoder_name: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    embed_dim: int = 384
    learning_rate: float = 3e-5
    head_lr_multiplier: float = 10.0
    label_smoothing: float = 0.1
    flip_probability: float = 0.5
    padding: int = 10
    random_erasing_probability: float = 0.5
    transreid_sie: bool = True
    transreid_jpm: bool = True
    transreid_num_parts: int = 4
    eval_interval: int = 5
    # AMP was already active in this pipeline; the flag only exists so memory
    # profiling and ablations can turn it off without editing the trainer.
    use_amp: bool = True
    # Save every evaluated epoch so the camera-pair gap can be measured as a
    # function of model capability rather than at the single best checkpoint.
    snapshot_epochs: bool = False


@dataclass
class UntransReIDConfig(BaseConfig):
    """Zero-ID-label UntransReID-style baseline adapted for local training."""

    output_dir: str = "./outputs/untransreid"
    checkpoint_name: str = "best_model.pth"
    batch_size: int = 32
    grad_accum_steps: int = 2
    num_workers: int = 2
    epochs: int = 30
    warmup_epochs: int = 5
    image_size: Tuple[int, int] = (256, 128)
    encoder_name: str = "vit_small_patch16_224.dino"
    embed_dim: int = 384
    learning_rate: float = 3e-5

    # Supply PersonViT/TransReID-SSL weights here for the paper-style SSL
    # initialization. An empty path falls back to generic DINO SSL weights from
    # timm and is reported as an adapted, not exact, UntransReID baseline.
    ssl_pretrained_path: str = ""
    ssl_checkpoint_uses_half_normalization: bool = True
    cluster_batch_size: int = 64
    cluster_samples_per_track: int = 4
    train_samples_per_track: int = 8
    dbscan_eps: float = 0.35
    dbscan_min_samples: int = 2
    silhouette_sample_size: int = 2000
    cluster_temperature: float = 0.05
    cluster_momentum: float = 0.1
    mask_ratio: float = 0.30
    mask_temperature: float = 0.10
    mask_consistency_weight: float = 1.0


@dataclass
class ProposedConfig(BaseConfig):
    """Member 3: LeWM-inspired semi-supervised temporal ReID."""

    output_dir: str = "./outputs/proposed_lewm"
    checkpoint_name: str = "best_model.pth"
    batch_size: int = 8
    grad_accum_steps: int = 4
    num_workers: int = 2
    encoder_name: str = "vit_tiny_patch16_224"
    embed_dim: int = 192
    pretrained: bool = False
    learning_rate: float = 1e-4

    sequence_length: int = 4
    sequence_stride: int = 2
    labeled_fraction: float = 0.25
    labeled_split_seed: int = 42
    predictor_depth: int = 3
    predictor_heads: int = 3
    predictor_dropout: float = 0.1
    sigreg_num_projections: int = 256
    sigreg_knots: int = 17
    sigreg_weight: float = 0.1
    temporal_prediction_weight: float = 1.0
    supervised_ce_weight: float = 1.0
    supervised_triplet_weight: float = 1.0
    attribute_weight: float = 0.1


@dataclass
class ProposedV2Config(BaseConfig):
    """Cluster-enhanced semi-supervised aerial-ground ReID proposal."""

    output_dir: str = "./outputs/proposed_v2"
    checkpoint_name: str = "best_model.pth"
    image_size: Tuple[int, int] = (256, 128)
    norm_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    norm_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    batch_size: int = 8
    instances_per_identity: int = 4
    grad_accum_steps: int = 2
    num_workers: int = 2
    epochs: int = 30
    warmup_epochs: int = 3

    encoder_name: str = "vit_small_patch16_224.dino"
    embed_dim: int = 384
    pretrained: bool = False
    ssl_pretrained_path: str = "pretrained/checkpoint0240.pth"
    # Stage 1 is the same-label-budget PersonViT BOT fine-tune.  Loading it is
    # safe only when its labeled PID set exactly matches this run.
    initialization_checkpoint: str = "./outputs/personvit_ft_25/best_model.pth"
    require_initialization_checkpoint: bool = True
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.05

    # The budget is a fraction of identities whose complete training PID is
    # visible. All remaining identities are treated as unlabeled tracks.
    labeled_fraction: float = 0.25
    labeled_split_seed: int = 42
    train_pairs_per_track: int = 4
    temporal_frame_gap: int = 1

    cluster_batch_size: int = 64
    cluster_samples_per_track: int = 2
    # A conservative radius avoids the severe identity over-merging observed
    # with PersonViT features on AG-ReID.v2 around eps=0.35.
    dbscan_eps: float = 0.20
    dbscan_min_samples: int = 2
    silhouette_sample_size: int = 2000
    recluster_interval: int = 5
    cluster_temperature: float = 0.05
    cluster_momentum: float = 0.1

    label_smoothing: float = 0.1
    supervised_ce_weight: float = 1.0
    supervised_triplet_weight: float = 1.0
    cluster_weight: float = 1.0
    temporal_weight: float = 0.2
    attribute_weight: float = 0.1
    view_classification_weight: float = 0.1
    view_orthogonality_weight: float = 0.05
    temporal_stop_gradient: bool = True
    reliable_attributes: Tuple[str, ...] = (
        "gender",
        "height",
        "weight",
        "upper",
        "lower",
        "bag",
    )


@dataclass
class ProposedV3Config(ProposedV2Config):
    """Second-stage V2 refinement with EMA, local features and re-ranking."""

    output_dir: str = "./outputs/proposed_v3"
    checkpoint_name: str = "best_model.pth"
    # The local machine completed V3 at 32, preserving eight identities per
    # P x K batch for a substantially stronger batch-hard triplet signal.
    batch_size: int = 32
    epochs: int = 20
    warmup_epochs: int = 1
    grad_accum_steps: int = 2

    # V3 is intentionally a second stage. It reuses the best label-free V2
    # checkpoint instead of discarding the already learned aerial-ground space.
    initialization_checkpoint: str = "./outputs/proposed_v2/best_model.pth"
    require_initialization_checkpoint: bool = True

    pseudo_warmup_epochs: int = 2
    ema_decay: float = 0.999
    recluster_interval: int = 4

    jpm_parts: int = 4
    # Keep the already strong V2 global embedding dominant while the new local
    # branches are still adapting.
    local_feature_weight: float = 0.25
    local_identity_loss_weight: float = 0.25
    local_view_subtraction: float = 0.25

    # DBSCAN operates on a k-reciprocal Jaccard/cosine blended distance in V3,
    # so its epsilon is not comparable to V2's raw-cosine epsilon=0.20.
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.30
    dbscan_eps_start: float = 0.50
    dbscan_eps_end: float = 0.60
    minimum_cluster_silhouette: float = 0.10
    rerank_eps_search_min: float = 0.20
    rerank_eps_search_step: float = 0.025
    cosine_eps_search_min: float = 0.14
    cosine_eps_search_max: float = 0.28
    cosine_eps_search_step: float = 0.02

    # AG-ReID crops average W/H ~= 0.45.  The generic near-square
    # RandomResizedCrop used by the SSL lane can remove much of a person's
    # body, so V3 uses the standard ReID resize/pad/crop recipe instead.
    preserve_person_aspect_augmentation: bool = True
    augmentation_padding: int = 10
    random_erasing_probability: float = 0.5


@dataclass
class ProposedV3FineTuneConfig(ProposedV3Config):
    """Low-learning-rate continuation from the best completed V3 teacher."""

    output_dir: str = "./outputs/proposed_v3_finetune"
    initialization_checkpoint: str = "./outputs/proposed_v3/best_model.pth"
    epochs: int = 15
    warmup_epochs: int = 1
    pseudo_warmup_epochs: int = 0
    backbone_learning_rate: float = 1e-6
    head_learning_rate: float = 1e-5
    ema_decay: float = 0.9995


@dataclass
class CargoConfig(TransReIDConfig):
    """CARGO trained with the exact AG-ReID.v2 recipe, for a comparable matrix.

    The point of running CARGO at all is to tell whether the flat camera-pair
    matrix on AG-ReID.v2 reflects that dataset or aerial-ground ReID after
    ordinary supervision. That comparison only holds if the two are trained the
    same way, so everything inherited from TransReIDConfig is left alone; only
    what CARGO structurally requires is overridden.
    """

    data_root: str = "D:/datasets/cargo"
    train_dir: str = "train"
    output_dir: str = "./outputs/cargo"
    # CARGO ships 13 real cameras (1-5 aerial, 6-13 ground) rather than
    # AG-ReID.v2's three, and its ids are already dense, so SIE indexes them
    # directly instead of going through camera_to_view.
    sie_num_views: int = 14
    sie_identity_map: bool = True
    # 2,500 training identities against AG-ReID.v2's 807.
    expected_train_identities: int = 2500
    strict_dataset_integrity: bool = False
    # 30 epochs is the ceiling of this recipe on CARGO, not under-training:
    # mAP went 42.1 -> 45.5 -> 46.9 -> 46.6 across epochs 10/20/25/30 while the
    # train loss kept falling, which is overfitting, not room to grow.
    epochs: int = 30
    # Evaluate every 5 epochs so the camera-pair matrix can be recomputed at
    # several points; convergence is judged by the matrix settling, not by mAP.
    eval_interval: int = 5
    # Pick up from outputs/cargo/last.pth if a previous run was interrupted.
    resume: bool = True
