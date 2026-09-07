"""Proposed V3: second-stage high-coverage semi-supervised ReID refinement.

V3 is deliberately separate from ``proposed_v2.py`` and never overwrites its
checkpoint.  It starts from the best V2 model, adds JPM-style horizontal patch
features, clusters an EMA teacher with a k-reciprocal Jaccard distance, and
progressively expands the DBSCAN radius after a short labeled-only warm-up.

The official query/gallery labels remain final-evaluation-only.  Exactly one
V3 checkpoint is retained and selected with the same conservative label-free
coverage/separation principle as V2.
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..attributes import ATTRIBUTE_CLASS_COUNTS, ATTRIBUTE_IGNORE_INDEX, ATTRIBUTE_NAMES
from ..config import ProposedV3Config
from ..data import TrackPrototypeDataset, UnlabeledReIDDataset, load_track_attributes
from ..losses import TripletLoss
from ..identity import IDENTITY_SCHEME, require_current_identity_scheme
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)
from .proposed_v2 import (
    TrackPairDataset,
    build_pk_loader,
    select_labeled_identities,
)
from .untransreid import cluster_nce, load_ssl_backbone, update_cluster_memory


class ProposedV3Model(nn.Module):
    """PersonViT global identity plus horizontal local-patch identity cues."""

    def __init__(self, cfg: ProposedV3Config, num_labeled_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=False,
            num_classes=0,
            img_size=cfg.image_size,
        )
        dim = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        if dim != cfg.embed_dim:
            raise ValueError(f"Expected embedding dimension {cfg.embed_dim}, got {dim}")
        if not cfg.ssl_pretrained_path:
            raise ValueError("Proposed V3 requires a PersonViT SSL checkpoint")
        load_ssl_backbone(self.encoder, cfg.ssl_pretrained_path)

        self.view_norm = nn.LayerNorm(dim)
        self.view_adapter = nn.Linear(dim, dim, bias=False)
        self.bnneck = nn.BatchNorm1d(dim)
        self.bnneck.bias.requires_grad_(False)
        self.identity_classifier = nn.Linear(dim, num_labeled_classes, bias=False)
        self.view_classifier = nn.Linear(dim, 3)
        self.temporal_predictor = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.attribute_heads = nn.ModuleDict(
            {
                name: nn.Linear(dim, ATTRIBUTE_CLASS_COUNTS[name])
                for name in cfg.reliable_attributes
            }
        )
        self.local_bnnecks = nn.ModuleList(
            [nn.BatchNorm1d(dim) for _ in range(cfg.jpm_parts)]
        )
        self.local_classifiers = nn.ModuleList(
            [
                nn.Linear(dim, num_labeled_classes, bias=False)
                for _ in range(cfg.jpm_parts)
            ]
        )
        self.retrieval_dim = dim
        self._initialize_heads()

    def _initialize_heads(self):
        nn.init.normal_(self.view_adapter.weight, std=0.001)
        nn.init.normal_(self.bnneck.weight, 1.0, 0.02)
        nn.init.zeros_(self.bnneck.bias)
        nn.init.normal_(self.identity_classifier.weight, std=0.001)
        nn.init.normal_(self.view_classifier.weight, std=0.001)
        nn.init.zeros_(self.view_classifier.bias)
        for head in self.attribute_heads.values():
            nn.init.normal_(head.weight, std=0.001)
            nn.init.zeros_(head.bias)
        for neck, classifier in zip(self.local_bnnecks, self.local_classifiers):
            nn.init.normal_(neck.weight, 1.0, 0.02)
            nn.init.zeros_(neck.bias)
            neck.bias.requires_grad_(False)
            nn.init.normal_(classifier.weight, std=0.001)

    def _global_and_local_tokens(self, images):
        tokens = self.encoder.forward_features(images)
        if tokens.ndim != 3:
            raise RuntimeError(
                f"V3 expects ViT token features [B,N,D], got {tuple(tokens.shape)}"
            )
        prefix_tokens = int(getattr(self.encoder, "num_prefix_tokens", 1))
        global_token = tokens[:, 0]
        patches = tokens[:, prefix_tokens:]
        grid_height, grid_width = self.encoder.patch_embed.grid_size
        expected = int(grid_height * grid_width)
        if patches.size(1) != expected:
            raise RuntimeError(
                f"Expected {expected} patch tokens at grid "
                f"{grid_height}x{grid_width}, got {patches.size(1)}"
            )
        patch_grid = patches.view(
            patches.size(0), grid_height, grid_width, patches.size(-1)
        )
        stripes = torch.tensor_split(patch_grid, self.cfg.jpm_parts, dim=1)
        if any(stripe.size(1) == 0 for stripe in stripes):
            raise ValueError("jpm_parts exceeds the ViT patch-grid height")
        local_tokens = [stripe.mean(dim=(1, 2)) for stripe in stripes]
        return global_token, local_tokens

    def encode_components(self, images):
        global_token, local_tokens = self._global_and_local_tokens(images)
        view_feature = self.view_adapter(self.view_norm(global_token))
        identity_feature = global_token - view_feature
        neck_feature = self.bnneck(identity_feature)
        global_retrieval = F.normalize(neck_feature, dim=1)

        local_identities = [
            token - self.cfg.local_view_subtraction * view_feature
            for token in local_tokens
        ]
        local_necks = [
            neck(feature)
            for neck, feature in zip(self.local_bnnecks, local_identities)
        ]
        local_retrievals = [F.normalize(feature, dim=1) for feature in local_necks]
        local_consensus = torch.stack(local_retrievals, dim=0).mean(dim=0)
        retrieval = F.normalize(
            global_retrieval + self.cfg.local_feature_weight * local_consensus,
            dim=1,
        )
        return {
            "identity": identity_feature,
            "view": view_feature,
            "neck": neck_feature,
            "retrieval": retrieval,
            "local_identities": local_identities,
            "local_necks": local_necks,
        }

    def forward(self, images):
        """Combined global/local retrieval path for clustering and evaluation."""
        return self.encode_components(images)["retrieval"]

    def forward_pairs(self, frames):
        batch, time = frames.shape[:2]
        components = self.encode_components(frames.flatten(0, 1))
        neck = components["neck"]
        identity_logits = self.identity_classifier(neck)
        local_logits = [
            classifier(feature)
            for classifier, feature in zip(
                self.local_classifiers, components["local_necks"]
            )
        ]
        return {
            "identity": components["identity"].view(batch, time, -1),
            "view": components["view"].view(batch, time, -1),
            "retrieval": components["retrieval"].view(batch, time, -1),
            "identity_logits": identity_logits.view(batch, time, -1),
            "local_identity_logits": [
                logits.view(batch, time, -1) for logits in local_logits
            ],
            "view_logits": self.view_classifier(components["view"]).view(
                batch, time, -1
            ),
            "attribute_logits": {
                name: head(neck) for name, head in self.attribute_heads.items()
            },
        }

    def optimizer_groups(self):
        heads = [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("encoder.")
        ]
        return [
            {"params": self.encoder.parameters(), "lr": self.cfg.backbone_learning_rate},
            {"params": heads, "lr": self.cfg.head_learning_rate},
        ]


def load_v2_initialization(model, cfg: ProposedV3Config, labeled_pid_to_class):
    path = cfg.initialization_checkpoint
    if not path or not os.path.isfile(path):
        if cfg.require_initialization_checkpoint:
            raise FileNotFoundError(
                f"V3 initialization checkpoint not found: {path}. "
                "Run proposed_v2 first or pass --init-checkpoint."
            )
        print("V3 warning: V2 checkpoint unavailable; using PersonViT only")
        return 0
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_current_identity_scheme(payload, path)
    checkpoint_pids = payload.get("labeled_pids")
    expected_pids = sorted(labeled_pid_to_class)
    if checkpoint_pids is None or list(checkpoint_pids) != expected_pids:
        raise RuntimeError(
            "V3 initialization and the current run use different labeled P+T+A "
            "identity sets. Rerun the preceding stage with the same label budget."
        )
    source = payload.get("model", payload)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    if not any(key.startswith("encoder.") for key in compatible):
        raise RuntimeError(f"No V2 encoder tensors were compatible with {path}")
    model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded stage initialization: {len(compatible)}/{len(target)} tensors "
        f"from epoch {payload.get('epoch', '?')}"
    )
    return len(compatible)


@torch.no_grad()
def update_ema_teacher(teacher, student, decay):
    teacher_parameters = dict(teacher.named_parameters())
    for name, parameter in student.named_parameters():
        teacher_parameters[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, buffer in student.named_buffers():
        teacher_buffers[name].copy_(buffer.detach())


@torch.no_grad()
def extract_track_prototypes_v3(model, dataset, cfg: ProposedV3Config):
    prototype_dataset = TrackPrototypeDataset(dataset, cfg.cluster_samples_per_track)
    loader = DataLoader(
        prototype_dataset,
        batch_size=cfg.cluster_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    model.eval()
    sums = torch.zeros(dataset.num_tracks, model.retrieval_dim, device=cfg.device)
    counts = torch.zeros(dataset.num_tracks, 1, device=cfg.device)
    use_amp = str(cfg.device).startswith("cuda")
    for images, track_indices in tqdm(loader, desc="V3 teacher features", leave=False):
        images = images.to(cfg.device, non_blocking=True)
        track_indices = track_indices.to(cfg.device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            features = model(images)
        sums.index_add_(0, track_indices, features.float())
        counts.index_add_(
            0,
            track_indices,
            torch.ones(track_indices.size(0), 1, device=cfg.device),
        )
    if torch.any(counts == 0):
        raise RuntimeError("At least one track received no V3 prototype samples")
    return F.normalize(sums / counts, dim=1).cpu()


def k_reciprocal_distance(features, k1=20, k2=6, blend=0.30):
    """Compute the standard k-reciprocal Jaccard/cosine blended distance."""
    values = F.normalize(features.float(), dim=1).cpu().numpy().astype(np.float32)
    sample_count = values.shape[0]
    if sample_count < 3:
        raise ValueError("k-reciprocal clustering requires at least three samples")
    k1 = min(max(1, int(k1)), sample_count - 1)
    k2 = min(max(1, int(k2)), sample_count)
    cosine = np.clip(1.0 - values @ values.T, 0.0, 2.0).astype(np.float32)

    rank_width = min(sample_count, max(k1 + 1, k2))
    partial = np.argpartition(cosine, rank_width - 1, axis=1)[:, :rank_width]
    rows = np.arange(sample_count)[:, None]
    order = np.argsort(cosine[rows, partial], axis=1)
    initial_rank = partial[rows, order]
    reciprocal_features = np.zeros_like(cosine, dtype=np.float32)

    half_k = max(1, int(round(k1 / 2)))
    for index in range(sample_count):
        forward = initial_rank[index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.any(backward == index, axis=1)]
        expansion = reciprocal.copy()
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : half_k + 1]
            candidate_backward = initial_rank[candidate_forward, : half_k + 1]
            candidate_reciprocal = candidate_forward[
                np.any(candidate_backward == candidate, axis=1)
            ]
            overlap = np.intersect1d(
                candidate_reciprocal, reciprocal, assume_unique=False
            )
            if len(overlap) > (2.0 / 3.0) * len(candidate_reciprocal):
                expansion = np.append(expansion, candidate_reciprocal)
        expansion = np.unique(expansion)
        weights = np.exp(-cosine[index, expansion])
        reciprocal_features[index, expansion] = weights / max(weights.sum(), 1e-12)

    if k2 > 1:
        query_expansion = np.zeros_like(reciprocal_features)
        for index in range(sample_count):
            query_expansion[index] = reciprocal_features[
                initial_rank[index, :k2]
            ].mean(axis=0)
        reciprocal_features = query_expansion

    inverted_index = [
        np.flatnonzero(reciprocal_features[:, index] > 0)
        for index in range(sample_count)
    ]
    jaccard = np.ones_like(cosine, dtype=np.float32)
    for index in range(sample_count):
        nonzero = np.flatnonzero(reciprocal_features[index] > 0)
        candidates = np.unique(
            np.concatenate([inverted_index[item] for item in nonzero])
        )
        minimum_sum = np.zeros(sample_count, dtype=np.float32)
        for candidate in candidates:
            minimum_sum[candidate] = np.minimum(
                reciprocal_features[index, nonzero],
                reciprocal_features[candidate, nonzero],
            ).sum()
        jaccard[index] = 1.0 - minimum_sum / np.maximum(
            2.0 - minimum_sum, 1e-12
        )
    distance = (1.0 - blend) * jaccard + blend * cosine
    np.fill_diagonal(distance, 0.0)
    return np.maximum(distance, distance.T).astype(np.float32)


def _cluster_precomputed(prototypes, distance, cfg, epsilon, distance_mode):
    raw_labels = DBSCAN(
        eps=float(epsilon),
        min_samples=cfg.dbscan_min_samples,
        metric="precomputed",
        n_jobs=-1,
    ).fit_predict(distance)
    cluster_ids = sorted(set(raw_labels) - {-1})
    if len(cluster_ids) < 2:
        raise RuntimeError(
            f"V3 {distance_mode} DBSCAN produced {len(cluster_ids)} clusters "
            f"at eps={epsilon:.3f}"
        )
    remap = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    labels = np.asarray(
        [remap.get(int(label), -1) for label in raw_labels], dtype=np.int64
    )
    valid = labels >= 0
    coverage = float(valid.mean())
    valid_features = prototypes[valid].numpy()
    valid_labels = labels[valid]
    sample_size = min(cfg.silhouette_sample_size, len(valid_labels))
    silhouette = float(
        silhouette_score(
            valid_features,
            valid_labels,
            metric="cosine",
            sample_size=sample_size if sample_size < len(valid_labels) else None,
            random_state=cfg.split_seed,
        )
    )
    memory = torch.stack(
        [
            F.normalize(prototypes[torch.from_numpy(labels == index)].mean(dim=0), dim=0)
            for index in range(len(cluster_ids))
        ]
    )
    stats = {
        "clusters": len(cluster_ids),
        "coverage": coverage,
        "silhouette": silhouette,
        "epsilon": float(epsilon),
        "mean_cluster_size": float(valid.sum() / len(cluster_ids)),
        "distance_mode": distance_mode,
    }
    return labels, memory, stats


def _epsilon_grid(start, stop, step):
    if step <= 0:
        raise ValueError("DBSCAN epsilon search step must be positive")
    low, high = sorted((float(start), float(stop)))
    values = np.arange(low, high + step * 0.5, step, dtype=np.float64)
    values = np.append(values, float(stop))
    return sorted({round(float(value), 6) for value in values})


def _raw_cosine_distance(prototypes):
    values = F.normalize(prototypes.float(), dim=1).numpy().astype(np.float32)
    distance = np.clip(1.0 - values @ values.T, 0.0, 2.0).astype(np.float32)
    np.fill_diagonal(distance, 0.0)
    return np.maximum(distance, distance.T)


def select_safe_clustering_v3(prototypes, cfg: ProposedV3Config, requested_epsilon):
    """Sweep both re-ranked and raw-cosine DBSCAN before accepting labels.

    The original V3 accepted eps=0.50 before applying its silhouette guard,
    which produced 77.6% coverage but negative separation.  This selector makes
    the guard mandatory at epoch zero and at every later re-clustering.  Raw
    cosine is included as a known-safe fallback so V3 cannot be forced to train
    on a broken k-reciprocal partition.
    """

    reranked = k_reciprocal_distance(
        prototypes,
        k1=cfg.rerank_k1,
        k2=cfg.rerank_k2,
        blend=cfg.rerank_lambda,
    )
    cosine = _raw_cosine_distance(prototypes)
    searches = (
        (
            "k-reciprocal",
            reranked,
            _epsilon_grid(
                cfg.rerank_eps_search_min,
                requested_epsilon,
                cfg.rerank_eps_search_step,
            ),
        ),
        (
            "cosine-fallback",
            cosine,
            _epsilon_grid(
                cfg.cosine_eps_search_min,
                cfg.cosine_eps_search_max,
                cfg.cosine_eps_search_step,
            ),
        ),
    )
    best = None
    best_score = float("-inf")
    best_unsafe = None
    for distance_mode, distance, candidates in searches:
        for epsilon in candidates:
            try:
                result = _cluster_precomputed(
                    prototypes, distance, cfg, epsilon, distance_mode
                )
            except (RuntimeError, ValueError):
                continue
            stats = result[2]
            if (
                best_unsafe is None
                or stats["silhouette"] > best_unsafe["silhouette"]
            ):
                best_unsafe = stats
            if stats["silhouette"] < cfg.minimum_cluster_silhouette:
                continue
            score = (
                stats["coverage"] * stats["silhouette"]
                + 1e-6 * stats["clusters"]
            )
            if score > best_score:
                best = result
                best_score = score
    if best is None:
        details = "no valid DBSCAN candidate"
        if best_unsafe is not None:
            details = (
                f"best silhouette={best_unsafe['silhouette']:.4f} at "
                f"{best_unsafe['distance_mode']} eps={best_unsafe['epsilon']:.3f}"
            )
        raise RuntimeError(
            "V3 refused to create unsafe initial pseudo labels: " + details
        )
    stats = best[2]
    print(
        f"V3 cluster search selected {stats['distance_mode']} "
        f"eps={stats['epsilon']:.3f} | clusters={stats['clusters']} | "
        f"coverage={stats['coverage']:.2%} | "
        f"silhouette={stats['silhouette']:.4f}"
    )
    return best


def reliable_attribute_loss(logits, targets, cfg: ProposedV3Config):
    losses = []
    valid_count = 0
    for name in cfg.reliable_attributes:
        target = targets[:, ATTRIBUTE_NAMES.index(name)]
        valid = target != ATTRIBUTE_IGNORE_INDEX
        if torch.any(valid):
            losses.append(F.cross_entropy(logits[name][valid], target[valid]))
            valid_count += int(valid.sum().item())
    if not losses:
        return sum(value.sum() * 0.0 for value in logits.values()), 0
    return torch.stack(losses).mean(), valid_count


def branch_objective_v3(branch, model, triplet, batch, cfg, memory=None):
    frames, labels, attributes, views, track_indices = (
        value.to(cfg.device, non_blocking=True) for value in batch
    )
    outputs = model.forward_pairs(frames)
    batch_size, time = outputs["identity"].shape[:2]
    repeated_views = views.repeat_interleave(time)
    view_loss = F.cross_entropy(
        outputs["view_logits"].flatten(0, 1), repeated_views
    )
    identity_unit = F.normalize(outputs["identity"].float(), dim=-1)
    view_unit = F.normalize(outputs["view"].float(), dim=-1)
    orthogonality = (identity_unit * view_unit).sum(dim=-1).square().mean()
    prediction = model.temporal_predictor(outputs["identity"][:, 0])
    temporal_target = outputs["identity"][:, 1]
    if cfg.temporal_stop_gradient:
        temporal_target = temporal_target.detach()
    temporal = 1.0 - F.cosine_similarity(
        prediction.float(), temporal_target.float(), dim=1
    ).mean()
    repeated_attributes = attributes.repeat_interleave(time, dim=0)
    attribute_loss, valid_attributes = reliable_attribute_loss(
        outputs["attribute_logits"], repeated_attributes, cfg
    )
    total = (
        cfg.temporal_weight * temporal
        + cfg.attribute_weight * attribute_loss
        + cfg.view_classification_weight * view_loss
        + cfg.view_orthogonality_weight * orthogonality
    )

    repeated_labels = labels.repeat_interleave(time)
    zero = outputs["identity"].sum() * 0.0
    identity_ce = zero
    local_ce = zero
    metric_loss = zero
    cluster_loss = zero
    if branch == "labeled":
        identity_ce = F.cross_entropy(
            outputs["identity_logits"].flatten(0, 1),
            repeated_labels,
            label_smoothing=cfg.label_smoothing,
        )
        local_losses = [
            F.cross_entropy(
                logits.flatten(0, 1),
                repeated_labels,
                label_smoothing=cfg.label_smoothing,
            )
            for logits in outputs["local_identity_logits"]
        ]
        local_ce = torch.stack(local_losses).mean()
        if repeated_labels.unique().numel() > 1:
            metric_loss = triplet(
                outputs["retrieval"].flatten(0, 1), repeated_labels
            )
        total = (
            total
            + cfg.supervised_ce_weight * identity_ce
            + cfg.local_identity_loss_weight * local_ce
            + cfg.supervised_triplet_weight * metric_loss
        )
    elif branch == "pseudo":
        if memory is None:
            raise ValueError("The V3 pseudo branch requires cluster memory")
        cluster_loss = cluster_nce(
            outputs["retrieval"].flatten(0, 1),
            repeated_labels,
            memory,
            cfg.cluster_temperature,
        )
        total = total + cfg.cluster_weight * cluster_loss
    else:
        raise ValueError(f"Unknown V3 branch {branch}")

    parts = {
        "id_ce": identity_ce,
        "local_ce": local_ce,
        "triplet": metric_loss,
        "cluster": cluster_loss,
        "temporal": temporal,
        "attribute": attribute_loss,
        "view": view_loss,
        "orthogonality": orthogonality,
    }
    memory_payload = (
        outputs["retrieval"].flatten(0, 1).detach(),
        repeated_labels.detach(),
    )
    return total, parts, valid_attributes, memory_payload, track_indices


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _selection_score(stats):
    return (
        stats["coverage"] * max(stats["silhouette"], 0.0)
        + 1e-6 * stats["clusters"]
    )


def _progressive_epsilon(cfg: ProposedV3Config, epoch):
    progress = min(max(epoch / max(cfg.epochs, 1), 0.0), 1.0)
    return cfg.dbscan_eps_start + progress * (
        cfg.dbscan_eps_end - cfg.dbscan_eps_start
    )


def _save_best(
    checkpoint, model, cfg, epoch, stats, score, labeled_pids, total_pids
):
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "epoch": epoch,
            "cluster_stats": stats,
            "selection_proxy": score,
            "labeled_pids": sorted(labeled_pids),
            "identity_scheme": IDENTITY_SCHEME,
            "train_identity_count": total_pids,
            "method": "Proposed V3 EMA-JPM-k-reciprocal semi-supervised ReID",
        },
        checkpoint,
    )


def train_proposed_v3(cfg: ProposedV3Config):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    query_loader, gallery_loader = build_eval_loaders(cfg)

    source = UnlabeledReIDDataset(cfg)
    labeled_pid_to_class, total_pids = select_labeled_identities(source, cfg)
    attributes = load_track_attributes(
        os.path.join(cfg.data_root, cfg.attr_file),
        cfg.ignore_unknown_attributes,
    )
    labeled_dataset = TrackPairDataset(
        source,
        cfg,
        labeled_pid_to_class,
        branch="labeled",
        attribute_mapping=attributes,
    )
    model = ProposedV3Model(cfg, len(labeled_pid_to_class)).to(cfg.device)
    load_v2_initialization(model, cfg, labeled_pid_to_class)
    teacher = copy.deepcopy(model).to(cfg.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    triplet = TripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(), weight_decay=cfg.weight_decay
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, cfg.epochs, cfg.warmup_epochs, eta_min=1e-7
    )
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        f"Proposed V3 | init={cfg.initialization_checkpoint} | "
        f"labeled identities={len(labeled_pid_to_class)}/{total_pids} "
        f"({len(labeled_pid_to_class)/total_pids:.2%}) | "
        f"JPM parts={cfg.jpm_parts} | EMA={cfg.ema_decay}"
    )
    prototypes = extract_track_prototypes_v3(teacher, source, cfg)
    track_labels, memory, cluster_stats = select_safe_clustering_v3(
        prototypes, cfg, cfg.dbscan_eps_start
    )
    best_proxy = _selection_score(cluster_stats)
    best_epoch = 0
    _save_best(
        checkpoint,
        teacher,
        cfg,
        best_epoch,
        cluster_stats,
        best_proxy,
        labeled_pid_to_class,
        total_pids,
    )
    print(
        f"Initial V3 DBSCAN | clusters={cluster_stats['clusters']} | "
        f"coverage={cluster_stats['coverage']:.2%} | "
        f"silhouette={cluster_stats['silhouette']:.4f} | "
        f"eps={cluster_stats['epsilon']:.3f} | "
        f"distance={cluster_stats['distance_mode']}"
    )

    component_names = (
        "total",
        "id_ce",
        "local_ce",
        "triplet",
        "cluster",
        "temporal",
        "attribute",
        "view",
        "orthogonality",
    )
    for epoch in range(1, cfg.epochs + 1):
        pseudo_active = epoch > cfg.pseudo_warmup_epochs
        labeled_loader = build_pk_loader(labeled_dataset, cfg, epoch)
        pseudo_loader = None
        if pseudo_active:
            pseudo_dataset = TrackPairDataset(
                source,
                cfg,
                labeled_pid_to_class,
                branch="pseudo",
                track_labels=track_labels,
                attribute_mapping=attributes,
            )
            pseudo_loader = build_pk_loader(pseudo_dataset, cfg, epoch)
            steps = max(len(labeled_loader), len(pseudo_loader))
            pseudo_iterator = iter(pseudo_loader)
        else:
            steps = len(labeled_loader)
            pseudo_iterator = None
        labeled_iterator = iter(labeled_loader)
        memory = memory.to(cfg.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        valid_attribute_total = 0
        progress = tqdm(
            range(1, steps + 1),
            desc=f"Proposed-V3 {epoch:03d}/{cfg.epochs}",
            leave=False,
        )

        for step in progress:
            labeled_batch, labeled_iterator = _next_or_restart(
                labeled_iterator, labeled_loader
            )
            branches = [("labeled", labeled_batch)]
            if pseudo_active:
                pseudo_batch, pseudo_iterator = _next_or_restart(
                    pseudo_iterator, pseudo_loader
                )
                branches.append(("pseudo", pseudo_batch))

            round_total = 0.0
            for branch, current_batch in branches:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, parts, valid_attributes, memory_payload, _ = (
                        branch_objective_v3(
                            branch,
                            model,
                            triplet,
                            current_batch,
                            cfg,
                            memory=memory if branch == "pseudo" else None,
                        )
                    )
                    scaled_loss = loss / cfg.grad_accum_steps
                scaler.scale(scaled_loss).backward()
                round_total += loss.item()
                for name, value in parts.items():
                    totals[name] += value.item()
                valid_attribute_total += valid_attributes
                if branch == "pseudo":
                    features, pseudo_labels = memory_payload
                    update_cluster_memory(
                        memory, features, pseudo_labels, cfg.cluster_momentum
                    )

            totals["total"] += round_total
            if step % cfg.grad_accum_steps == 0 or step == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema_teacher(teacher, model, cfg.ema_decay)
            progress.set_postfix(
                loss=f"{totals['total']/step:.3f}",
                pseudo="on" if pseudo_active else "warmup",
                coverage=f"{cluster_stats['coverage']:.1%}",
            )

        scheduler.step()
        for name in component_names:
            writer.add_scalar(f"Train/{name}", totals[name] / steps, epoch)
        writer.add_scalar("Train/valid_attributes", valid_attribute_total / steps, epoch)
        writer.add_scalar("Train/backbone_lr", scheduler.get_last_lr()[0], epoch)
        print(
            f"Epoch {epoch:03d} | total {totals['total']/steps:.4f} | "
            f"CE {totals['id_ce']/steps:.4f} | "
            f"local {totals['local_ce']/steps:.4f} | "
            f"cluster {totals['cluster']/steps:.4f}"
        )

        should_recluster = (
            epoch > cfg.pseudo_warmup_epochs
            and (epoch % cfg.recluster_interval == 0 or epoch == cfg.epochs)
        )
        if should_recluster:
            epsilon = _progressive_epsilon(cfg, epoch)
            prototypes = extract_track_prototypes_v3(teacher, source, cfg)
            try:
                next_labels, next_memory, next_stats = select_safe_clustering_v3(
                    prototypes, cfg, epsilon
                )
            except (RuntimeError, ValueError) as error:
                print(
                    f"Epoch {epoch:03d} V3 clustering warning: {error}; "
                    "keeping the previous pseudo labels"
                )
                next_labels = track_labels
                next_memory = memory.detach().cpu()
                next_stats = cluster_stats

            score = _selection_score(next_stats)
            for name, value in next_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Clustering/{name}", value, epoch)
            writer.add_scalar("Clustering/selection_score", score, epoch)
            print(
                f"V3 recluster {epoch:03d} | clusters {next_stats['clusters']} | "
                f"coverage {next_stats['coverage']:.2%} | "
                f"silhouette {next_stats['silhouette']:.4f} | "
                f"distance {next_stats['distance_mode']} | "
                f"score {score:.5f}"
            )
            if score > best_proxy:
                best_proxy = score
                best_epoch = epoch
                _save_best(
                    checkpoint,
                    teacher,
                    cfg,
                    epoch,
                    next_stats,
                    score,
                    labeled_pid_to_class,
                    total_pids,
                )
            track_labels, memory, cluster_stats = (
                next_labels,
                next_memory,
                next_stats,
            )

    state = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
    require_current_identity_scheme(state, checkpoint)
    model.load_state_dict(state["model"])
    del teacher
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()
    rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
    print(
        f"Proposed V3 final (best proxy epoch {state['epoch']:03d}) | "
        f"Rank-1 {rank1:.2%} | mAP {mean_ap:.2%}"
    )
    writer.add_hparams(
        {
            "labeled_fraction": cfg.labeled_fraction,
            "rerank_k1": cfg.rerank_k1,
            "rerank_k2": cfg.rerank_k2,
            "ema_decay": cfg.ema_decay,
            "jpm_parts": cfg.jpm_parts,
        },
        {"final/rank1": rank1, "final/mAP": mean_ap},
    )
    writer.close()
    return checkpoint, rank1, mean_ap
