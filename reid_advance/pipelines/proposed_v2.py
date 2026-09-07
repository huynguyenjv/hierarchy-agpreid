"""Proposed V2: cluster-enhanced semi-supervised aerial-ground ReID.

This lane is intentionally independent from ``proposed.py``. It combines a
PersonViT-pretrained ViT-S/16 encoder and BNNeck with five complementary
signals: exact PID supervision for a fixed identity-label budget, DBSCAN and
ClusterNCE on the remaining identities, adjacent-frame prediction, reliable
coarse attributes, and lightweight view disentanglement.

The official query/gallery identities are touched only once, after training.
The single checkpoint is selected by a conservative label-free clustering
proxy, not by test mAP.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..attributes import (
    ATTRIBUTE_CLASS_COUNTS,
    ATTRIBUTE_IGNORE_INDEX,
    ATTRIBUTE_NAMES,
)
from ..config import ProposedV2Config
from ..data import (
    RandomIdentitySampler,
    UnlabeledReIDDataset,
    load_track_attributes,
    parse_camera_id,
    select_identity_budget,
    track_key,
)
from ..losses import TripletLoss
from ..identity import (
    IDENTITY_SCHEME,
    parse_identity_id,
    require_current_identity_scheme,
    validate_identity_count,
)
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)
from .untransreid import (
    cluster_nce,
    cluster_tracks,
    extract_track_prototypes,
    load_ssl_backbone,
    update_cluster_memory,
)


def camera_to_view(camera_id: int) -> int:
    """Map AG cameras C0/C2/C3 to compact aerial/wearable/CCTV labels."""
    mapping = {0: 0, 2: 1, 3: 2}
    if camera_id not in mapping:
        raise ValueError(f"Unsupported AG-ReID.v2 camera C{camera_id}")
    return mapping[camera_id]


def _frame_number(path: str) -> int:
    match = re.search(r"F(?P<frame>\d+)", os.path.basename(path))
    return int(match.group("frame")) if match else 0


def select_labeled_identities(source: UnlabeledReIDDataset, cfg: ProposedV2Config):
    """Select complete identities for the simulated annotation budget."""
    selected, total = select_identity_budget(
        (path for path, _ in source.items),
        cfg.filename_pattern,
        cfg.labeled_fraction,
        cfg.labeled_split_seed,
        minimum=cfg.batch_size // cfg.instances_per_identity,
    )
    if cfg.strict_dataset_integrity:
        validate_identity_count(
            total, cfg.expected_train_identities, "AG-ReID.v2 label budget"
        )
    return selected, total


class TrackPairDataset(Dataset):
    """Bounded adjacent-frame pairs for either the labeled or pseudo branch."""

    def __init__(
        self,
        source: UnlabeledReIDDataset,
        cfg: ProposedV2Config,
        labeled_pid_to_class,
        branch: str,
        track_labels=None,
        attribute_mapping=None,
    ):
        if branch not in ("labeled", "pseudo"):
            raise ValueError("branch must be 'labeled' or 'pseudo'")
        self.source = source
        self.branch = branch
        self.labels = []
        self.items = []
        attributes = attribute_mapping
        if attributes is None:
            attributes = load_track_attributes(
                os.path.join(cfg.data_root, cfg.attr_file),
                cfg.ignore_unknown_attributes,
            )
        default_attr = torch.full(
            (len(ATTRIBUTE_NAMES),), ATTRIBUTE_IGNORE_INDEX, dtype=torch.long
        )

        for track_index in range(source.num_tracks):
            source_indices = sorted(
                source.indices_by_track[track_index],
                key=lambda index: _frame_number(source.items[index][0]),
            )
            if len(source_indices) <= cfg.temporal_frame_gap:
                continue
            first_path = source.items[source_indices[0]][0]
            pid = parse_identity_id(first_path, cfg.filename_pattern)
            is_labeled = pid in labeled_pid_to_class
            if branch == "labeled":
                if not is_labeled:
                    continue
                label = labeled_pid_to_class[pid]
            else:
                if is_labeled or track_labels is None:
                    continue
                label = int(track_labels[track_index])
                if label < 0:
                    continue

            possible = len(source_indices) - cfg.temporal_frame_gap
            pair_count = min(cfg.train_pairs_per_track, possible)
            starts = np.linspace(0, possible - 1, pair_count).round().astype(int)
            attr = attributes.get(track_key(first_path), default_attr)
            view = camera_to_view(parse_camera_id(first_path))
            for start in sorted(set(starts.tolist())):
                item = (
                    source_indices[start],
                    source_indices[start + cfg.temporal_frame_gap],
                    label,
                    attr,
                    view,
                    track_index,
                )
                self.items.append(item)
                self.labels.append(label)

        if not self.items:
            raise RuntimeError(f"The proposed-v2 {branch} branch has no usable pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        first, second, label, attributes, view, track_index = self.items[index]
        frames = torch.stack(
            [self.source.load_train_image(first), self.source.load_train_image(second)]
        )
        return frames, label, attributes, view, track_index


def build_pk_loader(dataset, cfg: ProposedV2Config, epoch: int):
    sampler = RandomIdentitySampler(
        dataset.labels,
        cfg.batch_size,
        cfg.instances_per_identity,
        seed=cfg.split_seed,
    )
    if len(set(dataset.labels)) < sampler.identities_per_batch:
        raise RuntimeError(
            f"Only {len(set(dataset.labels))} labels in {dataset.branch} branch; "
            "lower batch_size or adjust the label/DBSCAN configuration"
        )
    sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
    )


class ProposedV2Model(nn.Module):
    """PersonViT, residual view separation, BNNeck and auxiliary heads."""

    def __init__(self, cfg: ProposedV2Config, num_labeled_classes: int):
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
            raise ValueError("Proposed V2 requires a PersonViT SSL checkpoint")
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

    def encode_components(self, images):
        base = self.encoder(images)
        view_feature = self.view_adapter(self.view_norm(base))
        identity_feature = base - view_feature
        neck_feature = self.bnneck(identity_feature)
        retrieval = F.normalize(neck_feature, dim=1)
        return identity_feature, view_feature, neck_feature, retrieval

    def forward(self, images):
        """Retrieval path used by clustering and the official evaluator."""
        return self.encode_components(images)[-1]

    def forward_pairs(self, frames):
        batch, time = frames.shape[:2]
        identity, view, neck, retrieval = self.encode_components(
            frames.flatten(0, 1)
        )
        identity_logits = self.identity_classifier(neck)
        view_logits = self.view_classifier(view)
        attribute_logits = {
            name: head(neck) for name, head in self.attribute_heads.items()
        }
        return {
            "identity": identity.view(batch, time, -1),
            "view": view.view(batch, time, -1),
            "retrieval": retrieval.view(batch, time, -1),
            "identity_logits": identity_logits.view(batch, time, -1),
            "view_logits": view_logits.view(batch, time, -1),
            "attribute_logits": attribute_logits,
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


def load_personvit_finetune_initialization(
    model: ProposedV2Model, cfg: ProposedV2Config, labeled_pid_to_class
):
    """Warm-start V2 from the same-budget supervised PersonViT stage.

    The labeled PID audit is mandatory: accepting a checkpoint tuned with more
    identities would silently leak labels into the semi-supervised experiment.
    """
    path = cfg.initialization_checkpoint
    if not path or not os.path.isfile(path):
        if cfg.require_initialization_checkpoint:
            raise FileNotFoundError(
                f"PersonViT fine-tune checkpoint not found: {path}. "
                "Run `python run.py personvit_ft` first or pass --init-checkpoint."
            )
        print("Proposed V2 warning: using the raw PersonViT teacher only")
        return 0

    payload = torch.load(path, map_location="cpu", weights_only=False)
    require_current_identity_scheme(payload, path)
    expected_pids = sorted(labeled_pid_to_class)
    checkpoint_pids = payload.get("labeled_pids")
    if checkpoint_pids is None:
        raise RuntimeError(
            f"{path} has no labeled_pids audit metadata; refusing possible label leakage"
        )
    if list(checkpoint_pids) != expected_pids:
        raise RuntimeError(
            "PersonViT fine-tune and Proposed V2 use different labeled PID sets. "
            "Use the same --labeled-fraction and labeled_split_seed."
        )

    source = payload.get("model", payload)
    target = model.state_dict()
    permitted = ("encoder.", "bnneck.", "identity_classifier.")
    compatible = {
        key: value
        for key, value in source.items()
        if key.startswith(permitted)
        and key in target
        and target[key].shape == value.shape
    }
    encoder_tensors = sum(key.startswith("encoder.") for key in compatible)
    if encoder_tensors < len(model.encoder.state_dict()) // 2:
        raise RuntimeError(
            f"Too few PersonViT encoder tensors were compatible with {path}"
        )
    model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded same-budget PersonViT fine-tune: {len(compatible)} tensors "
        f"({encoder_tensors} encoder) from epoch {payload.get('epoch', '?')}"
    )
    return len(compatible)


def reliable_attribute_loss(logits, targets, cfg: ProposedV2Config):
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


def common_pair_losses(model, outputs, attributes, views, cfg: ProposedV2Config):
    batch, time = outputs["identity"].shape[:2]
    repeated_views = views.repeat_interleave(time)
    view_loss = F.cross_entropy(
        outputs["view_logits"].flatten(0, 1), repeated_views
    )
    identity_unit = F.normalize(outputs["identity"].float(), dim=-1)
    view_unit = F.normalize(outputs["view"].float(), dim=-1)
    orthogonality = (identity_unit * view_unit).sum(dim=-1).square().mean()

    prediction = model.temporal_predictor(outputs["identity"][:, 0])
    target = outputs["identity"][:, 1]
    if cfg.temporal_stop_gradient:
        target = target.detach()
    temporal = 1.0 - F.cosine_similarity(
        prediction.float(), target.float(), dim=1
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
    parts = {
        "temporal": temporal,
        "attribute": attribute_loss,
        "view": view_loss,
        "orthogonality": orthogonality,
    }
    return total, parts, valid_attributes


def branch_objective(
    branch,
    model,
    triplet,
    batch,
    cfg: ProposedV2Config,
    memory=None,
):
    frames, labels, attributes, views, track_indices = (
        value.to(cfg.device, non_blocking=True) for value in batch
    )
    outputs = model.forward_pairs(frames)
    common, parts, valid_attributes = common_pair_losses(
        model, outputs, attributes, views, cfg
    )
    repeated_labels = labels.repeat_interleave(frames.size(1))
    zero = outputs["identity"].sum() * 0.0
    identity_ce = zero
    metric_loss = zero
    cluster_loss = zero
    total = common

    if branch == "labeled":
        identity_ce = F.cross_entropy(
            outputs["identity_logits"].flatten(0, 1),
            repeated_labels,
            label_smoothing=cfg.label_smoothing,
        )
        if repeated_labels.unique().numel() > 1:
            metric_loss = triplet(
                F.normalize(outputs["identity"].flatten(0, 1), dim=1),
                repeated_labels,
            )
        total = (
            total
            + cfg.supervised_ce_weight * identity_ce
            + cfg.supervised_triplet_weight * metric_loss
        )
    elif branch == "pseudo":
        if memory is None:
            raise ValueError("The pseudo branch requires cluster memory")
        cluster_loss = cluster_nce(
            outputs["retrieval"].flatten(0, 1),
            repeated_labels,
            memory,
            cfg.cluster_temperature,
        )
        total = total + cfg.cluster_weight * cluster_loss
    else:
        raise ValueError(f"Unknown branch {branch}")

    parts.update(
        {"id_ce": identity_ce, "triplet": metric_loss, "cluster": cluster_loss}
    )
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


def _clustering_selection_score(stats: dict[str, float]) -> float:
    """Reward covered tracks only when their clusters remain separated.

    A coverage-heavy proxy selects an excessively large DBSCAN radius on this
    dataset because merging identities reduces the number of outliers. Positive
    silhouette times coverage instead favours conservative pseudo labels. The
    tiny cluster-count term only provides a deterministic fallback/tie-breaker.
    """

    return (
        stats["coverage"] * max(stats["silhouette"], 0.0)
        + 1e-6 * stats["clusters"]
    )


def train_proposed_v2(cfg: ProposedV2Config):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    query_loader, gallery_loader = build_eval_loaders(cfg)

    source = UnlabeledReIDDataset(cfg)
    labeled_pid_to_class, total_pids = select_labeled_identities(source, cfg)
    attribute_mapping = load_track_attributes(
        os.path.join(cfg.data_root, cfg.attr_file),
        cfg.ignore_unknown_attributes,
    )
    labeled_dataset = TrackPairDataset(
        source,
        cfg,
        labeled_pid_to_class,
        branch="labeled",
        attribute_mapping=attribute_mapping,
    )
    model = ProposedV2Model(cfg, len(labeled_pid_to_class)).to(cfg.device)
    load_personvit_finetune_initialization(model, cfg, labeled_pid_to_class)
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
        f"Proposed V2 | PersonViT={cfg.ssl_pretrained_path} | "
        f"stage1={cfg.initialization_checkpoint} | "
        f"labeled identities={len(labeled_pid_to_class)}/{total_pids} "
        f"({len(labeled_pid_to_class)/total_pids:.2%}) | "
        f"labeled pairs={len(labeled_dataset):,}"
    )
    prototypes = extract_track_prototypes(model, source, cfg)
    track_labels, memory, cluster_stats = cluster_tracks(prototypes, cfg)
    print(
        f"Initial DBSCAN | clusters={cluster_stats['clusters']} | "
        f"coverage={cluster_stats['coverage']:.2%} | "
        f"silhouette={cluster_stats['silhouette']:.4f}"
    )

    best_proxy = float("-inf")
    best_epoch = -1
    component_names = (
        "total",
        "id_ce",
        "triplet",
        "cluster",
        "temporal",
        "attribute",
        "view",
        "orthogonality",
    )

    for epoch in range(1, cfg.epochs + 1):
        pseudo_dataset = TrackPairDataset(
            source,
            cfg,
            labeled_pid_to_class,
            branch="pseudo",
            track_labels=track_labels,
            attribute_mapping=attribute_mapping,
        )
        labeled_loader = build_pk_loader(labeled_dataset, cfg, epoch)
        pseudo_loader = build_pk_loader(pseudo_dataset, cfg, epoch)
        steps = max(len(labeled_loader), len(pseudo_loader))
        labeled_iterator = iter(labeled_loader)
        pseudo_iterator = iter(pseudo_loader)
        memory = memory.to(cfg.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        valid_attribute_total = 0
        progress = tqdm(
            range(1, steps + 1),
            desc=f"Proposed-V2 {epoch:03d}/{cfg.epochs}",
            leave=False,
        )

        for step in progress:
            labeled_batch, labeled_iterator = _next_or_restart(
                labeled_iterator, labeled_loader
            )
            pseudo_batch, pseudo_iterator = _next_or_restart(
                pseudo_iterator, pseudo_loader
            )

            round_total = 0.0
            for branch, current_batch in (
                ("labeled", labeled_batch),
                ("pseudo", pseudo_batch),
            ):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, parts, valid_attributes, memory_payload, _ = branch_objective(
                        branch,
                        model,
                        triplet,
                        current_batch,
                        cfg,
                        memory=memory if branch == "pseudo" else None,
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
                        memory,
                        features,
                        pseudo_labels,
                        cfg.cluster_momentum,
                    )

            totals["total"] += round_total
            if step % cfg.grad_accum_steps == 0 or step == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(
                loss=f"{totals['total']/step:.3f}",
                clusters=cluster_stats["clusters"],
                coverage=f"{cluster_stats['coverage']:.1%}",
            )

        scheduler.step()
        for name in component_names:
            writer.add_scalar(f"Train/{name}", totals[name] / steps, epoch)
        writer.add_scalar("Train/valid_attributes", valid_attribute_total / steps, epoch)
        writer.add_scalar("Train/backbone_lr", scheduler.get_last_lr()[0], epoch)
        print(
            f"Epoch {epoch:03d} | total {totals['total']/steps:.4f} | "
            f"CE {totals['id_ce']/steps:.4f} | cluster {totals['cluster']/steps:.4f} | "
            f"temporal {totals['temporal']/steps:.4f}"
        )

        should_recluster = (
            epoch % cfg.recluster_interval == 0 or epoch == cfg.epochs
        )
        if should_recluster:
            prototypes = extract_track_prototypes(model, source, cfg)
            try:
                next_labels, next_memory, next_stats = cluster_tracks(prototypes, cfg)
            except (RuntimeError, ValueError) as error:
                print(
                    f"Epoch {epoch:03d} clustering warning: {error}; "
                    "keeping the previous pseudo labels"
                )
                next_labels = track_labels
                next_memory = memory.detach().cpu()
                next_stats = cluster_stats

            for name, value in next_stats.items():
                writer.add_scalar(f"Clustering/{name}", value, epoch)
            print(
                f"Recluster {epoch:03d} | clusters {next_stats['clusters']} | "
                f"coverage {next_stats['coverage']:.2%} | "
                f"silhouette {next_stats['silhouette']:.4f}"
            )
            selection_score = _clustering_selection_score(next_stats)
            writer.add_scalar("Clustering/selection_score", selection_score, epoch)
            if selection_score > best_proxy:
                best_proxy = selection_score
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "epoch": epoch,
                        "cluster_stats": next_stats,
                        "selection_proxy": best_proxy,
                        "labeled_pids": sorted(labeled_pid_to_class),
                        "identity_scheme": IDENTITY_SCHEME,
                        "train_identity_count": total_pids,
                        "method": "Proposed V2 cluster-temporal-attribute-view ReID",
                    },
                    checkpoint,
                )
            track_labels, memory, cluster_stats = (
                next_labels,
                next_memory,
                next_stats,
            )

    if best_epoch < 0 or not os.path.isfile(checkpoint):
        raise RuntimeError("Proposed V2 finished without producing a checkpoint")
    state = torch.load(
        checkpoint, map_location=cfg.device, weights_only=False
    )
    model.load_state_dict(state["model"])
    rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
    print(
        f"Proposed V2 final (best proxy epoch {state['epoch']:03d}) | "
        f"Rank-1 {rank1:.2%} | mAP {mean_ap:.2%}"
    )
    writer.add_hparams(
        {
            "labeled_fraction": cfg.labeled_fraction,
            "dbscan_eps": cfg.dbscan_eps,
            "recluster_interval": cfg.recluster_interval,
        },
        {"final/rank1": rank1, "final/mAP": mean_ap},
    )
    writer.close()
    return checkpoint, rank1, mean_ap
