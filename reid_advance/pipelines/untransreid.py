"""Local-friendly UntransReID-style baseline for unlabeled AG-ReID.v2.

This is an adaptation, not an exact reproduction. It retains target-domain
pseudo-label clustering, cluster-level contrastive memory, and masked-patch
consistency. To keep clustering practical on a local machine, DBSCAN operates
on bounded single-camera track prototypes rather than all 51k image features.
No identity token from a path is parsed during training.
"""

from __future__ import annotations

import os

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

from ..config import UntransReIDConfig
from ..data import (
    PseudoLabeledDataset,
    RandomIdentitySampler,
    TrackPrototypeDataset,
    UnlabeledReIDDataset,
)
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)


def _strip_prefixes(name: str) -> str:
    prefixes = ("module.", "backbone.", "encoder.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _find_state_dict(checkpoint):
    state = checkpoint
    for key in ("teacher", "state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("The SSL checkpoint does not contain a state dictionary")
    return {_strip_prefixes(name): value for name, value in state.items()}


def load_ssl_backbone(encoder, path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SSL pretrained checkpoint not found: {path}")
    # Official PersonViT checkpoints include argparse/numpy metadata in addition
    # to tensors. Only use weights_only=False for a checkpoint from a trusted
    # source such as the repository linked by the official PersonViT project.
    state = _find_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    target = encoder.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in target and hasattr(value, "shape") and value.shape == target[name].shape
    }
    if len(compatible) < max(10, len(target) // 2):
        raise ValueError(
            f"Only {len(compatible)}/{len(target)} encoder tensors match {path}; "
            "check that it is a ViT-S/16 PersonViT or TransReID-SSL checkpoint"
        )
    missing, unexpected = encoder.load_state_dict(compatible, strict=False)
    print(
        f"Loaded SSL backbone: {len(compatible)}/{len(target)} tensors | "
        f"missing {len(missing)} | unexpected {len(unexpected)}"
    )


class UntransReIDSmall(nn.Module):
    def __init__(self, cfg: UntransReIDConfig):
        super().__init__()
        use_timm_pretraining = cfg.pretrained and not cfg.ssl_pretrained_path
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=use_timm_pretraining,
            num_classes=0,
            img_size=cfg.image_size,
        )
        dim = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        if dim != cfg.embed_dim:
            raise ValueError(f"Expected embedding dimension {cfg.embed_dim}, got {dim}")
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.bottleneck.weight, 1.0, 0.02)
        nn.init.zeros_(self.bottleneck.bias)
        if cfg.ssl_pretrained_path:
            load_ssl_backbone(self.encoder, cfg.ssl_pretrained_path)
        else:
            initialization = (
                f"generic SSL ({cfg.encoder_name})" if use_timm_pretraining else "random"
            )
            print(
                f"WARNING: ssl_pretrained_path is empty; using {initialization} "
                "initialization. Report this run as UntransReID-style, not an exact "
                "TransReID-SSL/PersonViT initialization."
            )

    def _encode_tokens(self, images, mask_ratio=0.0):
        tokens = self.encoder.patch_embed(images)
        if mask_ratio > 0.0:
            mask = torch.rand(tokens.shape[:2], device=tokens.device) < mask_ratio
            tokens = torch.where(
                mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens
            )
        tokens = self.encoder._pos_embed(tokens)
        tokens = self.encoder.patch_drop(tokens)
        tokens = self.encoder.norm_pre(tokens)
        tokens = self.encoder.blocks(tokens)
        tokens = self.encoder.norm(tokens)
        return self.encoder.forward_head(tokens, pre_logits=True)

    def forward(self, images, mask_ratio=0.0):
        raw = self._encode_tokens(images, mask_ratio)
        return F.normalize(self.bottleneck(raw), dim=1)


@torch.no_grad()
def extract_track_prototypes(model, dataset, cfg):
    prototype_dataset = TrackPrototypeDataset(
        dataset, cfg.cluster_samples_per_track
    )
    loader = DataLoader(
        prototype_dataset,
        batch_size=cfg.cluster_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    model.eval()
    sums = torch.zeros(dataset.num_tracks, cfg.embed_dim, device=cfg.device)
    counts = torch.zeros(dataset.num_tracks, 1, device=cfg.device)
    use_amp = str(cfg.device).startswith("cuda")
    for images, track_indices in tqdm(loader, desc="track features", leave=False):
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
        raise RuntimeError("At least one track received no prototype samples")
    return F.normalize(sums / counts, dim=1).cpu()


def cluster_tracks(prototypes, cfg):
    raw_labels = DBSCAN(
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit_predict(prototypes.numpy())
    cluster_ids = sorted(set(raw_labels) - {-1})
    if len(cluster_ids) < 2:
        raise RuntimeError(
            f"DBSCAN produced {len(cluster_ids)} clusters at eps={cfg.dbscan_eps}. "
            "Tune UntransReIDConfig.dbscan_eps before training."
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
    proxy_score = coverage * (silhouette + 1.0) / 2.0
    centers = []
    for cluster_id in range(len(cluster_ids)):
        centers.append(prototypes[labels == cluster_id].mean(dim=0))
    memory = F.normalize(torch.stack(centers), dim=1)
    stats = {
        "clusters": len(cluster_ids),
        "coverage": coverage,
        "silhouette": silhouette,
        "proxy_score": proxy_score,
    }
    return labels, memory, stats


def build_pseudo_loader(dataset, track_labels, cfg, epoch):
    pseudo_dataset = PseudoLabeledDataset(
        dataset, track_labels, cfg.train_samples_per_track
    )
    sampler = RandomIdentitySampler(
        pseudo_dataset.labels,
        cfg.batch_size,
        cfg.instances_per_identity,
        seed=cfg.split_seed,
    )
    if len(set(pseudo_dataset.labels)) < sampler.identities_per_batch:
        raise RuntimeError(
            "Too few DBSCAN clusters for the configured P x K batch; lower "
            "batch_size or dbscan_eps"
        )
    sampler.set_epoch(epoch)
    loader = DataLoader(
        pseudo_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    return loader


def cluster_nce(features, labels, memory, temperature):
    logits = features.float() @ memory.t()
    return F.cross_entropy(logits / temperature, labels)


def masked_consistency(features, masked_features, temperature):
    targets = torch.arange(features.size(0), device=features.device)
    logits = features.float() @ masked_features.float().t() / temperature
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)
    )


@torch.no_grad()
def update_cluster_memory(memory, features, labels, momentum):
    for cluster_id in labels.unique():
        index = int(cluster_id.item())
        batch_center = features[labels == cluster_id].float().mean(dim=0)
        updated = momentum * memory[index] + (1.0 - momentum) * batch_center
        memory[index] = F.normalize(updated, dim=0)


def train_untransreid(cfg: UntransReIDConfig):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if cfg.ssl_pretrained_path and cfg.ssl_checkpoint_uses_half_normalization:
        cfg.norm_mean = (0.5, 0.5, 0.5)
        cfg.norm_std = (0.5, 0.5, 0.5)
        print("Using PersonViT/TransReID normalization: mean=std=(0.5, 0.5, 0.5)")
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)

    dataset = UnlabeledReIDDataset(cfg)
    query_loader, gallery_loader = build_eval_loaders(cfg)
    model = UntransReIDSmall(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, cfg.epochs, cfg.warmup_epochs
    )
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_proxy = float("-inf")

    print(
        f"Unlabeled images: {len(dataset.items):,} | single-camera tracks: "
        f"{dataset.num_tracks:,} | identity labels used for training: 0"
    )
    prototypes = extract_track_prototypes(model, dataset, cfg)
    track_labels, memory, cluster_stats = cluster_tracks(prototypes, cfg)

    for epoch in range(1, cfg.epochs + 1):
        loader = build_pseudo_loader(dataset, track_labels, cfg, epoch)
        memory = memory.to(cfg.device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "cluster": 0.0, "mask": 0.0}
        progress = tqdm(loader, desc=f"UntransReID {epoch:03d}/{cfg.epochs}", leave=False)
        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                features = model(images)
                masked_features = model(images, cfg.mask_ratio)
                loss_cluster = cluster_nce(
                    features, labels, memory, cfg.cluster_temperature
                )
                loss_mask = masked_consistency(
                    features, masked_features, cfg.mask_temperature
                )
                loss = loss_cluster + cfg.mask_consistency_weight * loss_mask
                scaled_loss = loss / cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            update_cluster_memory(
                memory, features.detach(), labels, cfg.cluster_momentum
            )
            totals["total"] += loss.item()
            totals["cluster"] += loss_cluster.item()
            totals["mask"] += loss_mask.item()
            progress.set_postfix(
                loss=f"{totals['total']/step:.3f}",
                clusters=cluster_stats["clusters"],
                coverage=f"{cluster_stats['coverage']:.1%}",
            )
        scheduler.step()

        prototypes = extract_track_prototypes(model, dataset, cfg)
        try:
            next_labels, next_memory, next_stats = cluster_tracks(prototypes, cfg)
        except RuntimeError as error:
            print(f"Epoch {epoch:03d} clustering warning: {error}; keeping prior labels")
            next_labels, next_memory, next_stats = track_labels, memory.cpu(), cluster_stats

        for name, value in totals.items():
            writer.add_scalar(f"Train/{name}", value / len(loader), epoch)
        for name, value in next_stats.items():
            writer.add_scalar(f"Clustering/{name}", value, epoch)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
        print(
            f"Epoch {epoch:03d} | loss {totals['total']/len(loader):.4f} | "
            f"clusters {next_stats['clusters']} | coverage {next_stats['coverage']:.2%} | "
            f"silhouette {next_stats['silhouette']:.4f}"
        )

        if next_stats["proxy_score"] > best_proxy:
            best_proxy = next_stats["proxy_score"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "unsupervised_proxy": best_proxy,
                    "cluster_stats": next_stats,
                    "method": "local UntransReID-style AG-ReID.v2 adaptation",
                },
                checkpoint,
            )
        track_labels, memory, cluster_stats = next_labels, next_memory, next_stats

    state = torch.load(checkpoint, map_location=cfg.device)
    model.load_state_dict(state["model"])
    rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
    print(
        f"UntransReID final (best proxy epoch {state['epoch']:03d}) | "
        f"Rank-1: {rank1:.2%} | mAP: {mean_ap:.2%}"
    )
    writer.add_hparams(
        {
            "dbscan_eps": cfg.dbscan_eps,
            "mask_ratio": cfg.mask_ratio,
            "ssl_checkpoint": int(bool(cfg.ssl_pretrained_path)),
        },
        {"final/rank1": rank1, "final/mAP": mean_ap},
    )
    writer.close()
    return checkpoint, rank1, mean_ap
