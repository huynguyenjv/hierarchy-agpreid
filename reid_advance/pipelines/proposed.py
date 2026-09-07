"""LeWorldModel-inspired semi-supervised temporal ReID on AG-ReID.v2.

This is an adaptation, not a reproduction of LeWorldModel. AG-ReID.v2 has
video frames but no robot actions, so the predictor learns action-free temporal
dynamics within each identity tracklet. The paper's two core ingredients are
preserved: next-embedding prediction and SIGReg anti-collapse regularization.
"""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..attributes import ATTRIBUTE_CLASS_COUNTS
from ..config import ProposedConfig
from ..data import TemporalTrackletDataset
from ..losses import TripletLoss, masked_attribute_loss
from ..identity import IDENTITY_SCHEME, require_current_identity_scheme
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer from LeWorldModel."""

    def __init__(self, knots=17, num_projections=256):
        super().__init__()
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        trapezoid = torch.full((knots,), 2 * dt, dtype=torch.float32)
        trapezoid[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", gaussian_cf)
        self.register_buffer("weights", trapezoid * gaussian_cf)

    def forward(self, embeddings):
        """embeddings: [time, batch, dimension]."""
        directions = torch.randn(
            embeddings.size(-1), self.num_projections, device=embeddings.device
        )
        directions = F.normalize(directions, dim=0)
        projected = (embeddings @ directions).unsqueeze(-1) * self.t
        error = (
            (projected.cos().mean(-3) - self.phi).square()
            + projected.sin().mean(-3).square()
        )
        statistic = (error @ self.weights) * embeddings.size(-2)
        return statistic.mean()


class TemporalPredictor(nn.Module):
    """Causal action-free predictor for AG-ReID tracklet embeddings."""

    def __init__(self, dim, sequence_length, depth, heads, dropout):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, sequence_length, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, history):
        length = history.size(1)
        x = history + self.position[:, :length]
        causal_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
        )
        return self.norm(self.transformer(x, mask=causal_mask))


class LeWMReID(nn.Module):
    def __init__(self, cfg: ProposedConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            img_size=cfg.image_size,
        )
        dim = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        if dim != cfg.embed_dim:
            raise ValueError(f"Expected embedding dimension {cfg.embed_dim}, got {dim}")
        # LeWM places a one-layer projection with BN after the encoder CLS token.
        self.projector = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.predictor = TemporalPredictor(
            dim,
            cfg.sequence_length,
            cfg.predictor_depth,
            cfg.predictor_heads,
            cfg.predictor_dropout,
        )
        self.identity_head = nn.Linear(dim, num_classes, bias=False)
        self.attribute_heads = nn.ModuleDict({
            name: nn.Linear(dim, classes)
            for name, classes in ATTRIBUTE_CLASS_COUNTS.items()
        })

    def encode(self, images):
        raw = self.encoder(images)
        return self.projector(raw)

    def forward(self, images):
        if images.ndim == 4:
            return F.normalize(self.encode(images), dim=1)
        batch, time = images.shape[:2]
        embeddings = self.encode(images.flatten(0, 1)).view(batch, time, -1)
        predictions = self.predictor(embeddings[:, :-1])
        flat = embeddings.flatten(0, 1)
        identity_logits = self.identity_head(flat).view(batch, time, -1)
        attribute_logits = {
            name: head(flat) for name, head in self.attribute_heads.items()
        }
        return embeddings, predictions, identity_logits, attribute_logits


def split_by_track(dataset, validation_fraction, seed):
    generator = torch.Generator().manual_seed(seed)
    track_groups = {}
    for index, (paths, _, _) in enumerate(dataset.windows):
        track_groups.setdefault(os.path.dirname(paths[0]), []).append(index)
    tracks = list(sorted(track_groups))
    order = torch.randperm(len(tracks), generator=generator).tolist()
    number_val = max(1, round(len(tracks) * validation_fraction))
    val_tracks = {tracks[index] for index in order[:number_val]}
    train_indices, val_indices = [], []
    for track, indices in track_groups.items():
        (val_indices if track in val_tracks else train_indices).extend(indices)
    # Guarantee every identity represented in the training partition has at
    # least one labeled window; otherwise its classifier row is untrainable.
    train_by_pid = {}
    for index in train_indices:
        _, pid, is_labeled = dataset.windows[index]
        train_by_pid.setdefault(pid, []).append((index, is_labeled))
    for entries in train_by_pid.values():
        if not any(is_labeled for _, is_labeled in entries):
            index = entries[0][0]
            paths, pid, _ = dataset.windows[index]
            dataset.windows[index] = (paths, pid, True)
    dataset.labeled_window_count = sum(flag for _, _, flag in dataset.windows)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_objective(cfg, sigreg, triplet, model, batch, device):
    frames, labels, attributes = (
        value.to(device, non_blocking=True) for value in batch
    )
    embeddings, predictions, identity_logits, attribute_logits = model(frames)
    prediction_loss = F.mse_loss(predictions, embeddings[:, 1:])
    # Trigonometric normality statistics are kept in fp32 under AMP.
    sigreg_loss = sigreg(embeddings.float().transpose(0, 1))

    labeled = labels >= 0
    zero = embeddings.sum() * 0.0
    supervised_ce = zero
    supervised_triplet = zero
    if torch.any(labeled):
        repeated_labels = labels[labeled].repeat_interleave(embeddings.size(1))
        labeled_logits = identity_logits[labeled].flatten(0, 1)
        labeled_features = embeddings[labeled].flatten(0, 1)
        supervised_ce = F.cross_entropy(labeled_logits, repeated_labels)
        if repeated_labels.unique().numel() > 1:
            supervised_triplet = triplet(labeled_features, repeated_labels)

    repeated_attributes = attributes.repeat_interleave(embeddings.size(1), dim=0)
    attribute_loss, valid_attributes = masked_attribute_loss(
        attribute_logits, attribute_logits, repeated_attributes
    )
    total = (
        cfg.temporal_prediction_weight * prediction_loss
        + cfg.sigreg_weight * sigreg_loss
        + cfg.supervised_ce_weight * supervised_ce
        + cfg.supervised_triplet_weight * supervised_triplet
        + cfg.attribute_weight * attribute_loss
    )
    components = {
        "prediction": prediction_loss,
        "sigreg": sigreg_loss,
        "id_ce": supervised_ce,
        "triplet": supervised_triplet,
        "attribute": attribute_loss,
    }
    return total, components, int(labeled.sum().item()), valid_attributes


def _loader(dataset, cfg, shuffle, drop_last):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
    )


def train_proposed(cfg: ProposedConfig):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))
    dataset = TemporalTrackletDataset(cfg)
    train_set, val_set = split_by_track(dataset, cfg.validation_fraction, cfg.split_seed)
    train_loader = _loader(train_set, cfg, True, True)
    val_loader = _loader(val_set, cfg, False, True)
    query_loader, gallery_loader = build_eval_loaders(cfg)
    print(
        f"Temporal windows: {len(dataset):,} | identities: {dataset.num_classes} "
        f"(P+T+A) | labeled: "
        f"{dataset.labeled_window_count/len(dataset):.2%} | train/val: "
        f"{len(train_set):,}/{len(val_set):,}"
    )

    model = LeWMReID(cfg, dataset.num_classes).to(cfg.device)
    sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_num_projections).to(cfg.device)
    triplet = TripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = warmup_cosine_scheduler(
        optimizer, cfg.epochs, cfg.warmup_epochs
    )
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    best_validation = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("total", "prediction", "sigreg", "id_ce", "triplet", "attribute")}
        progress = tqdm(train_loader, desc=f"LeWM-ReID {epoch:03d}/{cfg.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, parts, labeled, valid_attributes = build_objective(
                    cfg, sigreg, triplet, model, batch, cfg.device
                )
                scaled_loss = loss / cfg.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            totals["total"] += loss.item()
            for name, value in parts.items():
                totals[name] += value.item()
            progress.set_postfix(
                loss=f"{totals['total']/step:.3f}", labeled=labeled, attr=valid_attributes
            )
        scheduler.step()

        model.eval()
        validation = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, _, _, _ = build_objective(
                        cfg, sigreg, triplet, model, batch, cfg.device
                    )
                validation += loss.item()
        validation /= len(val_loader)
        for name, value in totals.items():
            writer.add_scalar(f"Train/{name}", value / len(train_loader), epoch)
        writer.add_scalar("Validation/loss", validation, epoch)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch:03d} | train {totals['total']/len(train_loader):.4f} | val {validation:.4f}")

        if validation < best_validation:
            best_validation = validation
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "validation_loss": validation,
                    "identity_scheme": IDENTITY_SCHEME,
                    "train_identity_count": dataset.num_classes,
                    "method": "LeWM-inspired action-free temporal ReID adaptation",
                },
                checkpoint,
            )

    state = torch.load(checkpoint, map_location=cfg.device)
    require_current_identity_scheme(state, checkpoint)
    model.load_state_dict(state["model"])
    rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
    print(f"LeWM-ReID final | Rank-1: {rank1:.2%} | mAP: {mean_ap:.2%}")
    writer.add_hparams(
        {"labeled_fraction": cfg.labeled_fraction, "sigreg_weight": cfg.sigreg_weight},
        {"final/rank1": rank1, "final/mAP": mean_ap},
    )
    writer.close()
    return checkpoint, rank1, mean_ap
