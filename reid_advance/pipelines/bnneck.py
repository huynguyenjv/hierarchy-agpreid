"""ViT-S/16 + BNNeck supervised baseline for AG-ReID.v2.

The neck, P x K sampler, augmentations, identity loss and metric loss retain the
Bag-of-Tricks recipe. The backbone and optimizer are deliberately modernized to
match the local TransReID-S lane for a controlled ViT-to-ViT comparison.
"""

from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import BNNeckConfig
from ..data import RandomIdentitySampler, SupervisedReIDDataset
from ..losses import TripletLoss
from ..identity import IDENTITY_SCHEME
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)


class BNNeckModel(nn.Module):
    def __init__(self, cfg: BNNeckConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            img_size=cfg.image_size,
        )
        print(
            f"BNNeck backbone: {cfg.encoder_name} | pretrained={cfg.pretrained} "
            f"| input={cfg.image_size}"
        )
        dim = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        if dim != cfg.embed_dim:
            raise ValueError(f"Expected embedding dimension {cfg.embed_dim}, got {dim}")
        self.bottleneck = nn.BatchNorm1d(cfg.embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(cfg.embed_dim, num_classes, bias=False)
        nn.init.ones_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images):
        raw = self.encoder(images)
        normalized = self.bottleneck(raw)
        if self.training:
            return raw, self.classifier(normalized)
        return F.normalize(normalized, dim=1)

    def optimizer_groups(self):
        return [
            {"params": self.encoder.parameters(), "lr": self.cfg.learning_rate},
            {
                "params": [*self.bottleneck.parameters(), *self.classifier.parameters()],
                "lr": self.cfg.learning_rate * self.cfg.head_lr_multiplier,
            },
        ]


def train_bnneck(cfg: BNNeckConfig):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))
    dataset = SupervisedReIDDataset(cfg)
    print(
        f"BNNeck dataset | images={len(dataset):,} | "
        f"identities={dataset.num_classes} | identity=P+T+A"
    )
    sampler = RandomIdentitySampler(
        dataset.labels, cfg.batch_size, cfg.instances_per_identity, cfg.split_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    query_loader, gallery_loader = build_eval_loaders(cfg)
    model = BNNeckModel(cfg, dataset.num_classes).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.optimizer_groups(),
        weight_decay=cfg.weight_decay,
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        cfg.epochs,
        cfg.warmup_epochs,
    )
    triplet = TripletLoss(margin=0.3)
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_map = -1.0
    best_rank1 = -1.0
    best_epoch = -1
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        running = 0.0
        progress = tqdm(loader, desc=f"BNNeck {epoch:03d}/{cfg.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                features, logits = model(images)
                identity_loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=cfg.label_smoothing,
                )
                metric_loss = triplet(features, labels)
                loss = identity_loss + metric_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(
                loss=f"{running/(progress.n+1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        scheduler.step()
        writer.add_scalar("Train/loss", running / len(loader), epoch)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
            print(f"BNNeck epoch {epoch:03d} | Rank-1 {rank1:.2%} | mAP {mean_ap:.2%}")
            writer.add_scalar("Evaluation/rank1", rank1, epoch)
            writer.add_scalar("Evaluation/mAP", mean_ap, epoch)
            if mean_ap > best_map:
                best_map = mean_ap
                best_rank1 = rank1
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.__dict__,
                        "epoch": epoch,
                        "rank1": rank1,
                        "mAP": mean_ap,
                        "identity_scheme": IDENTITY_SCHEME,
                        "train_identity_count": dataset.num_classes,
                    },
                    checkpoint,
                )
    print(
        f"BNNeck best (epoch {best_epoch:03d}) | Rank-1 {best_rank1:.2%} "
        f"| mAP {best_map:.2%} | checkpoint {checkpoint}"
    )
    if best_epoch < 0:
        raise RuntimeError("BNNeck finished without producing a valid checkpoint")
    writer.close()
    return checkpoint, best_map
