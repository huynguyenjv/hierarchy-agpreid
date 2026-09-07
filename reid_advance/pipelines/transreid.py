"""Local-friendly TransReID-S supervised baseline for AG-ReID.v2.

This keeps the two central TransReID ideas: side-information embeddings (SIE)
and optional local part features inspired by JPM.  The default configuration
uses ViT-Small and disables JPM so it can fit on a low-memory GPU.
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

from ..config import TransReIDConfig
from ..data import RandomIdentitySampler, TransReIDDataset
from ..losses import TripletLoss
from ..identity import IDENTITY_SCHEME, require_current_identity_scheme
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)


def camera_to_view(camera_ids: torch.Tensor) -> torch.Tensor:
    """AG-ReID.v2 cameras: C0 aerial, C2 wearable, C3 CCTV."""
    result = torch.zeros_like(camera_ids, dtype=torch.long)
    result[camera_ids == 2] = 1
    result[camera_ids == 3] = 2
    return result


class TransReIDSmall(nn.Module):
    accepts_camera_ids = True

    def __init__(self, cfg: TransReIDConfig, num_classes: int):
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

        self.sie = nn.Embedding(3, dim) if cfg.transreid_sie else None
        if self.sie is not None:
            nn.init.trunc_normal_(self.sie.weight, std=0.02)
        self.global_bn = nn.BatchNorm1d(dim)
        self.global_bn.bias.requires_grad_(False)
        self.global_classifier = nn.Linear(dim, num_classes, bias=False)

        self.local_bns = nn.ModuleList()
        self.local_classifiers = nn.ModuleList()
        if cfg.transreid_jpm:
            for _ in range(cfg.transreid_num_parts):
                bn = nn.BatchNorm1d(dim)
                bn.bias.requires_grad_(False)
                self.local_bns.append(bn)
                self.local_classifiers.append(nn.Linear(dim, num_classes, bias=False))
        self._init_heads()

    def _init_heads(self):
        modules = [self.global_bn, self.global_classifier, *self.local_bns, *self.local_classifiers]
        for module in modules:
            if isinstance(module, nn.BatchNorm1d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)

    def _forward_tokens(self, images, camera_ids):
        # timm VisionTransformer path, with SIE inserted before transformer blocks.
        x = self.encoder.patch_embed(images)
        x = self.encoder._pos_embed(x)
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)
        if self.sie is not None:
            views = camera_to_view(camera_ids)
            x = x + self.sie(views).unsqueeze(1)
        x = self.encoder.blocks(x)
        return self.encoder.norm(x)

    def forward(self, images, camera_ids=None):
        if camera_ids is None:
            camera_ids = torch.zeros(images.size(0), device=images.device, dtype=torch.long)
        tokens = self._forward_tokens(images, camera_ids)
        global_raw = tokens[:, 0]
        global_bn = self.global_bn(global_raw)

        raw_features = [global_raw]
        retrieval_features = [global_bn]
        logits = [self.global_classifier(global_bn)]

        if self.cfg.transreid_jpm:
            patch_tokens = tokens[:, 1:]
            chunks = torch.tensor_split(patch_tokens, self.cfg.transreid_num_parts, dim=1)
            for chunk, bn, classifier in zip(chunks, self.local_bns, self.local_classifiers):
                raw = chunk.mean(dim=1)
                normalized = bn(raw)
                raw_features.append(raw)
                retrieval_features.append(normalized)
                logits.append(classifier(normalized))

        if self.training:
            return {"features": raw_features, "logits": logits}
        return F.normalize(torch.cat(retrieval_features, dim=1), dim=1)

    def optimizer_groups(self):
        heads = [self.global_bn, self.global_classifier, self.local_bns, self.local_classifiers]
        return [
            {"params": self.encoder.parameters(), "lr": self.cfg.learning_rate},
            {
                "params": self.sie.parameters() if self.sie is not None else [],
                "lr": self.cfg.learning_rate * self.cfg.head_lr_multiplier,
            },
            {
                "params": [p for module in heads for p in module.parameters()],
                "lr": self.cfg.learning_rate * self.cfg.head_lr_multiplier,
            },
        ]


def train_transreid(cfg: TransReIDConfig):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))

    dataset = TransReIDDataset(cfg)
    print(
        f"TransReID dataset | images={len(dataset):,} | "
        f"identities={dataset.num_classes} | identity=P+T+A"
    )
    sampler = RandomIdentitySampler(
        dataset.labels,
        cfg.batch_size,
        cfg.instances_per_identity,
        seed=cfg.split_seed,
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
    model = TransReIDSmall(cfg, dataset.num_classes).to(cfg.device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(
        optimizer, cfg.epochs, cfg.warmup_epochs
    )
    triplet = TripletLoss(margin=0.3)
    use_amp = str(cfg.device).startswith("cuda") and getattr(cfg, "use_amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    best_map = -1.0
    best_rank1 = -1.0
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(loader, desc=f"TransReID {epoch:03d}/{cfg.epochs}", leave=False)
        for step, (images, labels, camera_ids) in enumerate(progress, start=1):
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            camera_ids = camera_ids.to(cfg.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(images, camera_ids)
                ce = torch.stack([
                    F.cross_entropy(logit, labels, label_smoothing=cfg.label_smoothing)
                    for logit in output["logits"]
                ]).mean()
                tri = torch.stack([
                    triplet(feature, labels) for feature in output["features"]
                ]).mean()
                loss = (ce + tri) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.item() * cfg.grad_accum_steps
            progress.set_postfix(loss=f"{running/step:.4f}")
        scheduler.step()
        writer.add_scalar("Train/loss", running / len(loader), epoch)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
            print(f"TransReID epoch {epoch:03d} | Rank-1 {rank1:.2%} | mAP {mean_ap:.2%}")
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

    if best_epoch < 0:
        raise RuntimeError("TransReID finished without producing a valid checkpoint")
    state = torch.load(checkpoint, map_location=cfg.device)
    require_current_identity_scheme(state, checkpoint)
    model.load_state_dict(state["model"])
    print(
        f"TransReID best (epoch {state['epoch']:03d}) | "
        f"Rank-1: {best_rank1:.2%} | mAP: {best_map:.2%}"
    )
    writer.add_hparams(
        {"jpm": int(cfg.transreid_jpm), "batch_size": cfg.batch_size},
        {"best/rank1": best_rank1, "best/mAP": best_map},
    )
    writer.close()
    return checkpoint, best_rank1, best_map
