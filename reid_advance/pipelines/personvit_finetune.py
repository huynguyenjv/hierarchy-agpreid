"""PersonViT paper-style supervised downstream fine-tuning.

PersonViT itself is the LUPerson self-supervised pre-training stage.  Its
released downstream recipe is the TransReID-SSL/BOT global baseline: vanilla
ViT-S/16 CLS features, BNNeck, identity cross entropy and batch-hard triplet.
This local lane applies that recipe to a deterministic AG-ReID.v2 PID budget
without clustering, attributes, SIE, JPM, or any proposed-method component.
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

from ..config import PersonViTFineTuneConfig
from ..data import (
    RandomIdentitySampler,
    SupervisedReIDDataset,
    select_identity_budget,
)
from ..losses import TripletLoss
from ..identity import IDENTITY_SCHEME, require_current_identity_scheme
from ..runtime import (
    build_eval_loaders,
    evaluate_model,
    resolve_device,
    warmup_cosine_scheduler,
)
from .untransreid import load_ssl_backbone


class PersonViTFineTuneModel(nn.Module):
    """Vanilla PersonViT-S/16 CLS encoder with the paper's BOT heads."""

    def __init__(self, cfg: PersonViTFineTuneConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.encoder_name,
            pretrained=False,
            num_classes=0,
            img_size=cfg.image_size,
        )
        dimension = int(getattr(self.encoder, "num_features", cfg.embed_dim))
        if dimension != cfg.embed_dim:
            raise ValueError(
                f"Expected PersonViT embedding dimension {cfg.embed_dim}, got {dimension}"
            )
        if not cfg.ssl_pretrained_path:
            raise ValueError("PersonViT fine-tuning requires the teacher checkpoint")
        load_ssl_backbone(self.encoder, cfg.ssl_pretrained_path)

        self.bnneck = nn.BatchNorm1d(dimension)
        self.bnneck.bias.requires_grad_(False)
        self.identity_classifier = nn.Linear(dimension, num_classes, bias=False)
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)
        nn.init.normal_(self.identity_classifier.weight, std=0.001)

    def forward(self, images):
        # timm's vanilla ViT forward returns the final CLS token when
        # num_classes=0, matching z_cls in the PersonViT paper.
        global_feature = self.encoder(images)
        if self.training:
            neck_feature = self.bnneck(global_feature)
            return global_feature, self.identity_classifier(neck_feature)
        # The released PersonViT config uses TEST.NECK_FEAT='before' and
        # feature normalization, so retrieval uses normalized raw CLS features.
        return F.normalize(global_feature, dim=1)


def _optimizer(model, cfg, learning_rate):
    # Match the released TransReID optimizer's 2x LR for trainable biases.
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        groups.append(
            {
                "params": [parameter],
                "lr": learning_rate * (2.0 if name.endswith("bias") else 1.0),
                "weight_decay": cfg.weight_decay,
            }
        )
    return torch.optim.SGD(groups, momentum=cfg.momentum)


def train_personvit_finetune(cfg: PersonViTFineTuneConfig):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))
    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)

    complete_dataset = SupervisedReIDDataset(cfg)
    labeled_pid_to_class, total_pids = select_identity_budget(
        (path for path, _ in complete_dataset.items),
        cfg.filename_pattern,
        cfg.labeled_fraction,
        cfg.labeled_split_seed,
        minimum=cfg.batch_size // cfg.instances_per_identity,
    )
    dataset = SupervisedReIDDataset(cfg, pid_to_class=labeled_pid_to_class)
    sampler = RandomIdentitySampler(
        dataset.labels,
        cfg.batch_size,
        cfg.instances_per_identity,
        cfg.split_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
        persistent_workers=cfg.num_workers > 0,
    )
    query_loader, gallery_loader = build_eval_loaders(cfg)

    model = PersonViTFineTuneModel(cfg, dataset.num_classes).to(cfg.device)
    effective_lr = (
        cfg.reference_learning_rate * cfg.batch_size / cfg.reference_batch_size
    )
    optimizer = _optimizer(model, cfg, effective_lr)
    scheduler = warmup_cosine_scheduler(
        optimizer, cfg.epochs, cfg.warmup_epochs, eta_min=1e-7
    )
    triplet = TripletLoss(margin=cfg.triplet_margin)
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(
        "PersonViT fine-tune | vanilla ViT-S/16 CLS + BNNeck + CE + Triplet "
        "| no DBSCAN/SIE/JPM"
    )
    print(
        f"PersonViT={cfg.ssl_pretrained_path} | input={cfg.image_size} | "
        f"labeled identities={dataset.num_classes}/{total_pids} "
        f"({dataset.num_classes/total_pids:.2%}) | images={len(dataset):,} | "
        f"P x K={cfg.batch_size // cfg.instances_per_identity} x "
        f"{cfg.instances_per_identity} | SGD lr={effective_lr:.2e}"
    )

    best_objective = float("inf")
    best_epoch = -1
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        total_ce = 0.0
        total_triplet = 0.0
        progress = tqdm(
            loader,
            desc=f"PersonViT-FT {epoch:03d}/{cfg.epochs}",
            leave=False,
        )
        for images, labels in progress:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                raw_features, logits = model(images)
                identity_loss = F.cross_entropy(
                    logits,
                    labels,
                    label_smoothing=cfg.label_smoothing,
                )
                metric_loss = triplet(raw_features, labels)
                loss = identity_loss + metric_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ce += identity_loss.item()
            total_triplet += metric_loss.item()
            progress.set_postfix(
                loss=f"{total_loss/(progress.n+1):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        mean_loss = total_loss / len(loader)
        writer.add_scalar("Train/total", mean_loss, epoch)
        writer.add_scalar("Train/CE", total_ce / len(loader), epoch)
        writer.add_scalar("Train/triplet", total_triplet / len(loader), epoch)
        writer.add_scalar("Train/lr", scheduler.get_last_lr()[0], epoch)
        print(
            f"Epoch {epoch:03d} | total {mean_loss:.4f} | "
            f"CE {total_ce/len(loader):.4f} | "
            f"triplet {total_triplet/len(loader):.4f}"
        )

        # Use a label-only training proxy rather than repeatedly selecting on
        # the official query/gallery test set.  Only this single file is kept.
        if mean_loss < best_objective:
            best_objective = mean_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "training_objective": mean_loss,
                    "labeled_pids": sorted(labeled_pid_to_class),
                    "identity_scheme": IDENTITY_SCHEME,
                    "train_identity_count": total_pids,
                    "method": "PersonViT ViT-S/16 BOT supervised fine-tune",
                },
                checkpoint,
            )

    if best_epoch < 0:
        raise RuntimeError("PersonViT fine-tuning did not produce a checkpoint")
    state = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
    require_current_identity_scheme(state, checkpoint)
    model.load_state_dict(state["model"])
    rank1, mean_ap = evaluate_model(model, cfg, query_loader, gallery_loader)
    print(
        f"PersonViT-FT final (best label-only epoch {state['epoch']:03d}) | "
        f"Rank-1 {rank1:.2%} | mAP {mean_ap:.2%}"
    )
    writer.add_hparams(
        {"labeled_fraction": cfg.labeled_fraction, "batch_size": cfg.batch_size},
        {"final/rank1": rank1, "final/mAP": mean_ap},
    )
    writer.close()
    return checkpoint, rank1, mean_ap
