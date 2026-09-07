"""Train TransReID-S on CARGO with the AG-ReID.v2 recipe, unchanged.

This exists to answer one question: is the flat camera-pair matrix measured on
AG-ReID.v2 a property of that dataset, or of aerial-ground ReID once ordinary
supervision has done its work?  The comparison only means something if both
sides are trained identically, so this module reuses ``TransReIDSmall``, the
identity + triplet objective, the PK sampler and ``evaluate_rank`` verbatim.
What differs is only what CARGO structurally forces: its filenames, its 13
cameras, and its query/gallery directories in place of protocol text files.

It is deliberately not a CARGO SOTA attempt. Absolute mAP does not matter here;
what matters is the spread between camera pairs, measured the same way twice.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..cargo import CargoDataset, CargoEvalDataset
from ..config import CargoConfig
from ..data import RandomIdentitySampler
from ..evaluation import evaluate_rank, extract_features
from ..losses import TripletLoss
from ..runtime import resolve_device, warmup_cosine_scheduler
from .transreid import TransReIDSmall


def build_loaders(cfg: CargoConfig):
    dataset = CargoDataset(
        os.path.join(cfg.data_root, cfg.train_dir),
        cfg.image_size, cfg.norm_mean, cfg.norm_std,
        cfg.flip_probability, cfg.padding, cfg.random_erasing_probability,
    )
    print(f"CARGO train | images={len(dataset):,} | identities={dataset.num_classes:,}")

    sampler = RandomIdentitySampler(
        dataset.labels, cfg.batch_size, cfg.instances_per_identity, seed=cfg.split_seed
    )
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=str(cfg.device).startswith("cuda"),
    )

    eval_kwargs = {
        "batch_size": cfg.eval_batch_size, "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": str(cfg.device).startswith("cuda"),
    }
    query = DataLoader(
        CargoEvalDataset(os.path.join(cfg.data_root, cfg.query_dir),
                         cfg.image_size, cfg.norm_mean, cfg.norm_std),
        **eval_kwargs,
    )
    gallery = DataLoader(
        CargoEvalDataset(os.path.join(cfg.data_root, cfg.gallery_dir),
                         cfg.image_size, cfg.norm_mean, cfg.norm_std),
        **eval_kwargs,
    )
    print(f"CARGO eval  | query={len(query.dataset):,} | gallery={len(gallery.dataset):,}")
    return dataset, sampler, loader, query, gallery


@torch.no_grad()
def evaluate(model, cfg, query_loader, gallery_loader):
    query_features, query_pids, query_cams = extract_features(
        model, query_loader, cfg.device, flip_tta=cfg.flip_tta
    )
    gallery_features, gallery_pids, gallery_cams = extract_features(
        model, gallery_loader, cfg.device, flip_tta=cfg.flip_tta
    )
    distance = (1.0 - query_features @ gallery_features.t()).numpy()
    cmc, mean_ap = evaluate_rank(
        distance, query_pids, gallery_pids, query_cams, gallery_cams
    )
    return float(cmc[0]), float(mean_ap)


def train_cargo(cfg: CargoConfig):
    resolve_device(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset, sampler, loader, query_loader, gallery_loader = build_loaders(cfg)
    model = TransReIDSmall(cfg, dataset.num_classes).to(cfg.device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)
    triplet = TripletLoss(margin=0.3)
    use_amp = str(cfg.device).startswith("cuda") and cfg.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    resume_path = os.path.join(cfg.output_dir, "last.pth")
    best_map, best_rank1, best_epoch = -1.0, -1.0, -1
    start_epoch = 1

    # A 2.7-hour run on a shared machine will occasionally be interrupted, and
    # losing everything because the first eval had not been reached yet is a
    # waste this loop can simply avoid.
    if getattr(cfg, "resume", True) and os.path.exists(resume_path):
        saved = torch.load(resume_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"] + 1
        best_map = saved.get("best_map", -1.0)
        best_rank1 = saved.get("best_rank1", -1.0)
        best_epoch = saved.get("best_epoch", -1)
        print(f"resumed from {resume_path} at epoch {saved['epoch']}, "
              f"continuing at {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(loader, desc=f"CARGO {epoch:03d}/{cfg.epochs}", leave=False)
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
            progress.set_postfix(loss=f"{running / step:.4f}")
        scheduler.step()
        print(f"CARGO epoch {epoch:03d} | loss {running / len(loader):.4f}", flush=True)

        # Written every epoch, not just on eval epochs, so an interruption costs
        # at most one epoch instead of everything since the last evaluation.
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "best_rank1": best_rank1,
                "best_epoch": best_epoch,
                "config": {k: v for k, v in cfg.__dict__.items()},
                "dataset": "CARGO",
                "train_identity_count": dataset.num_classes,
            },
            resume_path,
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            rank1, mean_ap = evaluate(model, cfg, query_loader, gallery_loader)
            print(f"CARGO epoch {epoch:03d} | Rank-1 {rank1:.2%} | mAP {mean_ap:.2%}")

            # Keep every evaluated epoch under its own name. Convergence here is
            # judged by whether the camera-pair matrix has stopped moving, not
            # by mAP, so the matrix has to be computable at two epochs - which
            # the best-only checkpoint would make impossible once a later epoch
            # overwrites an earlier one.
            snapshot = {
                "model": model.state_dict(),
                "config": {k: v for k, v in cfg.__dict__.items()},
                "epoch": epoch,
                "rank1": rank1,
                "mAP": mean_ap,
                "dataset": "CARGO",
                "train_identity_count": dataset.num_classes,
            }
            torch.save(snapshot, os.path.join(cfg.output_dir, f"epoch_{epoch:03d}.pth"))

            if mean_ap > best_map:
                best_map, best_rank1, best_epoch = mean_ap, rank1, epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {k: v for k, v in cfg.__dict__.items()},
                        "epoch": epoch,
                        "rank1": rank1,
                        "mAP": mean_ap,
                        "dataset": "CARGO",
                        "train_identity_count": dataset.num_classes,
                    },
                    checkpoint,
                )

    if best_epoch < 0:
        raise RuntimeError("CARGO training finished without a valid checkpoint")
    print(f"CARGO best (epoch {best_epoch:03d}) | Rank-1 {best_rank1:.2%} | mAP {best_map:.2%}")
    return checkpoint, best_rank1, best_map
