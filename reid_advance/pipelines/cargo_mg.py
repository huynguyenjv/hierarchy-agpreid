"""The three-branch experiment: baseline, uniform-beta, view-aware.

All three share a backbone, a sampler, a batch size, a schedule and a seed. The
only thing that varies is the objective:

    baseline      L_ID + L_triplet
    uniform-beta  L_ID + L_triplet + alpha * L_MG, coarse_bias = 0
    view-aware    L_ID + L_triplet + alpha * L_MG, coarse_bias > 0

Uniform-beta is the branch that matters. It carries the full granularity
pyramid at identical cost, and differs from view-aware in one respect only:
whether the level weighting knows which pairs cross platforms. Without it,
"multi-granularity helps" cannot be separated from "multi-scale features help",
which is not a new claim.

The baseline runs on the view-balanced sampler too. That sampler changes batch
composition, not just what the loss sees, and the measured gap has already
proven sensitive to batch size; a baseline drawn differently would vary in two
ways at once.

Read the result as (mAP, gap), never mAP alone. Twenty-five points of mAP
already failed to move the gap on this dataset - a branch that lifts mAP while
leaving the gap where it was has reproduced capacity, not granularity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..cargo import CargoDataset, CargoEvalDataset, binary_view_tensor
from ..config import CargoConfig
from ..evaluation import evaluate_rank, extract_features
from ..hierarchy.granularity import spec_for_image
from ..hierarchy.losses import MultiGranularityLoss
from ..hierarchy.sampler import ViewBalancedPKSampler
from ..losses import TripletLoss
from ..runtime import resolve_device, warmup_cosine_scheduler
from .transreid import TransReIDSmall


@dataclass
class BranchConfig:
    """One arm of the experiment."""

    name: str
    use_mg: bool = False
    coarse_bias: float = 0.0
    alpha: float = 0.88          # measured for parity with L_ID + L_triplet


BRANCHES = {
    "baseline": BranchConfig("baseline", use_mg=False),
    "uniform": BranchConfig("uniform", use_mg=True, coarse_bias=0.0),
    "view_aware": BranchConfig("view_aware", use_mg=True, coarse_bias=2.0),
}


def build_loaders(cfg: CargoConfig):
    dataset = CargoDataset(
        os.path.join(cfg.data_root, cfg.train_dir),
        cfg.image_size, cfg.norm_mean, cfg.norm_std,
        cfg.flip_probability, cfg.padding, cfg.random_erasing_probability,
    )
    sampler = ViewBalancedPKSampler(
        dataset.items, cfg.batch_size, cfg.instances_per_identity, seed=cfg.split_seed
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
                         cfg.image_size, cfg.norm_mean, cfg.norm_std), **eval_kwargs
    )
    gallery = DataLoader(
        CargoEvalDataset(os.path.join(cfg.data_root, cfg.gallery_dir),
                         cfg.image_size, cfg.norm_mean, cfg.norm_std), **eval_kwargs
    )
    return dataset, sampler, loader, query, gallery


@torch.no_grad()
def evaluate(model, cfg, query_loader, gallery_loader):
    qf, qp, qc = extract_features(model, query_loader, cfg.device, flip_tta=cfg.flip_tta)
    gf, gp, gc = extract_features(model, gallery_loader, cfg.device, flip_tta=cfg.flip_tta)
    cmc, mean_ap = evaluate_rank((1.0 - qf @ gf.t()).numpy(), qp, gp, qc, gc)
    return float(cmc[0]), float(mean_ap)


def train_branch(cfg: CargoConfig, branch: BranchConfig):
    resolve_device(cfg)
    output_dir = os.path.join(cfg.output_dir, branch.name)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(cfg.split_seed)
    dataset, sampler, loader, query_loader, gallery_loader = build_loaders(cfg)
    model = TransReIDSmall(cfg, dataset.num_classes).to(cfg.device)

    parameter_groups = model.optimizer_groups()
    mg_loss = None
    if branch.use_mg:
        spec = spec_for_image(tuple(cfg.image_size))
        mg_loss = MultiGranularityLoss(
            spec, cfg.embed_dim, coarse_bias=branch.coarse_bias
        ).to(cfg.device)
        parameter_groups.append({
            "params": mg_loss.parameters(),
            "lr": cfg.learning_rate * cfg.head_lr_multiplier,
        })

    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine_scheduler(optimizer, cfg.epochs, cfg.warmup_epochs)
    triplet = TripletLoss(margin=0.3)
    use_amp = str(cfg.device).startswith("cuda") and cfg.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    resume_path = os.path.join(output_dir, "last.pth")
    start_epoch, best_map, best_rank1, best_epoch = 1, -1.0, -1.0, -1
    if os.path.exists(resume_path):
        saved = torch.load(resume_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(saved["model"])
        if mg_loss is not None and saved.get("mg_loss"):
            mg_loss.load_state_dict(saved["mg_loss"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"] + 1
        best_map = saved.get("best_map", -1.0)
        best_rank1 = saved.get("best_rank1", -1.0)
        best_epoch = saved.get("best_epoch", -1)
        print(f"[{branch.name}] resumed at epoch {saved['epoch']}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        running, running_mg = 0.0, 0.0
        progress = tqdm(loader, desc=f"{branch.name} {epoch:03d}/{cfg.epochs}", leave=False)
        for step, (images, labels, camera_ids) in enumerate(progress, start=1):
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)
            camera_ids = camera_ids.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # One encoder pass feeds both the identity heads and L_MG.
                tokens = model._forward_tokens(images, camera_ids)
                output = model(images, camera_ids, tokens=tokens)
                ce = torch.stack([
                    F.cross_entropy(logit, labels, label_smoothing=cfg.label_smoothing)
                    for logit in output["logits"]
                ]).mean()
                tri = torch.stack([
                    triplet(feature, labels) for feature in output["features"]
                ]).mean()
                loss = ce + tri

                if mg_loss is not None:
                    views = binary_view_tensor(camera_ids)
                    granular, breakdown = mg_loss(
                        tokens.float()[:, 1:], labels, views, tokens.float()[:, 0]
                    )
                    loss = loss + branch.alpha * granular
                    running_mg += float(granular)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss)
            progress.set_postfix(loss=f"{running / step:.4f}")

        scheduler.step()
        message = f"[{branch.name}] epoch {epoch:03d} | loss {running / len(loader):.4f}"
        if mg_loss is not None:
            message += f" | L_MG {running_mg / len(loader):.4f}"
        print(message, flush=True)

        snapshot = {
            "model": model.state_dict(),
            "mg_loss": mg_loss.state_dict() if mg_loss is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch, "best_map": best_map, "best_rank1": best_rank1,
            "best_epoch": best_epoch,
            "config": {k: v for k, v in cfg.__dict__.items()},
            "branch": branch.__dict__,
            "dataset": "CARGO",
            "train_identity_count": dataset.num_classes,
        }
        torch.save(snapshot, resume_path)

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            rank1, mean_ap = evaluate(model, cfg, query_loader, gallery_loader)
            print(f"[{branch.name}] epoch {epoch:03d} | Rank-1 {rank1:.2%} "
                  f"| mAP {mean_ap:.2%}", flush=True)
            snapshot.update({"rank1": rank1, "mAP": mean_ap})
            torch.save(snapshot, os.path.join(output_dir, f"epoch_{epoch:03d}.pth"))
            if mean_ap > best_map:
                best_map, best_rank1, best_epoch = mean_ap, rank1, epoch
                torch.save(snapshot, os.path.join(output_dir, "best_model.pth"))

    print(f"[{branch.name}] best epoch {best_epoch:03d} | Rank-1 {best_rank1:.2%} "
          f"| mAP {best_map:.2%}")
    return {"branch": branch.name, "best_epoch": best_epoch,
            "rank1": best_rank1, "mAP": best_map}


def train_all(cfg: CargoConfig, names=("baseline", "uniform", "view_aware")):
    """Run the branches in one go, so no configuration drifts between them."""
    results = []
    for name in names:
        results.append(train_branch(cfg, BRANCHES[name]))
        with open(os.path.join(cfg.output_dir, "branches.json"), "w") as handle:
            json.dump(results, handle, indent=2)
    return results
