"""Shared experiment runtime helpers."""

from __future__ import annotations

import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .data import EvalReIDDataset
from .evaluation import evaluate_rank, extract_features
from .identity import OFFICIAL_PROTOCOL_IDENTITY_COUNTS, validate_identity_count


def resolve_device(cfg):
    if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        cfg.device = "cpu"
    return cfg.device


def build_eval_loaders(cfg):
    protocol = os.path.join(cfg.data_root, cfg.eval_txt_file)
    query = EvalReIDDataset(
        cfg.data_root, protocol, "query", cfg.image_size, cfg.norm_mean, cfg.norm_std
    )
    gallery = EvalReIDDataset(
        cfg.data_root, protocol, "gallery", cfg.image_size, cfg.norm_mean, cfg.norm_std
    )
    query_pids = {pid for _, pid, _ in query.items}
    gallery_pids = {pid for _, pid, _ in gallery.items}
    if cfg.strict_dataset_integrity:
        expected = cfg.expected_protocol_identities
        if expected is None:
            expected = OFFICIAL_PROTOCOL_IDENTITY_COUNTS.get(
                os.path.basename(cfg.eval_txt_file)
            )
        validate_identity_count(
            len(query_pids), expected, "protocol query split"
        )
        validate_identity_count(
            len(gallery_pids), expected, "protocol gallery split"
        )
        if query_pids != gallery_pids:
            raise RuntimeError(
                "AG-ReID.v2 protocol query/gallery identity sets do not match"
            )
    print(
        f"Dataset audit | train identity scheme=P+T+A | "
        f"query={len(query):,}/{len(query_pids)} IDs | "
        f"gallery={len(gallery):,}/{len(gallery_pids)} IDs"
    )
    eval_bs = getattr(cfg, "eval_batch_size", cfg.batch_size)
    kwargs = {
        "batch_size": eval_bs,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": str(cfg.device).startswith("cuda"),
    }
    return DataLoader(query, **kwargs), DataLoader(gallery, **kwargs)


def evaluate_model(model, cfg, query_loader, gallery_loader):
    query_features, query_pids, query_cams = extract_features(
        model, query_loader, cfg.device, cfg.flip_tta
    )
    gallery_features, gallery_pids, gallery_cams = extract_features(
        model, gallery_loader, cfg.device, cfg.flip_tta
    )
    distance = (1.0 - query_features @ gallery_features.t()).numpy()
    cmc, mean_ap = evaluate_rank(
        distance, query_pids, gallery_pids, query_cams, gallery_cams
    )
    return cmc[0], mean_ap


def warmup_cosine_scheduler(optimizer, epochs, warmup_epochs, eta_min=1e-6):
    warmup_epochs = min(max(0, warmup_epochs), max(0, epochs - 1))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=eta_min
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
