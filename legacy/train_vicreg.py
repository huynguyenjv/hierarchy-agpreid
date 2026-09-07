import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from attributes import ATTRIBUTE_NAMES
from config import Config
from dataset import (
    EvalReIDDataset,
    HybridReIDDataset,
    RandomIdentitySampler,
    SupervisedReIDDataset,
)
from model import SupervisedBaseline, HybridSSLModel
from loss import vicreg_loss, masked_attribute_loss, TripletLoss
from evaluate import extract_features, evaluate_rank 


def run_evaluation(model, cfg, q_loader, g_loader):
    print("\n--- Đang tiến hành Evaluate ---")
    qf, q_pids, q_camids = extract_features(model, q_loader, cfg.device, cfg.flip_tta)
    gf, g_pids, g_camids = extract_features(model, g_loader, cfg.device, cfg.flip_tta)

    distmat = (1.0 - qf @ gf.t()).numpy()
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    
    return cmc[0], mAP 


def _split_hybrid_by_track(dataset, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    track_to_indices = {}
    for index, (path, _) in enumerate(dataset.items):
        track_to_indices.setdefault(os.path.dirname(path), []).append(index)
    tracks = sorted(track_to_indices)
    random.Random(seed).shuffle(tracks)
    num_val_tracks = max(1, round(len(tracks) * validation_fraction))
    val_tracks = set(tracks[:num_val_tracks])
    train_indices, val_indices = [], []
    for track, indices in track_to_indices.items():
        (val_indices if track in val_tracks else train_indices).extend(indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _loader(dataset, cfg, shuffle, drop_last):
    kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=str(cfg.device).startswith("cuda"),
    )
    if cfg.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def _build_scheduler(optimizer, cfg):
    warmup_epochs = min(max(0, cfg.warmup_epochs), max(0, cfg.epochs - 1))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs - warmup_epochs), eta_min=1e-6
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])


def _hybrid_loss(model, batch, cfg, device):
    v1, v2, targets = (value.to(device, non_blocking=True) for value in batch)
    z1, z2, _, _, logits1, logits2 = model(v1, v2)
    loss_ssl, components = vicreg_loss(
        z1,
        z2,
        sim_coeff=cfg.ssl_sim_coeff,
        var_coeff=cfg.ssl_var_coeff,
        cov_coeff=cfg.ssl_cov_coeff,
        return_components=True,
    )
    loss_attr, valid_labels = masked_attribute_loss(logits1, logits2, targets)
    loss = loss_ssl + cfg.attr_loss_weight * loss_attr
    return loss, loss_ssl, loss_attr, components, valid_labels

def train_supervised(cfg: Config, q_loader, g_loader):
    print(">>> RUNNING SCENARIO: Supervised Baseline (Upper Bound)")
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb", "supervised"))
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    dataset = SupervisedReIDDataset(cfg)
    cfg.num_classes = dataset.num_classes
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
    )
    
    model = SupervisedBaseline(cfg).to(cfg.device)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_tri = TripletLoss(margin=0.3)

    optimizer = torch.optim.AdamW(
        model.get_optim_params(cfg.supervised_base_lr), weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_map = 0.0
    patience_cnt = 0
    patience_limit = 20

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        total_loss, ce_loss_val, tri_loss_val = 0, 0, 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{cfg.epochs} [Train]", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=use_amp):
                features, logits = model(imgs)
                loss_ce = criterion_ce(logits, labels)
                loss_tri = criterion_tri(features, labels)
                loss = loss_ce + loss_tri
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item(); ce_loss_val += loss_ce.item(); tri_loss_val += loss_tri.item()
            pbar.set_postfix({'Total': f"{loss.item():.3f}", 'CE': f"{loss_ce.item():.3f}", 'Tri': f"{loss_tri.item():.3f}", 'LR': f"{scheduler.get_last_lr()[0]:.1e}"})
            
        scheduler.step()
        
        # Ghi log TensorBoard
        writer.add_scalar("Loss/Total", total_loss/len(loader), epoch)
        writer.add_scalar("Loss/CE", ce_loss_val/len(loader), epoch)
        writer.add_scalar("Loss/Triplet", tri_loss_val/len(loader), epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch:03d} | Train Total: {total_loss/len(loader):.4f}")
        if epoch % cfg.eval_interval == 0:
            rank1, map_score = run_evaluation(model, cfg, q_loader, g_loader)
            writer.add_scalar("Metric/Rank1", rank1, epoch)
            writer.add_scalar("Metric/mAP", map_score, epoch)
            print(f"         | Rank-1: {rank1:.2%} | mAP: {map_score:.2%}")
            
            if map_score > best_map:
                best_map = map_score
                patience_cnt = 0
                ckpt_path = os.path.join(cfg.output_dir, "supervised_best_model.pth")
                torch.save({"epoch": epoch, "model": model.state_dict(), "mAP": map_score, "config": cfg.__dict__}, ckpt_path)
                print(f"  -> Lưu Checkpoint MỚI (Best mAP: {best_map:.2%}) tại {ckpt_path}")
            else:
                patience_cnt += 1
                if patience_cnt >= patience_limit:
                    print(f"Early Stopping! mAP không tăng trong {patience_limit} epochs.")
                    break

def train_hybrid(cfg: Config, q_loader, g_loader):
    print(">>> RUNNING SCENARIO: VICReg + Attribute Guidance")
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb", "hybrid_vicreg"))
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    dataset = HybridReIDDataset(cfg)
    train_set, val_set = _split_hybrid_by_track(
        dataset, cfg.validation_fraction, cfg.split_seed
    )
    loader = _loader(train_set, cfg, shuffle=True, drop_last=True)
    # VICReg covariance and projector BatchNorm both require more than one item.
    val_loader = _loader(val_set, cfg, shuffle=False, drop_last=True)
    print(
        f"Attribute coverage: {dataset.attribute_match_rate:.2%} | "
        f"train images: {len(train_set):,} | validation images: {len(val_set):,}"
    )
    
    model = HybridSSLModel(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.get_optim_params(cfg.base_lr), weight_decay=cfg.weight_decay
    )
    scheduler = _build_scheduler(optimizer, cfg)
    use_amp = str(cfg.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    patience_cnt = 0
    patience_limit = 20
    ckpt_path = os.path.join(cfg.output_dir, cfg.hybrid_ckpt_name)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "ssl": 0.0, "attr": 0.0, "sim": 0.0, "var": 0.0, "cov": 0.0}
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{cfg.epochs} [Train]", leave=False)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, loss_ssl, loss_attr, components, valid_labels = _hybrid_loss(
                    model, batch, cfg, cfg.device
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += loss.item()
            totals["ssl"] += loss_ssl.item()
            totals["attr"] += loss_attr.item()
            for name in ("sim", "var", "cov"):
                totals[name] += components[name].item()
            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "SSL": f"{loss_ssl.item():.3f}",
                "Attr": f"{loss_attr.item():.3f}",
                "ValidAttr": valid_labels,
                "LR": f"{scheduler.get_last_lr()[0]:.1e}",
            })
        scheduler.step()

        for name, value in totals.items():
            writer.add_scalar(f"Train/{name}", value / len(loader), epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    val_loss, _, _, _, _ = _hybrid_loss(model, batch, cfg, cfg.device)
                val_total += val_loss.item()
        val_loss = val_total / len(val_loader)
        writer.add_scalar("Validation/loss", val_loss, epoch)
        print(
            f"Epoch {epoch:03d} | Train: {totals['loss']/len(loader):.4f} | "
            f"Validation: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "validation_loss": val_loss,
                    "config": cfg.__dict__,
                    "attribute_names": ATTRIBUTE_NAMES,
                },
                ckpt_path,
            )
            print(f"  -> Saved best validation checkpoint: {ckpt_path}")
        else:
            patience_cnt += 1

        if cfg.monitor_test_during_training and epoch % cfg.eval_interval == 0:
            rank1, map_score = run_evaluation(model, cfg, q_loader, g_loader)
            writer.add_scalar("TestMonitor/Rank1", rank1, epoch)
            writer.add_scalar("TestMonitor/mAP", map_score, epoch)
            print(f"         | Rank-1: {rank1:.2%} | mAP: {map_score:.2%}")

        if patience_cnt >= patience_limit:
            print(f"Early stopping: validation loss did not improve for {patience_limit} epochs.")
            break

    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state["model"])
    rank1, map_score = run_evaluation(model, cfg, q_loader, g_loader)
    print(f"Final official evaluation | Rank-1: {rank1:.2%} | mAP: {map_score:.2%}")
    writer.add_hparams(
        {"attr_weight": cfg.attr_loss_weight, "base_lr": cfg.base_lr},
        {"hparam/final_rank1": rank1, "hparam/final_mAP": map_score},
    )
    writer.close()

def main():
    print("Choose one independent entrypoint:")
    print("  python train_bnneck.py")
    print("  python train_transreid.py")
    print("  python train_proposed.py")

if __name__ == "__main__":
    main()
