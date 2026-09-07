"""AI-03: compute the View Stability Score for the 15 soft attributes.

Trains one ViT-S with 15 attribute heads on *ground* images of one set of
identities, then evaluates on held-out identities' ground and aerial images.
The aerial score is the VSS, and the ground-to-aerial drop is what decides
which attributes can sit near the root of the semantic tree.

Usage:
    venv/Scripts/python.exe tools/compute_vss.py
    venv/Scripts/python.exe tools/compute_vss.py --epochs 15 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_advance.attributes import ATTRIBUTE_IGNORE_INDEX, ATTRIBUTE_NAMES
from reid_advance.data import _scan_images, load_track_attributes, track_key
from reid_advance.hierarchy import platform_of_path
from reid_advance.hierarchy.vss import (
    VSS_THRESHOLD,
    AttributeHeads,
    attribute_ce_loss,
    score_attribute,
)
from reid_advance.identity import FILENAME_PATTERN


class AttributeDataset(Dataset):
    """Images of one platform group, labelled with identity-level attributes."""

    def __init__(self, items, attributes, image_size, train: bool):
        self.items = items
        self.attributes = attributes
        if train:
            self.transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(0.5),
                T.Pad(10),
                T.RandomCrop(image_size),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, _ = self.items[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, self.attributes[track_key(path)]


def build_splits(data_root: str, mat_path: str, test_fraction: float, seed: int):
    """Identity-disjoint split; attributes live at identity level so this matters.

    If an identity appeared on both sides, the model could recognise the person
    and recall their attributes rather than reading them off the image, and VSS
    would measure re-identification instead of attribute legibility.
    """
    items = _scan_images(os.path.join(data_root, "train_all"), FILENAME_PATTERN)
    attributes = load_track_attributes(mat_path, ignore_unknown=True)

    identities = sorted({pid for _, pid in items})
    rng = random.Random(seed)
    rng.shuffle(identities)
    cut = int(len(identities) * (1.0 - test_fraction))
    train_ids = set(identities[:cut])
    test_ids = set(identities[cut:])

    train_ground, test_ground, test_aerial = [], [], []
    for path, pid in items:
        if track_key(path) not in attributes:
            continue
        aerial = platform_of_path(path) == "aerial"
        if pid in train_ids:
            if not aerial:
                train_ground.append((path, pid))
        elif aerial:
            test_aerial.append((path, pid))
        else:
            test_ground.append((path, pid))

    return {
        "attributes": attributes,
        "train_ground": train_ground,
        "test_ground": test_ground,
        "test_aerial": test_aerial,
        "train_identities": len(train_ids),
        "test_identities": len(test_ids),
    }


class AttributeModel(nn.Module):
    def __init__(self, encoder_name: str, image_size):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, num_classes=0, img_size=image_size
        )
        self.heads = AttributeHeads(self.encoder.num_features)

    def forward(self, images):
        return self.heads(self.encoder(images))


@torch.no_grad()
def predict(model, loader, device, use_amp: bool):
    """Return per-attribute predictions and targets over a whole loader."""
    model.eval()
    predictions = {name: [] for name in ATTRIBUTE_NAMES}
    targets = {name: [] for name in ATTRIBUTE_NAMES}
    for images, attribute_targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        attribute_targets = attribute_targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        for index, name in enumerate(ATTRIBUTE_NAMES):
            target = attribute_targets[:, index]
            valid = target != ATTRIBUTE_IGNORE_INDEX
            if not bool(valid.any()):
                continue
            predictions[name].append(logits[name][valid].float().argmax(1).cpu().numpy())
            targets[name].append(target[valid].cpu().numpy())
    return (
        {k: (np.concatenate(v) if v else np.array([], dtype=np.int64))
         for k, v in predictions.items()},
        {k: (np.concatenate(v) if v else np.array([], dtype=np.int64))
         for k, v in targets.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="AG-ReID.v2")
    parser.add_argument("--mat", default="AG-ReID.v2/qut_attribute_v8.mat")
    parser.add_argument("--encoder", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="docs/vss_table.md")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    image_size = (256, 128)

    splits = build_splits(args.data_root, args.mat, args.test_fraction, args.seed)
    print(
        f"identities: {splits['train_identities']} train / "
        f"{splits['test_identities']} test (disjoint)\n"
        f"images: {len(splits['train_ground']):,} train-ground | "
        f"{len(splits['test_ground']):,} test-ground | "
        f"{len(splits['test_aerial']):,} test-aerial"
    )

    attributes = splits["attributes"]
    loaders = {
        "train": DataLoader(
            AttributeDataset(splits["train_ground"], attributes, image_size, True),
            batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
        ),
        "test_ground": DataLoader(
            AttributeDataset(splits["test_ground"], attributes, image_size, False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
        ),
        "test_aerial": DataLoader(
            AttributeDataset(splits["test_aerial"], attributes, image_size, False),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
        ),
    }

    model = AttributeModel(args.encoder, image_size).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.heads.parameters(),
             "lr": args.lr * args.head_lr_multiplier},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        progress = tqdm(loaders["train"], desc=f"VSS {epoch:02d}/{args.epochs}", leave=False)
        for step, (images, targets) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _ = attribute_ce_loss(model(images), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss)
            progress.set_postfix(loss=f"{running / step:.4f}")
        scheduler.step()
        print(f"epoch {epoch:02d} | loss {running / max(len(loaders['train']), 1):.4f}")

    ground_predictions, ground_targets = predict(model, loaders["test_ground"], device, use_amp)
    aerial_predictions, aerial_targets = predict(model, loaders["test_aerial"], device, use_amp)

    scores = [
        score_attribute(
            name,
            ground_predictions[name], ground_targets[name],
            aerial_predictions[name], aerial_targets[name],
        )
        for name in ATTRIBUTE_NAMES
    ]
    scores.sort(key=lambda s: s.vss, reverse=True)

    print(f"\n{'attribute':<12}{'VSS':>7}{'grnd-bal':>10}{'retain':>8}"
          f"{'aer-raw':>9}{'major':>8}{'lift':>8}  usable")
    for score in scores:
        usable = "yes" if score.vss >= VSS_THRESHOLD else "no"
        print(f"{score.name:<12}{score.vss:>7.3f}{score.ground_balanced_accuracy:>10.3f}"
              f"{score.retention:>8.2f}{score.aerial_accuracy:>9.3f}"
              f"{score.majority_baseline:>8.3f}{score.lift_over_majority:>+8.3f}  {usable}")

    payload = {
        "config": vars(args),
        "splits": {k: v for k, v in splits.items() if isinstance(v, int)},
        "images": {
            "train_ground": len(splits["train_ground"]),
            "test_ground": len(splits["test_ground"]),
            "test_aerial": len(splits["test_aerial"]),
        },
        "scores": [
            {
                "name": s.name, "vss": s.vss,
                "ground_balanced_accuracy": s.ground_balanced_accuracy,
                "aerial_balanced_accuracy": s.aerial_balanced_accuracy,
                "ground_accuracy": s.ground_accuracy,
                "aerial_accuracy": s.aerial_accuracy,
                "retention": s.retention,
                "majority_baseline": s.majority_baseline,
                "lift_over_majority": s.lift_over_majority,
                "support_ground": s.support_ground,
                "support_aerial": s.support_aerial,
                "per_class_aerial": s.per_class_aerial,
            }
            for s in scores
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# AI-03 - View Stability Score (Table 1)\n\n")
        handle.write(
            f"One `{args.encoder}` backbone with 15 attribute heads, trained for "
            f"{args.epochs} epochs on **ground images only**, evaluated on "
            "**identity-disjoint** held-out images.\n\n"
        )
        handle.write(
            f"- identities: {splits['train_identities']} train / "
            f"{splits['test_identities']} test, no overlap\n"
            f"- images: {len(splits['train_ground']):,} train-ground, "
            f"{len(splits['test_ground']):,} test-ground, "
            f"{len(splits['test_aerial']):,} test-aerial\n"
            f"- VSS = balanced accuracy on aerial images (threshold {VSS_THRESHOLD})\n\n"
        )
        handle.write(
            "Balanced accuracy is used instead of raw accuracy because several "
            "groups are dominated by one class; raw accuracy would reward a model "
            "that always predicts the majority. `lift` shows aerial raw accuracy "
            "minus that majority baseline.\n\n"
        )
        handle.write("| attribute | VSS (aerial bal.) | ground bal. | retention | "
                     "aerial raw | majority | lift | usable as level |\n")
        handle.write("|---|---|---|---|---|---|---|---|\n")
        for score in scores:
            handle.write(
                f"| `{score.name}` | **{score.vss:.3f}** | "
                f"{score.ground_balanced_accuracy:.3f} | {score.retention:.2f} | "
                f"{score.aerial_accuracy:.3f} | {score.majority_baseline:.3f} | "
                f"{score.lift_over_majority:+.3f} | "
                f"{'yes' if score.vss >= VSS_THRESHOLD else 'no'} |\n"
            )
        handle.write(
            "\n**Reading the table.** `retention` is aerial divided by ground "
            "balanced accuracy: it isolates how much of the signal the climb "
            "destroys, independently of how hard the attribute is in the first "
            "place. An attribute belongs near the root when both VSS and "
            "retention are high (KB 6.2).\n"
        )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
