"""AI-00: measure peak VRAM of the TransReID-S trainer across batch sizes.

The hierarchy branch needs the largest batch that fits, because CV-HWC is a
contrastive loss whose negatives all come from inside a single batch (gradient
accumulation cannot substitute for it).  The chosen batch in turn caps how deep
the semantic tree can be, so this runs before the tree is designed.

Usage:
    venv/Scripts/python.exe tools/profile_memory.py
    venv/Scripts/python.exe tools/profile_memory.py --batch-sizes 64 128 --no-jpm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_advance.config import TransReIDConfig
from reid_advance.losses import TripletLoss
from reid_advance.pipelines.transreid import TransReIDSmall


GIB = 1024 ** 3


def profile(batch_size: int, instances: int, jpm: bool, use_amp: bool,
            num_classes: int, steps: int, device: str) -> dict:
    """Run a few real train steps and report peak allocated/reserved memory."""
    cfg = TransReIDConfig()
    cfg.device = device
    cfg.batch_size = batch_size
    cfg.instances_per_identity = instances
    cfg.transreid_jpm = jpm
    cfg.use_amp = use_amp

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = TransReIDSmall(cfg, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.optimizer_groups(), weight_decay=cfg.weight_decay)
    triplet = TripletLoss(margin=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    height, width = cfg.image_size
    identities = max(1, batch_size // instances)
    # Labels mirror the P x K layout so the triplet loss finds real positives.
    labels = torch.arange(identities, device=device).repeat_interleave(instances)[:batch_size]
    # Cameras 0/2/3 are aerial/wearable/CCTV; alternate so SIE sees both views.
    camera_ids = torch.tensor([0, 3], device=device).repeat(batch_size // 2 + 1)[:batch_size]

    model.train()
    for _ in range(steps):
        images = torch.randn(batch_size, 3, height, width, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(images, camera_ids)
            ce = torch.stack([
                F.cross_entropy(logit, labels, label_smoothing=cfg.label_smoothing)
                for logit in output["logits"]
            ]).mean()
            tri = torch.stack([
                triplet(feature, labels) for feature in output["features"]
            ]).mean()
            loss = ce + tri
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    peak_alloc = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "batch_size": batch_size,
        "instances": instances,
        "identities": identities,
        "jpm": jpm,
        "amp": use_amp,
        "peak_allocated_gib": round(peak_alloc, 2),
        "peak_reserved_gib": round(peak_reserved, 2),
    }


def estimate_cvhwc_overhead(batch_size: int, levels: int) -> float:
    """Extra memory CV-HWC needs on top of the backbone, in GiB.

    The loss builds several (B, B) float matrices (similarity, weights, beta,
    log_prob and autograd's saved copies) plus one (B, B, L) match tensor.
    """
    pairwise = batch_size * batch_size * 4  # bytes, fp32
    match = batch_size * batch_size * levels * 4
    # ~8 live (B, B) tensors between forward and backward is a realistic upper
    # bound for the reference implementation in the knowledge base.
    return (pairwise * 8 + match) / GIB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=807)
    parser.add_argument("--no-jpm", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tree-levels", type=int, default=4)
    parser.add_argument("--output", default="docs/memory_profile.md")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. Run with venv/Scripts/python.exe (the global "
            "interpreter has a CPU-only torch build)."
        )

    device = "cuda"
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | {total:.2f} GiB | torch {torch.__version__}\n")

    jpm_modes = [True] if args.no_jpm is False else [False]
    if not args.no_jpm:
        jpm_modes = [True, False]
    amp_modes = [False] if args.no_amp else [True]

    rows = []
    for use_amp in amp_modes:
        for jpm in jpm_modes:
            for batch_size in args.batch_sizes:
                if batch_size % args.instances != 0:
                    print(f"skip batch={batch_size}: not divisible by K={args.instances}")
                    continue
                label = f"batch={batch_size:>4} K={args.instances} jpm={int(jpm)} amp={int(use_amp)}"
                try:
                    row = profile(
                        batch_size, args.instances, jpm, use_amp,
                        args.num_classes, args.steps, device,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    print(f"{label} -> OOM")
                    rows.append({
                        "batch_size": batch_size, "instances": args.instances,
                        "identities": batch_size // args.instances,
                        "jpm": jpm, "amp": use_amp,
                        "peak_allocated_gib": None, "peak_reserved_gib": None,
                    })
                    continue
                overhead = estimate_cvhwc_overhead(batch_size, args.tree_levels)
                row["cvhwc_overhead_gib"] = round(overhead, 3)
                row["projected_total_gib"] = round(row["peak_reserved_gib"] + overhead, 2)
                row["headroom_gib"] = round(total - row["projected_total_gib"], 2)
                rows.append(row)
                print(
                    f"{label} -> alloc {row['peak_allocated_gib']:.2f} | "
                    f"reserved {row['peak_reserved_gib']:.2f} | "
                    f"+CV-HWC {overhead:.2f} = {row['projected_total_gib']:.2f} GiB "
                    f"(headroom {row['headroom_gib']:.2f})"
                )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# AI-00 memory profile\n\n")
        handle.write(f"- GPU: {name} ({total:.2f} GiB)\n")
        handle.write(f"- torch {torch.__version__}\n")
        handle.write(f"- backbone: {TransReIDConfig.encoder_name} (ViT-S/16)\n")
        handle.write(f"- image size: {TransReIDConfig.image_size}\n")
        handle.write(f"- train identities assumed: {args.num_classes}\n")
        handle.write(f"- CV-HWC overhead projected for L={args.tree_levels}\n\n")
        handle.write(
            "| batch | P | K | JPM | AMP | peak alloc | peak reserved | "
            "+CV-HWC | projected | headroom |\n"
        )
        handle.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            if row["peak_allocated_gib"] is None:
                handle.write(
                    f"| {row['batch_size']} | {row['identities']} | {row['instances']} | "
                    f"{int(row['jpm'])} | {int(row['amp'])} | **OOM** | — | — | — | — |\n"
                )
                continue
            handle.write(
                f"| {row['batch_size']} | {row['identities']} | {row['instances']} | "
                f"{int(row['jpm'])} | {int(row['amp'])} | "
                f"{row['peak_allocated_gib']:.2f} | {row['peak_reserved_gib']:.2f} | "
                f"{row['cvhwc_overhead_gib']:.3f} | {row['projected_total_gib']:.2f} | "
                f"{row['headroom_gib']:.2f} |\n"
            )
        handle.write(
            "\n`projected` adds the estimated CV-HWC pairwise-matrix cost to the "
            "measured backbone peak. Pick the largest batch whose projected total "
            "leaves ~1 GiB of headroom.\n"
        )

    with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump({"gpu": name, "total_gib": round(total, 2), "rows": rows}, handle, indent=2)

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
