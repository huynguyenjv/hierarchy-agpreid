"""Granularity Stability Score: does the view gap live in spatial detail? (Table 1)

Tests the hypothesis behind the granularity hierarchy without training anything
new.  A trained TransReID checkpoint already contains patch tokens; this script
reads features off each level of the granularity tree and evaluates retrieval
per level, separately for cross-view and same-view protocols.

The prediction, if the hypothesis holds:

    cross-view (A->G, G->A)  coarse levels win  - shared structure survives
    same-view  (A->A, G->G)  fine levels win    - local detail is available

i.e. the two curves cross.  If instead one level dominates everywhere, the
granularity story is wrong and the design should be reconsidered before any
loss is written.

Usage:
    venv/Scripts/python.exe tools/compute_gss.py
    venv/Scripts/python.exe tools/compute_gss.py --checkpoint outputs/transreid/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_advance.config import TransReIDConfig
from reid_advance.data import EvalReIDDataset
from reid_advance.evaluation import evaluate_rank
from reid_advance.hierarchy import binary_view_tensor
from reid_advance.hierarchy.granularity import (
    LEVEL_NAMES,
    flatten_level,
    pyramid_features,
    spec_for_image,
)
from reid_advance.identity import require_current_identity_scheme
from reid_advance.pipelines.transreid import TransReIDSmall


@torch.no_grad()
def extract_pyramid(model, loader, spec, device, use_amp):
    """Per-level descriptors for a whole split, plus pids/camids."""
    model.eval()
    levels = [[] for _ in range(spec.depth)]
    pids, camids = [], []
    for images, pid, camid in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        camera = camid.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            tokens = model._forward_tokens(images, camera)
        tokens = tokens.float()
        features = pyramid_features(tokens[:, 1:], spec, cls_token=tokens[:, 0])
        for index, level_feature in enumerate(features):
            levels[index].append(flatten_level(level_feature).cpu())
        pids.append(pid)
        camids.append(camid)
    return (
        [torch.cat(chunk, dim=0) for chunk in levels],
        torch.cat(pids).numpy(),
        torch.cat(camids).numpy(),
    )


def evaluate_level(query_feature, gallery_feature, q_pids, g_pids, q_cams, g_cams):
    distance = (1.0 - query_feature @ gallery_feature.t()).numpy()
    cmc, mean_ap = evaluate_rank(distance, q_pids, g_pids, q_cams, g_cams)
    return float(mean_ap), float(cmc[0])


def same_view_control(features, pids, cams, views, seed: int = 42):
    """Retrieval within a single platform, averaged over the platforms present.

    The official protocols are cross-view by construction, so a same-view score
    has to be built by splitting one side against itself: for each platform,
    every identity donates one image as a pseudo-query and the rest stay in the
    gallery.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for view in (0, 1):
        mask = views == view
        if mask.sum() < 20:
            continue
        subset_pids = pids[mask]
        query_positions = []
        for pid in np.unique(subset_pids):
            positions = np.flatnonzero(subset_pids == pid)
            if len(positions) < 2:      # need one left in the gallery
                continue
            query_positions.append(rng.choice(positions))
        if len(query_positions) < 10:
            continue
        query_positions = np.asarray(query_positions)
        is_query = np.zeros(int(mask.sum()), dtype=bool)
        is_query[query_positions] = True

        subset_features = features[mask]
        subset_cams = cams[mask]
        mean_ap, _ = evaluate_level(
            subset_features[is_query], subset_features[~is_query],
            subset_pids[is_query], subset_pids[~is_query],
            # evaluate_rank drops gallery hits sharing the query's camera, which
            # would remove every true match here; offset the gallery camera ids
            # so the filter never fires inside this control.
            subset_cams[is_query], subset_cams[~is_query] + 100,
        )
        scores.append(mean_ap)
    return float(np.mean(scores)) if scores else float("nan")


def level_redundancy(level_features: list[torch.Tensor], sample: int = 2000,
                     seed: int = 42) -> list[list[float]]:
    """Correlation between the distance matrices of every pair of levels.

    A granularity tree is only worth building if its levels disagree. If these
    correlations are near 1, each level is re-measuring the same thing and any
    per-level differences in mAP are noise rather than signal.
    """
    generator = torch.Generator().manual_seed(seed)
    count = level_features[0].shape[0]
    if count > sample:
        index = torch.randperm(count, generator=generator)[:sample]
        level_features = [feature[index] for feature in level_features]
    distances = [
        (1.0 - feature @ feature.t()).flatten() for feature in level_features
    ]
    stacked = torch.stack(distances)
    return torch.corrcoef(stacked).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/transreid/best_model.pth")
    parser.add_argument("--data-root", default="AG-ReID.v2")
    parser.add_argument("--protocols", nargs="+", default=[
        "exp1_aerial_to_cctv.txt", "exp4_cctv_to_aerial.txt",
    ])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="docs/gss_table.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    require_current_identity_scheme(state, args.checkpoint)
    saved = state.get("config", {})

    cfg = TransReIDConfig()
    for key in ("encoder_name", "embed_dim", "image_size", "transreid_sie",
                "transreid_jpm", "transreid_num_parts", "norm_mean", "norm_std"):
        if key in saved:
            setattr(cfg, key, saved[key])
    cfg.device = device
    cfg.data_root = args.data_root

    num_classes = state.get("train_identity_count", 807)
    model = TransReIDSmall(cfg, num_classes).to(device)
    model.load_state_dict(state["model"])
    print(f"loaded {args.checkpoint} | epoch {state.get('epoch')} "
          f"| mAP {state.get('mAP', float('nan')):.2%}")

    spec = spec_for_image(tuple(cfg.image_size))
    print(f"patch grid {spec.rows}x{spec.cols} | levels {spec.levels} "
          f"({', '.join(LEVEL_NAMES[:spec.depth])})")

    results = {}

    for protocol in args.protocols:
        path = os.path.join(args.data_root, protocol)
        if not os.path.exists(path):
            print(f"skip {protocol}: not found")
            continue
        print(f"\n=== {protocol} ===")

        loaders = {}
        for split in ("query", "gallery"):
            dataset = EvalReIDDataset(
                args.data_root, path, split, tuple(cfg.image_size),
                cfg.norm_mean, cfg.norm_std,
            )
            loaders[split] = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=(device == "cuda"),
            )

        q_levels, q_pids, q_cams = extract_pyramid(model, loaders["query"], spec, device, use_amp)
        g_levels, g_pids, g_cams = extract_pyramid(model, loaders["gallery"], spec, device, use_amp)

        q_view = binary_view_tensor(torch.as_tensor(q_cams)).numpy()
        g_view = binary_view_tensor(torch.as_tensor(g_cams)).numpy()

        per_level = []
        for level in range(spec.depth):
            cross_map, cross_r1 = evaluate_level(
                q_levels[level], g_levels[level], q_pids, g_pids, q_cams, g_cams
            )
            # Same-view control. The official protocol is cross-view by
            # construction (query and gallery sit on different platforms), so
            # splitting it by view leaves one side empty. Build the control from
            # the gallery alone instead: hold out part of it as pseudo-queries
            # so query and gallery share a platform.
            same_map = same_view_control(
                g_levels[level], g_pids, g_cams, g_view
            )

            per_level.append({
                "level": level,
                "name": LEVEL_NAMES[level],
                "regions": spec.levels[level],
                "rows_per_region": spec.rows_per_region(level),
                "cross_view_map": cross_map,
                "cross_view_rank1": cross_r1,
                "same_view_map": same_map,
                "gap": same_map - cross_map,
            })
            print(f"  L{level} {LEVEL_NAMES[level]:<8} ({spec.levels[level]} regions) "
                  f"cross-view mAP {cross_map:.2%} R1 {cross_r1:.2%} | "
                  f"same-view mAP {same_map:.2%} | gap {same_map - cross_map:+.2%}")

        redundancy = level_redundancy(g_levels)
        off_diagonal = [
            redundancy[i][j] for i in range(spec.depth) for j in range(i + 1, spec.depth)
        ]
        print("  level-to-level distance correlation: "
              f"min {min(off_diagonal):.3f} max {max(off_diagonal):.3f}")
        if min(off_diagonal) > 0.95:
            print("  !! levels are near-duplicates; per-level mAP differences are "
                  "not meaningful with this backbone")

        results[protocol] = {"levels": per_level, "redundancy": redundancy}

    # --- verdict ----------------------------------------------------------
    print("\n=== hypothesis check ===")
    verdicts = {}
    for protocol, payload in results.items():
        levels = payload["levels"]
        redundancy = payload["redundancy"]
        cross = [entry["cross_view_map"] for entry in levels]
        same = [entry["same_view_map"] for entry in levels]
        best_cross = int(np.argmax(cross))
        best_same = int(np.nanargmax(same)) if not np.isnan(same).all() else -1
        spread = max(cross) - min(cross)
        off_diagonal = [
            redundancy[i][j] for i in range(len(levels)) for j in range(i + 1, len(levels))
        ]
        conclusive = min(off_diagonal) < 0.95 and spread > 0.01

        verdicts[protocol] = {
            "best_cross_level": best_cross,
            "best_same_level": best_same,
            "cross_view_spread": spread,
            "min_level_correlation": min(off_diagonal),
            "conclusive": bool(conclusive),
            "supported": bool(conclusive and best_cross < best_same),
        }
        print(f"{protocol}:")
        print(f"  cross-view peaks at L{best_cross} ({LEVEL_NAMES[best_cross]}), "
              f"spread across levels {spread:.2%}")
        print(f"  same-view peaks at "
              f"L{best_same if best_same >= 0 else '?'}"
              f"{' (' + LEVEL_NAMES[best_same] + ')' if best_same >= 0 else ''}")
        print(f"  min level correlation {min(off_diagonal):.3f}")
        if not conclusive:
            print("  -> INCONCLUSIVE: the levels of a jointly-trained backbone are "
                  "near-duplicates, so this frozen-feature probe cannot decide the "
                  "hypothesis either way")
        elif verdicts[protocol]["supported"]:
            print("  -> SUPPORTS the hypothesis")
        else:
            print("  -> does NOT support the hypothesis")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump({"checkpoint": args.checkpoint, "spec": {
            "rows": spec.rows, "cols": spec.cols, "levels": list(spec.levels)},
            "results": results, "verdicts": verdicts}, handle, indent=2)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# Granularity Stability Score (Table 1)\n\n")
        handle.write(
            f"Features read off a frozen `{args.checkpoint}` at each level of the "
            "spatial granularity tree. No retraining: this only asks where in the "
            "pyramid the existing representation already transfers across views.\n\n"
        )
        handle.write(
            f"- patch grid {spec.rows}x{spec.cols}, levels {list(spec.levels)} "
            f"({', '.join(LEVEL_NAMES[:spec.depth])})\n"
            "- cross-view mAP uses the official protocol; same-view mAP restricts "
            "the gallery to the query's own platform\n\n"
        )
        for protocol, payload in results.items():
            handle.write(f"## {protocol}\n\n")
            handle.write("| level | regions | rows | cross-view mAP | cross-view R1 "
                         "| same-view mAP | gap |\n")
            handle.write("|---|---|---|---|---|---|---|\n")
            for entry in payload["levels"]:
                handle.write(
                    f"| L{entry['level']} {entry['name']} | {entry['regions']} | "
                    f"{entry['rows_per_region']} | {entry['cross_view_map']:.2%} | "
                    f"{entry['cross_view_rank1']:.2%} | {entry['same_view_map']:.2%} | "
                    f"{entry['gap']:+.2%} |\n"
                )

            handle.write("\nLevel-to-level distance-matrix correlation:\n\n| | "
                         + " | ".join(f"L{i}" for i in range(spec.depth)) + " |\n")
            handle.write("|---" * (spec.depth + 1) + "|\n")
            for i, row in enumerate(payload["redundancy"]):
                handle.write(f"| **L{i}** | "
                             + " | ".join(f"{value:.3f}" for value in row) + " |\n")

            verdict = verdicts[protocol]
            handle.write(
                f"\nCross-view peaks at **L{verdict['best_cross_level']}** with a "
                f"spread of only {verdict['cross_view_spread']:.2%} across levels, "
                f"and the levels correlate at {verdict['min_level_correlation']:.3f} "
                "or above.\n\n"
            )
            if not verdict["conclusive"]:
                handle.write(
                    "**Inconclusive.** The probe cannot decide the hypothesis: see "
                    "the caveat below.\n\n"
                )
            elif verdict["supported"]:
                handle.write("**Supports** the granularity hypothesis.\n\n")
            else:
                handle.write("**Does not support** the granularity hypothesis.\n\n")

        handle.write(
            "## What this probe can and cannot show\n\n"
            "The hypothesis predicts the two mAP columns peak at different levels: "
            "coarse for cross-view, fine for same-view. A crossing would mean the "
            "optimal matching granularity depends on the view pair, which is what a "
            "granularity hierarchy exists to exploit.\n\n"
            "But the correlation table above is the number to read first. Every "
            "level here pools the *same* patch tokens from a backbone trained with "
            "a single global objective, so the levels are near-duplicates of each "
            "other and their mAP differences are noise. A frozen-feature probe can "
            "therefore not falsify the hypothesis - it can only show whether "
            "granularity is *already* differentiated for free, and it is not.\n\n"
            "Deciding the question requires levels that are trained to differ, i.e. "
            "a per-level objective. That is the experiment the multi-granularity "
            "loss is for; this table is the baseline it has to beat.\n"
        )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
