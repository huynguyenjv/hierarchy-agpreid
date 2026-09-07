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
    """Retrieval within one platform, kept cross-camera to stay comparable.

    The official protocols are cross-view by construction - query and gallery
    always sit on different platforms - so there is no ready-made same-view
    ground truth. It has to be built by splitting one side against itself, and
    exactly how that split is done decides whether the number means anything.

    Two constraints make it comparable to the cross-view mAP it is contrasted
    with:

    1. **Cross-camera is preserved.** ``evaluate_rank`` drops gallery hits that
       share the query's camera, because retrieving another shot from the same
       camera is far easier than true re-identification. A same-view control
       that disabled that filter would be measuring an easier task and would
       inflate the gap. Only identities appearing on at least two cameras of the
       same platform are used, the query is drawn from one camera, and the
       gallery keeps only that identity's *other* cameras.
    2. **One query per identity**, so no identity dominates the average.

    Ground spans two cameras (wearable C2, CCTV C3) so a ground same-view score
    is well defined. Aerial is a single camera (C0) and is skipped: it has no
    cross-camera same-view pair at all.

    Note this cannot be computed from one protocol. Each split of each official
    protocol contains exactly one camera (exp1 gallery is all C3, exp4 gallery
    all C0, ...), so the caller must pool images across protocols before calling
    this; 245 test identities appear on both C2 and C3 once pooled.
    """
    rng = np.random.default_rng(seed)
    scores = []
    details = {}
    for view in (0, 1):
        mask = views == view
        if mask.sum() < 20:
            continue
        subset_pids = pids[mask]
        subset_cams = cams[mask]
        subset_features = features[mask]

        query_index, gallery_index = [], []
        for pid in np.unique(subset_pids):
            positions = np.flatnonzero(subset_pids == pid)
            cameras = np.unique(subset_cams[positions])
            if len(cameras) < 2:
                continue                      # no cross-camera pair possible
            query_camera = rng.choice(cameras)
            candidates = positions[subset_cams[positions] == query_camera]
            query_index.append(rng.choice(candidates))
            gallery_index.extend(positions[subset_cams[positions] != query_camera])

        if len(query_index) < 10:
            continue
        query_index = np.asarray(query_index)
        gallery_index = np.asarray(gallery_index)

        mean_ap, _ = evaluate_level(
            subset_features[query_index], subset_features[gallery_index],
            subset_pids[query_index], subset_pids[gallery_index],
            subset_cams[query_index], subset_cams[gallery_index],
        )
        scores.append(mean_ap)
        details[int(view)] = {
            "map": mean_ap,
            "queries": int(len(query_index)),
            "gallery": int(len(gallery_index)),
        }
    return (float(np.mean(scores)) if scores else float("nan")), details


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


def build_ground_pool(data_root: str, protocols: list[str], cfg):
    """Every ground image across all protocols, deduplicated by path.

    A same-view control needs identities seen from two ground cameras, and no
    single protocol split contains more than one camera, so the pool has to be
    assembled across protocols.
    """
    items: dict[str, tuple[int, int]] = {}
    for protocol in protocols:
        path = os.path.join(data_root, protocol)
        if not os.path.exists(path):
            continue
        for split in ("query", "gallery"):
            dataset = EvalReIDDataset(
                data_root, path, split, tuple(cfg.image_size),
                cfg.norm_mean, cfg.norm_std,
            )
            for image_path, pid, camid in dataset.items:
                if camid in (2, 3):          # ground only
                    items[image_path] = (pid, camid)

    ordered = sorted(items.items())
    return [(path, pid, camid) for path, (pid, camid) in ordered]


class _PathListDataset(torch.utils.data.Dataset):
    """Minimal dataset over an explicit (path, pid, camid) list."""

    def __init__(self, items, image_size, norm_mean, norm_std):
        from PIL import Image
        from reid_advance.data import build_eval_transform

        self._open = Image.open
        self.items = items
        self.transform = build_eval_transform(image_size, norm_mean, norm_std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, pid, camid = self.items[index]
        return self.transform(self._open(path).convert("RGB")), pid, camid


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

    # Same-view control, computed once: it is a property of the checkpoint, not
    # of any one protocol, and it needs images pooled across all of them.
    all_protocols = [
        "exp1_aerial_to_cctv.txt", "exp2_aerial_to_wearable.txt",
        "exp4_cctv_to_aerial.txt", "exp5_wearable_to_aerial.txt",
    ]
    ground_items = build_ground_pool(args.data_root, all_protocols, cfg)
    print(f"\nground pool for the same-view control: {len(ground_items):,} images")
    ground_loader = DataLoader(
        _PathListDataset(ground_items, tuple(cfg.image_size), cfg.norm_mean, cfg.norm_std),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    ground_levels, ground_pids, ground_cams = extract_pyramid(
        model, ground_loader, spec, device, use_amp
    )
    ground_view = binary_view_tensor(torch.as_tensor(ground_cams)).numpy()
    same_view = []
    for level in range(spec.depth):
        mean_ap, detail = same_view_control(
            ground_levels[level], ground_pids, ground_cams, ground_view
        )
        same_view.append({"map": mean_ap, "detail": detail})
        print(f"  L{level} {LEVEL_NAMES[level]:<8} same-view (ground, cross-camera) "
              f"mAP {mean_ap:.2%}")

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
            same_map = same_view[level]["map"]
            same_detail = same_view[level]["detail"]

            per_level.append({
                "level": level,
                "name": LEVEL_NAMES[level],
                "regions": spec.levels[level],
                "rows_per_region": spec.rows_per_region(level),
                "cross_view_map": cross_map,
                "cross_view_rank1": cross_r1,
                "same_view_map": same_map,
                "same_view_detail": same_detail,
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
            "- cross-view mAP uses the official protocol unchanged\n\n"
        )
        handle.write(
            "## How the same-view control is built\n\n"
            "The official protocols are cross-view by construction: query and "
            "gallery always sit on different platforms, so the dataset ships no "
            "same-view ground truth. The control below is therefore **constructed**, "
            "and the construction is part of the claim - any statement of the form "
            "\"the hierarchy closes X% of the gap\" inherits whatever this "
            "denominator measures.\n\n"
            "Construction, from the gallery split alone:\n\n"
            "1. Keep only identities photographed by **at least two cameras of the "
            "same platform**.\n"
            "2. Draw one camera at random per identity; one of its images becomes "
            "the query.\n"
            "3. That identity's images from its *other* cameras form the gallery.\n"
            "4. Score with the same `evaluate_rank` as the cross-view number, with "
            "its same-pid-same-camera filter left **on**.\n\n"
            "Point 4 is the one that matters. Retrieving another shot from the very "
            "same camera is a much easier task than re-identification, so a control "
            "that allowed it would measure something easier than the cross-view "
            "number it is contrasted with and would inflate the gap. An earlier "
            "version of this script did exactly that by offsetting the gallery "
            "camera ids; the figures below come from the corrected version.\n\n"
            "**The control is ground-only.** Ground spans two cameras (wearable C2, "
            "CCTV C3) so a cross-camera same-view pair exists. Aerial is a single "
            "camera (C0), so aerial has no same-view cross-camera pair at all and is "
            "skipped. \"Same-view mAP\" below therefore means *ground-to-ground*, "
            "never aerial-to-aerial.\n\n"
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
