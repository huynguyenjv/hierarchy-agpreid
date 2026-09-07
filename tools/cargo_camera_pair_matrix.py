"""Camera-pair matrix and same-view control for CARGO.

The AG-ReID.v2 counterpart (``tools/camera_pair_matrix.py``) found a flat
matrix: no camera pair was meaningfully harder than another, and aerial-ground
scored slightly *better* than ground-ground. With one dataset that finding is
indistinguishable from a quirk of that dataset. CARGO answers whether it
generalises.

CARGO is the stronger measurement of the two. AG-ReID.v2 has a single aerial
camera, so aerial-to-aerial retrieval does not exist there and the same-view
control had to be ground-only. CARGO has five aerial and eight ground cameras,
so both same-view controls are constructible - 2,351 gallery identities appear
on two or more cameras of each platform.

Everything scoreable is aggregated three ways:

    aerial <-> ground   the axis the topic assumed was hard
    aerial <-> aerial   same-platform control, impossible on AG-ReID.v2
    ground <-> ground   same-platform control

Usage:
    venv/Scripts/python.exe tools/cargo_camera_pair_matrix.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_advance.cargo import (
    AERIAL_CAMERAS,
    GROUND_CAMERAS,
    CargoEvalDataset,
    platform_of_camera,
)
from reid_advance.config import CargoConfig
from reid_advance.evaluation import evaluate_rank
from reid_advance.pipelines.transreid import TransReIDSmall


@torch.no_grad()
def extract(model, loader, device, use_amp):
    model.eval()
    features, pids, cams = [], [], []
    for images, pid, camid in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        camera = camid.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            feature = model(images, camera)
        features.append(feature.float().cpu())
        pids.append(pid)
        cams.append(camid)
    return torch.cat(features), torch.cat(pids).numpy(), torch.cat(cams).numpy()


def score_pair(features, pids, cams, query_cam, gallery_cam, seed=42, max_queries=600):
    """mAP for one ordered camera pair, one query image per identity.

    The two cameras differ by construction, so evaluate_rank's
    same-pid-same-camera filter never removes a true match here - the same
    property the AG-ReID.v2 same-view control had to be rebuilt to preserve.
    """
    rng = np.random.default_rng(seed)
    query_mask = cams == query_cam
    gallery_mask = cams == gallery_cam
    if query_mask.sum() < 10 or gallery_mask.sum() < 10:
        return None

    gallery_pids = pids[gallery_mask]
    shared = np.intersect1d(np.unique(pids[query_mask]), np.unique(gallery_pids))
    if len(shared) < 10:
        return None
    if len(shared) > max_queries:
        shared = rng.choice(shared, max_queries, replace=False)

    query_source = np.flatnonzero(query_mask)
    query_positions = np.asarray([
        rng.choice(query_source[pids[query_source] == pid]) for pid in shared
    ])

    distance = (1.0 - features[query_positions] @ features[gallery_mask].t()).numpy()
    cmc, mean_ap = evaluate_rank(
        distance, pids[query_positions], gallery_pids,
        cams[query_positions], cams[gallery_mask],
    )
    return {
        "map": float(mean_ap), "rank1": float(cmc[0]),
        "identities": int(len(shared)),
        "gallery_images": int(gallery_mask.sum()),
    }


#: Below this many identities in common, a pair's mAP says more about an empty
#: gallery than about view difficulty, so it is reported but not interpreted.
MIN_SHARED_IDENTITIES = 100


def shared_identity_matrix(pids, cams) -> dict[tuple[int, int], int]:
    """Identities each camera pair has in common.

    A camera pair that shares few identities has almost no true matches to
    retrieve, so its mAP measures gallery emptiness rather than view
    difficulty. This is the same failure that produced the bogus AG-ReID.v2
    same-view numbers, wearing different clothes, so it is counted explicitly
    and printed next to the mAP rather than trusted implicitly.
    """
    per_camera = {
        int(camera): set(pids[cams == camera].tolist())
        for camera in sorted(set(cams.tolist()))
    }
    return {
        (left, right): len(per_camera[left] & per_camera[right])
        for left, right in itertools.permutations(sorted(per_camera), 2)
    }


def pair_kind(query_cam: int, gallery_cam: int) -> str:
    left, right = platform_of_camera(query_cam), platform_of_camera(gallery_cam)
    if left != right:
        return "aerial-ground"
    return f"{left}-{left}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/cargo/best_model.pth")
    parser.add_argument("--data-root", default="D:/datasets/cargo")
    parser.add_argument("--split", default="gallery",
                        help="CARGO's query split has only 312 images; the "
                             "gallery carries all 13 cameras and 2,500 ids")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="docs/cargo_camera_pair_matrix.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = state.get("config", {})
    cfg = CargoConfig()
    for key in ("encoder_name", "embed_dim", "image_size", "transreid_sie",
                "transreid_jpm", "transreid_num_parts", "norm_mean", "norm_std",
                "sie_num_views", "sie_identity_map"):
        if key in saved:
            setattr(cfg, key, saved[key])
    cfg.device = device

    model = TransReIDSmall(cfg, state.get("train_identity_count", 2500)).to(device)
    model.load_state_dict(state["model"])
    print(f"loaded {args.checkpoint} | epoch {state.get('epoch')} "
          f"| reported mAP {state.get('mAP', float('nan')):.2%}")

    dataset = CargoEvalDataset(
        os.path.join(args.data_root, args.split),
        tuple(cfg.image_size), cfg.norm_mean, cfg.norm_std,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    print(f"scoring the '{args.split}' split: {len(dataset):,} images")

    features, pids, cams = extract(model, loader, device, use_amp)
    cameras = sorted(set(cams.tolist()))
    print(f"cameras present: {cameras}")

    shared = shared_identity_matrix(pids, cams)
    counts = sorted(shared.values())
    thin = [pair for pair, count in shared.items() if count < MIN_SHARED_IDENTITIES]
    print(f"shared identities per pair: min {counts[0]}, median "
          f"{counts[len(counts) // 2]}, max {counts[-1]}")
    if thin:
        print(f"  {len(thin)} of {len(shared)} pairs share fewer than "
              f"{MIN_SHARED_IDENTITIES} identities and will be excluded from the "
              "aggregates")
    else:
        print(f"  all {len(shared)} pairs clear the {MIN_SHARED_IDENTITIES}-identity "
              "floor, so no cell is an empty-gallery artifact")

    results = {}
    for query_cam, gallery_cam in itertools.permutations(cameras, 2):
        entry = score_pair(features, pids, cams, query_cam, gallery_cam)
        if entry is None:
            continue
        common = shared[(query_cam, gallery_cam)]
        results[f"Cam{query_cam}->Cam{gallery_cam}"] = {
            "query_camera": query_cam, "gallery_camera": gallery_cam,
            "kind": pair_kind(query_cam, gallery_cam),
            "shared_identities": common,
            "reliable": common >= MIN_SHARED_IDENTITIES,
            **entry,
        }

    # Aggregate only over cells with enough true matches to mean anything.
    by_kind = defaultdict(list)
    for entry in results.values():
        if entry["reliable"]:
            by_kind[entry["kind"]].append(entry["map"])

    print(f"\n{'kind':<16}{'pairs':>7}{'mean mAP':>10}{'min':>9}{'max':>9}")
    summary = {}
    for kind in ("aerial-ground", "aerial-aerial", "ground-ground"):
        scores = by_kind.get(kind, [])
        if not scores:
            continue
        summary[kind] = {
            "pairs": len(scores), "mean": float(np.mean(scores)),
            "min": float(min(scores)), "max": float(max(scores)),
        }
        print(f"{kind:<16}{len(scores):>7}{np.mean(scores):>9.2%}"
              f"{min(scores):>9.2%}{max(scores):>9.2%}")

    all_scores = [e["map"] for e in results.values() if e["reliable"]]
    spread = max(all_scores) - min(all_scores)
    cross = summary.get("aerial-ground", {}).get("mean")
    same_scores = by_kind.get("aerial-aerial", []) + by_kind.get("ground-ground", [])
    same = float(np.mean(same_scores)) if same_scores else None

    print(f"\nspread across all pairs: {spread:.2%}")
    if cross is not None and same is not None:
        print(f"aerial-ground {cross:.2%} vs same-platform {same:.2%} "
              f"-> gap {same - cross:+.2%}")

    if cross is not None and same is not None and same - cross > 0.05:
        verdict = ("CARGO RETAINS an aerial-ground gap: same-platform retrieval "
                   f"beats cross-platform by {same - cross:.2%}")
    elif spread < 0.05:
        verdict = ("FLAT, like AG-ReID.v2 - no pair is meaningfully harder, so "
                   "the finding generalises across two datasets")
    else:
        verdict = ("UNEVEN but not along the aerial-ground axis, as on "
                   "AG-ReID.v2")
    print(f"\nverdict: {verdict}")

    payload = {
        "checkpoint": args.checkpoint, "split": args.split,
        "epoch": state.get("epoch"), "reported_map": state.get("mAP"),
        "pairs": results, "by_kind": summary,
        "shared_identities": {f"Cam{a}->Cam{b}": n for (a, b), n in shared.items()},
        "min_shared_identities": MIN_SHARED_IDENTITIES,
        "spread": spread, "aerial_ground_mean": cross, "same_platform_mean": same,
        "verdict": verdict,
    }
    base = os.path.splitext(args.output)[0]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(base + ".json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# CARGO camera-pair retrieval matrix\n\n")
        handle.write(
            f"Checkpoint `{args.checkpoint}` (epoch {state.get('epoch')}), trained "
            "with the same recipe as the AG-ReID.v2 baseline: TransReID-S ViT-S/16, "
            "identity + triplet loss, PK sampler, both platforms in the training "
            "split. Same recipe is what makes the two matrices comparable.\n\n"
            f"Scored on the `{args.split}` split, one query image per identity per "
            "pair.\n\n"
        )
        handle.write("## By pair kind\n\n")
        handle.write("| kind | pairs | mean mAP | min | max |\n|---|---|---|---|---|\n")
        for kind, entry in summary.items():
            handle.write(f"| {kind} | {entry['pairs']} | {entry['mean']:.2%} | "
                         f"{entry['min']:.2%} | {entry['max']:.2%} |\n")
        handle.write(
            "\nAG-ReID.v2 could not report an `aerial-aerial` row at all: it has a "
            "single aerial camera, so aerial same-view retrieval does not exist "
            "there. CARGO's five aerial cameras make this the stricter test.\n\n"
        )
        handle.write("## Shared identities per camera pair\n\n")
        handle.write(
            "Read this before the mAP table. A pair sharing few identities has "
            "almost no true matches to retrieve, so its mAP would measure an "
            "empty gallery rather than view difficulty - the same failure mode "
            "that produced the discarded AG-ReID.v2 same-view figures. Cells "
            f"below {MIN_SHARED_IDENTITIES} shared identities are excluded from "
            "every aggregate above.\n\n"
        )
        handle.write("| | " + " | ".join(f"Cam{c}" for c in cameras) + " |\n")
        handle.write("|---" * (len(cameras) + 1) + "|\n")
        for left in cameras:
            cells = []
            for right in cameras:
                if left == right:
                    cells.append("-")
                else:
                    count = shared[(left, right)]
                    cells.append(f"{count}" if count >= MIN_SHARED_IDENTITIES
                                 else f"**{count}**")
            handle.write(f"| **Cam{left}** | " + " | ".join(cells) + " |\n")
        handle.write(
            f"\nRange: {counts[0]} to {counts[-1]} identities per pair. "
            + (f"{len(thin)} pair(s) fall below the floor and are struck from the "
               "aggregates.\n\n" if thin else
               "Every pair clears the floor, so no cell in the mAP table is an "
               "empty-gallery artifact.\n\n")
        )

        handle.write("## Hardest and easiest pairs\n\n")
        handle.write("| query | gallery | kind | mAP | Rank-1 | #IDs scored | shared IDs |\n")
        handle.write("|---|---|---|---|---|---|---|\n")
        ordered = sorted(
            (kv for kv in results.items() if kv[1]["reliable"]),
            key=lambda kv: kv[1]["map"],
        )
        for key, entry in ordered[:8] + ordered[-4:]:
            handle.write(
                f"| Cam{entry['query_camera']} | Cam{entry['gallery_camera']} | "
                f"{entry['kind']} | {entry['map']:.2%} | {entry['rank1']:.2%} | "
                f"{entry['identities']} | {entry['shared_identities']} |\n"
            )
        handle.write(f"\n- spread across all {len(results)} pairs: **{spread:.2%}**\n")
        if cross is not None and same is not None:
            handle.write(f"- aerial-ground mean {cross:.2%}, same-platform mean "
                         f"{same:.2%}, gap **{same - cross:+.2%}**\n")
        handle.write(f"\n**Verdict: {verdict}.**\n")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
