"""Where does the view gap actually live? Per-camera-pair retrieval.

Every earlier premise of this topic collapsed against a control:

- "coarse attributes survive the climb"  -> retention ~1.0 for 14 of 15 (AI-03)
- "aerial-ground is the hard direction"  -> the gap is slightly negative (GSS)

Both premises were about *aggregate* aerial-vs-ground. This script stops
aggregating: it scores retrieval for every ordered pair of cameras that exists
in the data, so the gap can be located rather than assumed.

AG-ReID.v2 has three cameras: C0 aerial (UAV), C2 wearable (smart glasses),
C3 CCTV. That gives six ordered cross-camera pairs. If one pair is markedly
harder than the rest, that pair is the real problem and any loss should target
it. If the matrix is flat, there is no gap left for a loss to close - which is
itself the finding, and has to be reported as such.

Note the checkpoint under test was trained on train_all, which contains all
three cameras, so it has already had cross-view supervision. A flat matrix is
therefore evidence about *this trained model*, not proof that the dataset is
inherently easy; the --random-init flag scores an untrained backbone for
contrast.

Usage:
    venv/Scripts/python.exe tools/camera_pair_matrix.py
    venv/Scripts/python.exe tools/camera_pair_matrix.py --random-init
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_advance.config import TransReIDConfig
from reid_advance.data import EvalReIDDataset, build_eval_transform
from reid_advance.evaluation import evaluate_rank
from reid_advance.hierarchy import CAMERA_TO_PLATFORM
from reid_advance.identity import require_current_identity_scheme
from reid_advance.pipelines.transreid import TransReIDSmall

CAMERA_LABEL = {0: "C0 aerial", 2: "C2 wearable", 3: "C3 cctv"}

ALL_PROTOCOLS = [
    "exp1_aerial_to_cctv.txt", "exp2_aerial_to_wearable.txt",
    "exp4_cctv_to_aerial.txt", "exp5_wearable_to_aerial.txt",
]


class PathListDataset(Dataset):
    def __init__(self, items, image_size, norm_mean, norm_std):
        from PIL import Image

        self._open = Image.open
        self.items = items
        self.transform = build_eval_transform(image_size, norm_mean, norm_std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, pid, camid = self.items[index]
        return self.transform(self._open(path).convert("RGB")), pid, camid


def collect_test_images(data_root: str, cfg) -> list[tuple[str, int, int]]:
    """Union of every image named by any official protocol, deduplicated."""
    items: dict[str, tuple[int, int]] = {}
    for protocol in ALL_PROTOCOLS:
        path = os.path.join(data_root, protocol)
        if not os.path.exists(path):
            continue
        for split in ("query", "gallery"):
            dataset = EvalReIDDataset(
                data_root, path, split, tuple(cfg.image_size),
                cfg.norm_mean, cfg.norm_std,
            )
            for image_path, pid, camid in dataset.items:
                items[image_path] = (pid, camid)
    return [(path, pid, camid) for path, (pid, camid) in sorted(items.items())]


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


def score_pair(features, pids, cams, query_cam: int, gallery_cam: int,
               seed: int = 42, max_queries: int = 600):
    """mAP for query images from one camera against gallery from another.

    One query per identity keeps prolific identities from dominating, and the
    cameras differ by construction so evaluate_rank's same-camera filter never
    removes a true match.
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

    query_positions = []
    query_source = np.flatnonzero(query_mask)
    for pid in shared:
        candidates = query_source[pids[query_source] == pid]
        query_positions.append(rng.choice(candidates))
    query_positions = np.asarray(query_positions)

    distance = (
        1.0 - features[query_positions] @ features[gallery_mask].t()
    ).numpy()
    cmc, mean_ap = evaluate_rank(
        distance, pids[query_positions], gallery_pids,
        cams[query_positions], cams[gallery_mask],
    )
    return {
        "map": float(mean_ap),
        "rank1": float(cmc[0]),
        "identities": int(len(shared)),
        "gallery_images": int(gallery_mask.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/transreid/best_model.pth")
    parser.add_argument("--data-root", default="AG-ReID.v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-init", action="store_true",
                        help="score an untrained backbone for contrast")
    parser.add_argument("--output", default="docs/camera_pair_matrix.md")
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

    model = TransReIDSmall(cfg, state.get("train_identity_count", 807)).to(device)
    if args.random_init:
        print("scoring a RANDOMLY INITIALISED backbone (contrast run)")
    else:
        model.load_state_dict(state["model"])
        print(f"loaded {args.checkpoint} | epoch {state.get('epoch')} "
              f"| reported mAP {state.get('mAP', float('nan')):.2%} "
              f"on {saved.get('eval_txt_file')}")
        print(f"trained on '{saved.get('train_dir')}' - note this split contains "
              "all three cameras, so the model has already had cross-view "
              "supervision")

    items = collect_test_images(args.data_root, cfg)
    per_camera = defaultdict(int)
    for _, _, camid in items:
        per_camera[camid] += 1
    print(f"\ntest pool: {len(items):,} images | "
          + " | ".join(f"{CAMERA_LABEL[c]} {n:,}" for c, n in sorted(per_camera.items())))

    loader = DataLoader(
        PathListDataset(items, tuple(cfg.image_size), cfg.norm_mean, cfg.norm_std),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    features, pids, cams = extract(model, loader, device, use_amp)

    cameras = sorted(per_camera)
    results = {}
    print(f"\n{'query':<14}{'gallery':<14}{'mAP':>8}{'Rank-1':>9}{'#IDs':>7}")
    for query_cam, gallery_cam in itertools.permutations(cameras, 2):
        entry = score_pair(features, pids, cams, query_cam, gallery_cam)
        if entry is None:
            continue
        key = f"C{query_cam}->C{gallery_cam}"
        results[key] = {
            "query_camera": query_cam, "gallery_camera": gallery_cam,
            "query_platform": CAMERA_TO_PLATFORM[query_cam],
            "gallery_platform": CAMERA_TO_PLATFORM[gallery_cam],
            "cross_platform": (
                CAMERA_TO_PLATFORM[query_cam] == "aerial"
            ) != (CAMERA_TO_PLATFORM[gallery_cam] == "aerial"),
            **entry,
        }
        print(f"{CAMERA_LABEL[query_cam]:<14}{CAMERA_LABEL[gallery_cam]:<14}"
              f"{entry['map']:>7.2%}{entry['rank1']:>9.2%}{entry['identities']:>7}")

    scores = [entry["map"] for entry in results.values()]
    spread = max(scores) - min(scores)
    aerial_ground = [e["map"] for e in results.values() if e["cross_platform"]]
    ground_ground = [e["map"] for e in results.values() if not e["cross_platform"]]

    print(f"\nspread across pairs: {spread:.2%} "
          f"(min {min(scores):.2%}, max {max(scores):.2%})")
    if aerial_ground:
        print(f"aerial<->ground mean: {np.mean(aerial_ground):.2%}")
    if ground_ground:
        print(f"ground<->ground mean: {np.mean(ground_ground):.2%}")

    hardest = min(results.items(), key=lambda kv: kv[1]["map"])
    print(f"hardest pair: {hardest[0]} at {hardest[1]['map']:.2%}")

    if spread < 0.05:
        verdict = (
            "FLAT - no pair is meaningfully harder than another, so there is no "
            "view gap left for a loss to close with this checkpoint"
        )
    elif aerial_ground and ground_ground and (
        np.mean(ground_ground) - np.mean(aerial_ground) > 0.05
    ):
        verdict = "AERIAL-GROUND is the hard axis, as the topic assumed"
    else:
        verdict = (
            f"UNEVEN but not along the aerial-ground axis; hardest pair is "
            f"{hardest[0]}"
        )
    print(f"\nverdict: {verdict}")

    payload = {
        "checkpoint": args.checkpoint,
        "random_init": args.random_init,
        "trained_on": saved.get("train_dir"),
        "pairs": results,
        "spread": spread,
        "aerial_ground_mean": float(np.mean(aerial_ground)) if aerial_ground else None,
        "ground_ground_mean": float(np.mean(ground_ground)) if ground_ground else None,
        "verdict": verdict,
    }
    suffix = "_random" if args.random_init else ""
    base = os.path.splitext(args.output)[0] + suffix
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(base + ".json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(base + ".md", "w", encoding="utf-8") as handle:
        handle.write("# Camera-pair retrieval matrix\n\n")
        handle.write(
            "Locates the view gap instead of assuming it. Every ordered pair of "
            "cameras that exists in AG-ReID.v2 is scored separately, one query per "
            "identity.\n\n"
        )
        if args.random_init:
            handle.write("**Randomly initialised backbone** (contrast run).\n\n")
        else:
            handle.write(
                f"Checkpoint `{args.checkpoint}`, epoch {state.get('epoch')}, "
                f"trained on `{saved.get('train_dir')}`.\n\n"
                "That training split contains all three cameras, so this model has "
                "already received cross-view supervision through its identity and "
                "triplet losses. Read the matrix as a statement about this trained "
                "model, not about the dataset in the abstract.\n\n"
            )
        handle.write("| query | gallery | platforms | mAP | Rank-1 | #IDs |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for key, entry in sorted(results.items(), key=lambda kv: kv[1]["map"]):
            kind = "aerial-ground" if entry["cross_platform"] else "ground-ground"
            handle.write(
                f"| {CAMERA_LABEL[entry['query_camera']]} | "
                f"{CAMERA_LABEL[entry['gallery_camera']]} | {kind} | "
                f"{entry['map']:.2%} | {entry['rank1']:.2%} | {entry['identities']} |\n"
            )
        handle.write(f"\n- spread across pairs: **{spread:.2%}**\n")
        if aerial_ground:
            handle.write(f"- aerial<->ground mean: {np.mean(aerial_ground):.2%}\n")
        if ground_ground:
            handle.write(f"- ground<->ground mean: {np.mean(ground_ground):.2%}\n")
        handle.write(f"\n**Verdict: {verdict}.**\n")

    print(f"\nWrote {base}.md")


if __name__ == "__main__":
    main()
