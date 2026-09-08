"""Does the aerial-ground gap shrink as the model gets stronger?

This replaces a comparison the data cannot support. The original plan was to
score both datasets at the same absolute mAP (~70%) so that "CARGO has a gap,
AG-ReID.v2 does not" could not be re-read as "the CARGO model is just weaker".
But CARGO tops out at 46.6% under this recipe - mAP went 42.1 -> 45.5 -> 46.9
-> 46.6 across epochs 10/20/25/30 while the train loss kept falling, which is
overfitting rather than room to grow. The two datasets reach different ceilings
with the same recipe, so no amount of extra training puts them at the same
point.

The question underneath was never "same mAP" though. It was whether the gap is
intrinsic to the dataset or an artifact of an under-trained model. That is
answerable as a *curve*: measure the aerial-ground gap at several checkpoints
spanning each dataset's own capability range, and see which way it moves.

    gap shrinks towards zero as mAP rises  ->  supervision dissolves it; a
                                               method has nothing left to close
    gap holds or grows as mAP rises        ->  intrinsic; capability alone does
                                               not remove it

Two curves are stronger evidence than one matched pair, because they cannot be
dismissed as a comparison made at one arbitrarily chosen operating point.

Usage:
    venv/Scripts/python.exe tools/gap_vs_capability.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_series(pattern: str, label: str) -> list[dict]:
    """Read the per-epoch matrix JSONs a dataset has produced."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        by_kind = payload.get("by_kind", {})
        if "aerial-ground" not in by_kind:
            continue
        cross = by_kind["aerial-ground"]["mean"]
        # Weight the two same-platform kinds by how many pairs each contributes.
        # An unweighted mean would let 20 aerial-aerial pairs count as much as
        # 56 ground-ground ones and would shift the gap by several points.
        same_pairs = [
            (by_kind[kind]["mean"], by_kind[kind]["pairs"])
            for kind in ("aerial-aerial", "ground-ground")
            if kind in by_kind
        ]
        if not same_pairs:
            continue
        total_pairs = sum(count for _, count in same_pairs)
        same_platform = sum(mean * count for mean, count in same_pairs) / total_pairs
        epoch = payload.get("epoch")
        if epoch is None:
            match = re.search(r"e?(\d+)\.json$", os.path.basename(path))
            epoch = int(match.group(1)) if match else None
        rows.append({
            "dataset": label,
            "source": os.path.basename(path),
            "epoch": epoch,
            "overall_map": payload.get("reported_map"),
            "aerial_ground": cross,
            "same_platform": float(same_platform),
            "gap": float(same_platform) - cross,
            "by_kind": {k: v["mean"] for k, v in by_kind.items()},
        })
    rows.sort(key=lambda row: (row["epoch"] is None, row["epoch"]))
    return rows


def trend(rows: list[dict]) -> dict:
    """Direction and slope of the gap against model capability."""
    usable = [r for r in rows if r["overall_map"] is not None]
    if len(usable) < 2:
        return {"verdict": "too few points", "slope": None}
    capability = np.array([r["overall_map"] for r in usable])
    gaps = np.array([r["gap"] for r in usable])
    # Gap change per point of mAP gained.
    slope = float(np.polyfit(capability, gaps, 1)[0])
    delta = float(gaps[-1] - gaps[0])
    if slope < -0.15:
        verdict = "SHRINKS with capability - supervision is dissolving the gap"
    elif slope > 0.15:
        verdict = "GROWS with capability - the gap is not a training artifact"
    else:
        verdict = "FLAT against capability - the gap is intrinsic, not under-training"
    return {
        "verdict": verdict, "slope": slope, "delta": delta,
        "capability_range": [float(capability.min()), float(capability.max())],
        "gap_range": [float(gaps.min()), float(gaps.max())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo-glob", default="docs/cargo_*matrix*.json")
    parser.add_argument("--agreid-glob", default="docs/agreid_matrix_e*.json")
    parser.add_argument("--output", default="docs/gap_vs_capability.md")
    args = parser.parse_args()

    series = {
        "CARGO": load_series(args.cargo_glob, "CARGO"),
        "AG-ReID.v2": load_series(args.agreid_glob, "AG-ReID.v2"),
    }

    trends = {}
    for name, rows in series.items():
        if not rows:
            print(f"{name}: no matrix JSONs found")
            continue
        print(f"\n=== {name} ===")
        print(f"{'epoch':>7}{'overall mAP':>13}{'aerial-ground':>15}"
              f"{'same-platform':>15}{'gap':>9}")
        for row in rows:
            overall = f"{row['overall_map']:.2%}" if row["overall_map"] is not None else "-"
            print(f"{str(row['epoch']):>7}{overall:>13}{row['aerial_ground']:>14.2%}"
                  f"{row['same_platform']:>15.2%}{row['gap']:>+9.2%}")
        trends[name] = trend(rows)
        info = trends[name]
        if info["slope"] is not None:
            print(f"  slope {info['slope']:+.3f} gap per unit mAP | "
                  f"end-to-end change {info['delta']:+.2%}")
        print(f"  -> {info['verdict']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(os.path.splitext(args.output)[0] + ".json", "w", encoding="utf-8") as handle:
        json.dump({"series": series, "trends": trends}, handle, indent=2)

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# Aerial-ground gap as a function of model capability\n\n")
        handle.write(
            "The plan was to compare both datasets at the same absolute mAP so "
            "that \"CARGO has a gap, AG-ReID.v2 does not\" could not be re-read as "
            "\"the CARGO model is simply weaker\". That comparison turned out to be "
            "impossible: CARGO tops out at 46.6% under this recipe while "
            "AG-ReID.v2 sits above 70%. mAP on CARGO went 42.1 -> 45.5 -> 46.9 -> "
            "46.6 across epochs 10/20/25/30 while the train loss kept falling, so "
            "this is the ceiling of the recipe, not under-training. Same recipe, "
            "two different ceilings.\n\n"
            "The underlying question survives without matched mAP. Measuring the "
            "gap at several checkpoints across each dataset's own capability range "
            "asks the same thing more directly: does more capability dissolve the "
            "gap, or not? A curve also resists the objection that any single "
            "matched operating point was chosen arbitrarily.\n\n"
        )
        for name, rows in series.items():
            if not rows:
                continue
            handle.write(f"## {name}\n\n")
            handle.write("| epoch | overall mAP | aerial-ground | same-platform | gap |\n")
            handle.write("|---|---|---|---|---|\n")
            for row in rows:
                overall = (f"{row['overall_map']:.2%}"
                           if row["overall_map"] is not None else "-")
                handle.write(
                    f"| {row['epoch']} | {overall} | {row['aerial_ground']:.2%} | "
                    f"{row['same_platform']:.2%} | **{row['gap']:+.2%}** |\n"
                )
            info = trends.get(name, {})
            if info.get("slope") is not None:
                handle.write(
                    f"\nSlope {info['slope']:+.3f} gap per unit of mAP, end-to-end "
                    f"change {info['delta']:+.2%} over a capability range of "
                    f"{info['capability_range'][0]:.2%}-{info['capability_range'][1]:.2%}.\n"
                )
            handle.write(f"\n**{info.get('verdict', 'n/a')}**\n\n")

        handle.write(
            "## Reading the two curves together\n\n"
            "If AG-ReID.v2's gap falls towards zero as its model strengthens while "
            "CARGO's holds flat across its own range, then the two datasets differ "
            "in kind and not merely in difficulty: ordinary supervision is enough "
            "to erase the view gap on one and not on the other. That is the causal "
            "claim, and it does not require the two models to sit at the same mAP.\n\n"
            "If instead both curves fall and CARGO's has simply not fallen yet, the "
            "honest reading is that the gap is a function of supervision on both, "
            "and CARGO is only further from the point where it vanishes.\n"
        )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
