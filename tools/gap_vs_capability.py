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

Which number to compare against AG-ReID.v2: its camera-pair matrix gives
ground-ground 71.59% against aerial-ground 73.18%, a gap of **-1.59%**. Do not
quote the -1.11% that appears in the GSS report - that came from a different
measurement (protocol cross-view against a hand-built same-view control) and is
not on the same axis as anything here.

AG-ReID.v2 has no aerial-aerial pairs, so its "all same-platform" and
"ground-only" gaps are the same number; the distinction only bites on CARGO.

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

        # Second definition, needed because the two datasets are not built the
        # same way. AG-ReID.v2 has a single aerial camera, so it has no
        # aerial-aerial row at all; comparing its gap to CARGO's full
        # same-platform mean would compare different quantities. Ground-ground
        # against aerial-ground is the one axis both datasets can produce, so
        # every conclusion is checked against it too.
        ground_only = by_kind.get("ground-ground", {}).get("mean")
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
            "ground_ground": ground_only,
            "gap_ground_only": (float(ground_only) - cross) if ground_only is not None else None,
            "by_kind": {k: v["mean"] for k, v in by_kind.items()},
        })
    rows.sort(key=lambda row: (row["epoch"] is None, row["epoch"]))
    return rows


def trend(rows: list[dict], key: str = "gap") -> dict:
    """Direction and slope of one gap definition against model capability."""
    usable = [r for r in rows if r["overall_map"] is not None and r.get(key) is not None]
    if len(usable) < 2:
        return {"verdict": "too few points", "slope": None}
    capability = np.array([r["overall_map"] for r in usable])
    gaps = np.array([r[key] for r in usable])
    # Gap change per point of mAP gained.
    slope = float(np.polyfit(capability, gaps, 1)[0])
    delta = float(gaps[-1] - gaps[0])
    # Slope alone is a poor verdict. A gap that starts near zero cannot fall
    # steeply in absolute terms however completely supervision dissolves it, so
    # a fixed slope threshold declares it "flat" while it is visibly crossing
    # into negative territory. What matters is where the gap ends up relative
    # to its own size, not how fast it got there.
    crosses_zero = bool(gaps[0] > 0 and gaps[-1] < 0)
    relative_change = float(delta / abs(gaps[0])) if gaps[0] else 0.0

    if crosses_zero:
        verdict = ("DISSOLVES - the gap starts positive and ends negative; "
                   "supervision removes it entirely")
    elif relative_change < -0.25:
        verdict = (f"SHRINKS - the gap loses {abs(relative_change):.0%} of its "
                   "size as capability rises")
    elif relative_change > 0.25:
        verdict = "GROWS with capability"
    else:
        verdict = ("HOLDS - the gap keeps its size across the capability range, "
                   "so it is intrinsic rather than a training artifact")
    return {
        "verdict": verdict, "slope": slope, "delta": delta,
        "relative_change": relative_change, "crosses_zero": crosses_zero,
        "start_gap": float(gaps[0]), "end_gap": float(gaps[-1]),
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
        print(f"{'epoch':>7}{'overall mAP':>13}{'a-g':>9}{'same-plat':>11}"
              f"{'gap':>9}{'g-g':>9}{'gap(gg)':>10}")
        for row in rows:
            overall = f"{row['overall_map']:.2%}" if row["overall_map"] is not None else "-"
            ground = f"{row['ground_ground']:.2%}" if row["ground_ground"] is not None else "-"
            gap_gg = (f"{row['gap_ground_only']:+.2%}"
                      if row["gap_ground_only"] is not None else "-")
            print(f"{str(row['epoch']):>7}{overall:>13}{row['aerial_ground']:>9.2%}"
                  f"{row['same_platform']:>11.2%}{row['gap']:>+9.2%}"
                  f"{ground:>9}{gap_gg:>10}")

        trends[name] = {
            "all_same_platform": trend(rows, "gap"),
            "ground_only": trend(rows, "gap_ground_only"),
        }
        for label, info in trends[name].items():
            if info["slope"] is None:
                continue
            print(f"  [{label}] slope {info['slope']:+.3f} per unit mAP | "
                  f"change {info['delta']:+.2%} | {info['verdict']}")

        # A slope is only meaningful if capability actually varied. With every
        # checkpoint bunched into a few points of mAP there is no capability
        # axis to regress against, and the honest reading is the level the gap
        # holds at, not its direction.
        info = trends[name]["all_same_platform"]
        if info["slope"] is not None:
            low, high = info["capability_range"]
            if high - low < 0.10:
                print(f"  !! mAP only spans {low:.1%}-{high:.1%} ({high - low:.1%} "
                      "wide): too narrow to read a slope. Report the level the gap "
                      f"holds at ({np.mean([r['gap'] for r in rows]):+.2%}) rather "
                      "than its direction.")

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
            handle.write("| epoch | overall mAP | aerial-ground | same-platform | "
                         "gap | ground-ground | gap (ground only) |\n")
            handle.write("|---|---|---|---|---|---|---|\n")
            for row in rows:
                overall = (f"{row['overall_map']:.2%}"
                           if row["overall_map"] is not None else "-")
                ground = (f"{row['ground_ground']:.2%}"
                          if row["ground_ground"] is not None else "-")
                gap_gg = (f"**{row['gap_ground_only']:+.2%}**"
                          if row["gap_ground_only"] is not None else "-")
                handle.write(
                    f"| {row['epoch']} | {overall} | {row['aerial_ground']:.2%} | "
                    f"{row['same_platform']:.2%} | **{row['gap']:+.2%}** | "
                    f"{ground} | {gap_gg} |\n"
                )
            for label, info in trends.get(name, {}).items():
                if info.get("slope") is None:
                    continue
                low, high = info["capability_range"]
                handle.write(
                    f"\n- `{label}`: slope {info['slope']:+.3f} per unit of mAP, "
                    f"change {info['delta']:+.2%} over mAP {low:.2%}-{high:.2%}. "
                    f"{info['verdict']}\n"
                )
                if high - low < 0.10:
                    handle.write(
                        f"  - mAP spans only {high - low:.1%}, too narrow for the "
                        "slope to mean anything; read the level, not the direction.\n"
                    )
            handle.write("\n")

        handle.write(
            "## Reading the two curves together\n\n"
            "If AG-ReID.v2's gap falls towards zero as its model strengthens while "
            "CARGO's holds flat across its own range, then the two datasets differ "
            "in kind and not merely in difficulty: ordinary supervision is enough "
            "to erase the view gap on one and not on the other. That is the causal "
            "claim, and it does not require the two models to sit at the same mAP.\n\n"
            "If instead both curves fall and CARGO's has simply not fallen yet, the "
            "honest reading is that the gap is a function of supervision on both, "
            "and CARGO is only further from the point where it vanishes.\n\n"
            "## Two definitions of the gap\n\n"
            "The datasets are not built alike: AG-ReID.v2 has one aerial camera and "
            "therefore no aerial-aerial row, while CARGO has five. Comparing "
            "CARGO's full same-platform mean against AG-ReID.v2's would compare "
            "different quantities, so both are reported.\n\n"
            "- **gap** - aerial-ground against all same-platform pairs, "
            "pair-weighted. Uses everything a dataset offers.\n"
            "- **gap (ground only)** - aerial-ground against ground-ground alone. "
            "The one axis both datasets can produce, so this is the like-for-like "
            "comparison.\n\n"
            "If the two disagree about the shape of a curve, the ground-only "
            "definition governs any cross-dataset claim.\n"
        )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
