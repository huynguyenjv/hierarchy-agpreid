"""BA-02/BA-03: audit the AG-ReID.v2 attribute annotations before tree design.

Three questions decide whether the semantic hierarchy is viable at all:

1. Coverage - can every training identity be resolved to a MAT row?
2. Identity-level stability - do all images of one identity share one attribute
   vector?  The knowledge base assumes attributes are annotated per identity,
   which is what makes a branch view-invariant.  If that is false, the whole
   "hierarchy bridges the aerial-ground gap" argument collapses.
3. Usability per group - entropy (is the split balanced?) and mutual
   information (are two groups redundant?), the criteria for picking levels.

Usage:
    venv/Scripts/python.exe tools/audit_attributes.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_advance.attributes import ATTRIBUTE_GROUPS
from reid_advance.data import _scan_images, load_track_attributes, track_key
from reid_advance.hierarchy import platform_of_path
from reid_advance.identity import FILENAME_PATTERN

IGNORE = -100
GROUPS = list(ATTRIBUTE_GROUPS)


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy in bits over the observed class distribution."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    """MI in bits between two label vectors, ignoring rows unknown in either."""
    valid = (left != IGNORE) & (right != IGNORE)
    if valid.sum() == 0:
        return 0.0
    left, right = left[valid], right[valid]
    joint = Counter(zip(left.tolist(), right.tolist()))
    total = len(left)
    left_counts = Counter(left.tolist())
    right_counts = Counter(right.tolist())
    result = 0.0
    for (a, b), count in joint.items():
        p_ab = count / total
        p_a = left_counts[a] / total
        p_b = right_counts[b] / total
        result += p_ab * np.log2(p_ab / (p_a * p_b))
    return float(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="AG-ReID.v2")
    parser.add_argument("--mat", default="AG-ReID.v2/qut_attribute_v8.mat")
    parser.add_argument("--split", default="train_all")
    parser.add_argument("--output", default="docs/attribute_audit.md")
    args = parser.parse_args()

    print(f"Loading MAT: {args.mat}")
    # ignore_unknown=False so the audit can *see* the unknown rate rather than
    # silently folding it into the ignore index.
    mapping = load_track_attributes(args.mat, ignore_unknown=False)
    print(f"MAT rows: {len(mapping):,}")

    split_dir = os.path.join(args.data_root, args.split)
    items = _scan_images(split_dir, FILENAME_PATTERN)
    print(f"Images in {args.split}: {len(items):,}")

    # --- coverage + identity-level stability -------------------------------
    vectors_per_identity: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    platforms_per_identity: dict[int, set[str]] = defaultdict(set)
    missing_images = 0
    missing_keys: set[str] = set()
    identity_rows: dict[int, np.ndarray] = {}

    for path, pid in items:
        key = track_key(path)
        platforms_per_identity[pid].add(platform_of_path(path))
        vector = mapping.get(key)
        if vector is None:
            missing_images += 1
            missing_keys.add(key)
            continue
        as_tuple = tuple(int(v) for v in vector.tolist())
        vectors_per_identity[pid].add(as_tuple)
        identity_rows.setdefault(pid, np.asarray(as_tuple, dtype=np.int64))

    all_pids = {pid for _, pid in items}
    covered = set(vectors_per_identity)
    unstable = {pid: v for pid, v in vectors_per_identity.items() if len(v) > 1}
    both_views = {
        pid for pid, platforms in platforms_per_identity.items()
        if "aerial" in platforms and platforms & {"wearable", "cctv"}
    }

    print(f"\nIdentities in {args.split}: {len(all_pids):,}")
    print(f"  resolved to a MAT row : {len(covered):,} "
          f"({len(covered) / max(len(all_pids), 1):.1%})")
    print(f"  images without a row  : {missing_images:,}")
    print(f"  identities with >1 attribute vector : {len(unstable):,}")
    print(f"  identities holding BOTH aerial and ground images : "
          f"{len(both_views):,} ({len(both_views) / max(len(all_pids), 1):.1%})")

    if unstable:
        print("  !! identity-level stability assumption VIOLATED")
    else:
        print("  OK identity-level stability assumption holds")

    # --- per-group entropy / unknown rate ----------------------------------
    matrix = np.full((len(identity_rows), len(GROUPS)), IGNORE, dtype=np.int64)
    for row, (_, vector) in enumerate(sorted(identity_rows.items())):
        matrix[row] = vector

    print(f"\nPer-group statistics over {matrix.shape[0]:,} identities")
    print(f"{'group':<12}{'classes':>8}{'unknown%':>10}{'entropy':>9}"
          f"{'maxclass%':>11}  distribution")
    stats = []
    for index, name in enumerate(GROUPS):
        fields = ATTRIBUTE_GROUPS[name]
        column = matrix[:, index]
        unknown_index = len(fields) - 1
        unknown = int((column == unknown_index).sum()) + int((column == IGNORE).sum())
        known = column[(column >= 0) & (column != unknown_index)]
        counts = np.bincount(known, minlength=len(fields))[:unknown_index]
        bits = entropy(counts)
        top = counts.max() / counts.sum() if counts.sum() else 1.0
        stats.append({
            "group": name, "classes": len(fields),
            "unknown_pct": 100 * unknown / len(column),
            "entropy": bits, "max_class_pct": 100 * top,
            "counts": counts.tolist(),
            "labels": [f[len(name):] for f in fields[:unknown_index]],
        })
        print(f"{name:<12}{len(fields):>8}{100*unknown/len(column):>9.1f}%"
              f"{bits:>9.2f}{100*top:>10.1f}%  {counts.tolist()}")

    # --- mutual information ------------------------------------------------
    size = len(GROUPS)
    mi = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mi[i, j] = mutual_information(matrix[:, i], matrix[:, j])

    print("\nHighest mutual information pairs (bits):")
    pairs = sorted(
        ((mi[i, j], GROUPS[i], GROUPS[j]) for i in range(size) for j in range(i + 1, size)),
        reverse=True,
    )
    for value, a, b in pairs[:8]:
        print(f"  {a:<12} x {b:<12} {value:.3f}")

    # --- report ------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("# BA-02/BA-03 attribute audit\n\n")
        handle.write(f"- MAT: `{args.mat}` ({len(mapping):,} rows)\n")
        handle.write(f"- Split: `{args.split}` ({len(items):,} images, "
                     f"{len(all_pids):,} identities, P+T+A scheme)\n\n")

        handle.write("## 1. Coverage and identity-level stability\n\n")
        handle.write("| check | value |\n|---|---|\n")
        handle.write(f"| identities resolved to a MAT row | {len(covered):,} / "
                     f"{len(all_pids):,} ({len(covered)/max(len(all_pids),1):.1%}) |\n")
        handle.write(f"| images without a MAT row | {missing_images:,} |\n")
        handle.write(f"| distinct unmatched track keys | {len(missing_keys):,} |\n")
        handle.write(f"| identities with >1 attribute vector | {len(unstable):,} |\n")
        handle.write(f"| identities with both aerial and ground images | "
                     f"{len(both_views):,} ({len(both_views)/max(len(all_pids),1):.1%}) |\n\n")
        verdict = "VIOLATED" if unstable else "HOLDS"
        handle.write(f"**Identity-level stability assumption: {verdict}.** ")
        handle.write(
            "Every image of an identity shares one attribute vector, so a branch "
            "of the tree is identical for the aerial and the ground images of "
            "that person - this is what lets the hierarchy act as a view-invariant "
            "bridge (KB 2.2).\n\n" if not unstable else
            "Attribute vectors disagree within an identity; majority-vote per "
            "identity before building the tree, and report this in the paper.\n\n"
        )
        handle.write(
            "The last row also matters for the sampler: only identities holding "
            "both aerial and ground images can ever produce a cross-view positive "
            "pair, which is what the beta_cross term of CV-HWC reweights.\n\n"
        )

        handle.write("## 2. Per-group statistics (identity level)\n\n")
        handle.write("| group | classes | unknown % | entropy (bits) | largest class % | distribution |\n")
        handle.write("|---|---|---|---|---|---|\n")
        for row in stats:
            handle.write(
                f"| {row['group']} | {row['classes']} | {row['unknown_pct']:.1f}% | "
                f"{row['entropy']:.2f} | {row['max_class_pct']:.1f}% | "
                f"{row['counts']} |\n"
            )
        handle.write(
            "\nA group is a poor tree level when the unknown rate is high, the "
            "entropy is near zero, or one class covers almost everything "
            "(KB 6.1 criterion 2).\n\n"
        )

        handle.write("## 3. Mutual information between groups (bits)\n\n")
        handle.write("| | " + " | ".join(GROUPS) + " |\n")
        handle.write("|---" * (size + 1) + "|\n")
        for i, name in enumerate(GROUPS):
            handle.write(f"| **{name}** | " +
                         " | ".join(f"{mi[i, j]:.2f}" for j in range(size)) + " |\n")
        handle.write("\nTop redundant pairs:\n\n")
        for value, a, b in pairs[:8]:
            handle.write(f"- `{a}` x `{b}`: {value:.3f}\n")
        handle.write(
            "\nAvoid stacking two high-MI groups as consecutive levels; the "
            "second one would barely subdivide the first (KB 6.1 criterion 3).\n"
        )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
