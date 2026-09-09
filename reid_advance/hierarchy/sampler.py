"""View-balanced P x K sampling (AI-04).

The cross-view term of the multi-granularity loss only ever fires on pairs that
share an identity but not a platform. Nothing guarantees a plain P x K sampler
produces any: it draws K images per identity at random, and an identity with
mostly ground shots will contribute a batch row with no aerial counterpart. The
loss would then run, the curve would look normal, and the term being tested
would be dead. This is the failure the knowledge base flags as the most common
one in this setting.

So the sampler splits each identity's quota deliberately: K/2 aerial and K/2
ground, falling back to whichever view is available when an identity cannot
fill its half, and recording how often that happens.

**All three method branches use this sampler, baseline included.** Batch
composition is not a neutral background: it changes which pairs the loss can
see at all, and the measured gap has already shown sensitivity to batch size. A
baseline drawn from a different sampler would differ from the other branches in
two ways at once - objective and batch statistics - and could not isolate
either.
"""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler

from ..cargo import binary_view_of_camera


class ViewBalancedPKSampler(Sampler):
    """P identities x K images, each identity split evenly across platforms."""

    def __init__(self, items, batch_size: int, instances_per_identity: int,
                 seed: int = 42, view_of_camera=binary_view_of_camera):
        if batch_size % instances_per_identity != 0:
            raise ValueError("batch_size must be divisible by instances_per_identity")
        if instances_per_identity % 2:
            raise ValueError(
                "instances_per_identity must be even so each identity can be "
                "split evenly between aerial and ground"
            )

        self.batch_size = batch_size
        self.instances_per_identity = instances_per_identity
        self.identities_per_batch = batch_size // instances_per_identity
        self.half = instances_per_identity // 2
        self.seed = seed
        self.epoch = 0

        # items: the dataset's (path, pid, camera_id) list, in index order.
        self.by_pid_view: dict[int, dict[int, list[int]]] = defaultdict(
            lambda: {0: [], 1: []}
        )
        for index, (_, pid, camera_id) in enumerate(items):
            self.by_pid_view[pid][view_of_camera(camera_id)].append(index)

        self.pids = sorted(self.by_pid_view)
        # Identities holding both views are the only ones that can ever supply a
        # cross-platform positive pair; the rest are padded from one view.
        self.dual_view_pids = [
            pid for pid in self.pids
            if self.by_pid_view[pid][0] and self.by_pid_view[pid][1]
        ]
        self.length = (
            len(self.pids) * instances_per_identity // batch_size
        ) * batch_size

        self.last_fallback_count = 0

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _draw(self, rng: random.Random, pid: int) -> list[int]:
        """K indices for one identity, half from each view where possible."""
        chosen: list[int] = []
        shortfall = 0
        for view in (1, 0):                      # aerial first, then ground
            pool = self.by_pid_view[pid][view]
            if len(pool) >= self.half:
                chosen.extend(rng.sample(pool, self.half))
            elif pool:
                chosen.extend(rng.choices(pool, k=self.half))
                shortfall += 1
            else:
                shortfall += self.half           # nothing of this view at all

        if len(chosen) < self.instances_per_identity:
            everything = self.by_pid_view[pid][0] + self.by_pid_view[pid][1]
            chosen.extend(
                rng.choices(everything, k=self.instances_per_identity - len(chosen))
            )
        self.last_fallback_count += shortfall
        return chosen[: self.instances_per_identity]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.last_fallback_count = 0

        order = list(self.pids)
        rng.shuffle(order)
        indices: list[int] = []
        for start in range(0, len(order), self.identities_per_batch):
            group = order[start : start + self.identities_per_batch]
            if len(group) < self.identities_per_batch:
                break
            for pid in group:
                indices.extend(self._draw(rng, pid))
        self.epoch += 1
        return iter(indices[: self.length])

    # -- diagnostics ------------------------------------------------------

    def batch_composition(self, batches: int = 5) -> list[dict]:
        """View statistics of the first batches, for the pre-training gate.

        Reports what actually matters: how many identities in a batch carry
        both platforms, and therefore how many cross-platform positive pairs
        the loss will have to work with. A batch of purely single-view
        identities makes the cross-view term a no-op.
        """
        lookup = {}
        for pid, views in self.by_pid_view.items():
            for view, indices in views.items():
                for index in indices:
                    lookup[index] = (pid, view)

        report = []
        iterator = iter(self)
        for _ in range(batches):
            batch = [next(iterator) for _ in range(self.batch_size)]
            views_by_pid: dict[int, list[int]] = defaultdict(list)
            for index in batch:
                pid, view = lookup[index]
                views_by_pid[pid].append(view)

            aerial = sum(v.count(1) for v in views_by_pid.values())
            dual = sum(
                1 for v in views_by_pid.values() if 0 in v and 1 in v
            )
            # Each dual-view identity contributes (#aerial x #ground) pairs.
            cross_pairs = sum(
                v.count(0) * v.count(1) for v in views_by_pid.values()
            )
            report.append({
                "identities": len(views_by_pid),
                "aerial_images": aerial,
                "ground_images": self.batch_size - aerial,
                "dual_view_identities": dual,
                "cross_platform_pairs": cross_pairs,
            })
        return report
