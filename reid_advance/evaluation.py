"""
evaluate.py
===========
Standard Re-ID evaluation: mAP, Rank-1, Rank-5, Rank-10.

Protocol
--------
* Extract L2-normalized encoder features for query and gallery.
* Distance = 1 - cosine similarity. Because features are unit-norm, ranking by
  this distance is equivalent to ranking by Euclidean distance.
* For each query, gallery samples with the SAME identity AND the SAME camera id
  are "junk" and are removed before computing metrics (standard Re-ID rule:
  re-identifying a person in the same camera view is trivial and not the task).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(model, loader, device, flip_tta: bool = False):
    model.eval()
    feats, pids, camids = [], [], []
    for imgs, pid, camid in tqdm(loader, desc="extract", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        camera_tensor = camid.to(device, non_blocking=True)
        if getattr(model, "accepts_camera_ids", False):
            f = model(imgs, camera_tensor)
        else:
            f = model(imgs)             # L2-normalized embedding
        if flip_tta:
            # Average the embedding with that of the horizontally-flipped image
            # and re-normalize. A free, label-free robustness boost: the model
            # sees both left-right orientations of the same crop.
            flipped = torch.flip(imgs, dims=[3])
            if getattr(model, "accepts_camera_ids", False):
                f_flip = model(flipped, camera_tensor)
            else:
                f_flip = model(flipped)
            f = F.normalize(f + f_flip, dim=1)
        feats.append(f.cpu())
        pids.append(pid)
        camids.append(camid)
    return (
        torch.cat(feats, dim=0),
        torch.cat(pids, dim=0).numpy(),
        torch.cat(camids, dim=0).numpy(),
    )


# --------------------------------------------------------------------------- #
#  Metric computation
# --------------------------------------------------------------------------- #
def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Returns (cmc, mAP). `distmat` is (num_query, num_gallery), smaller = closer.
    Junk (same pid + same camid) matches are removed per query.
    """
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)            # ascending distance
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc, all_AP, num_valid_q = [], [], 0.0
    for q_idx in range(num_q):
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]
        order = indices[q_idx]

        # Remove gallery samples with same pid AND same camera (junk).
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # This query has no valid (cross-camera) ground-truth match.
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        cmc_at_rank = np.empty(max_rank, dtype=np.float32)
        available = min(max_rank, len(cmc))
        cmc_at_rank[:available] = cmc[:available]
        if available < max_rank:
            cmc_at_rank[available:] = cmc[-1]
        all_cmc.append(cmc_at_rank)
        num_valid_q += 1.0

        # Average Precision for this query.
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    assert num_valid_q > 0, "No valid query (check labels / camera ids)."
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    return cmc, mAP
