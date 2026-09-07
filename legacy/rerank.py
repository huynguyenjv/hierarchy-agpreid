"""
rerank.py
=========
k-reciprocal re-ranking for Re-ID (Zhong, Zheng, Cao & Li, CVPR 2017,
"Re-ranking Person Re-identification with k-reciprocal Encoding").

Why this helps (and stays within the unsupervised, label-free paradigm)
-----------------------------------------------------------------------
The raw query->gallery distance ignores the *structure* of the gallery. Two
images of the same identity tend to be in each other's k-reciprocal nearest
neighbor set even across viewpoints. Re-ranking encodes each image by its
k-reciprocal neighborhood and recomputes distances via Jaccard similarity of
those neighborhoods, then blends with the original distance:

    d_final = (1 - lambda) * d_jaccard + lambda * d_original

This is purely a post-processing step on feature distances -- it uses NO
identity labels and changes nothing about training, so it is fully compatible
with the unsupervised ASPL-MSR philosophy. It is a standard, expected component
of competitive ReID evaluation and typically adds +3-8% mAP.
"""
from __future__ import annotations

import numpy as np


def re_ranking(q_g_dist: np.ndarray, q_q_dist: np.ndarray, g_g_dist: np.ndarray,
               k1: int = 20, k2: int = 6, lambda_value: float = 0.3) -> np.ndarray:
    """
    Parameters
    ----------
    q_g_dist : (num_query, num_gallery)  original query-gallery distances
    q_q_dist : (num_query, num_query)    query-query distances
    g_g_dist : (num_gallery, num_gallery) gallery-gallery distances
    k1, k2, lambda_value : standard re-ranking hyper-parameters

    Returns
    -------
    final_dist : (num_query, num_gallery) re-ranked distances (smaller = closer)
    """
    # Assemble the full (q+g) x (q+g) symmetric distance matrix.
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0,
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    # Normalize each column to [0, 1] by its max (standard implementation).
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors of i.
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        # Expand with half-k reciprocal neighbors of each member (robustness).
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward = initial_rank[candidate, : int(np.around(k1 / 2.)) + 1]
            candidate_backward = initial_rank[candidate_forward, : int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) \
                    > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]

    # Local query expansion over the top-k2 neighbors.
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    # Build inverted index for sparse Jaccard distance computation.
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist, V, jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
