"""
Similarity calculation and clustering algorithms for duplicate detection.
"""

import numpy as np
import torch
from typing import List

try:
    import faiss
except ImportError:
    print(
        "ERROR: FAISS is required. Install with: pip install faiss-cpu (or faiss-gpu)"
    )
    raise ImportError("FAISS is required for similarity calculations")

from .filesystem import ImgRec


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize feature vectors for cosine similarity"""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


class UnionFind:
    """Union-Find data structure with path compression"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def get_groups(self) -> List[List[int]]:
        """Return all connected components"""
        groups_map = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups_map.setdefault(root, []).append(i)
        return list(groups_map.values())


def build_groups_clip(
    records: List[ImgRec],
    features: np.ndarray,
    valid_indices: List[int],
    threshold: float = 0.985,
    k: int = 10,
) -> List[List[int]]:
    """
    Build duplicate groups using CLIP features and FAISS k-NN search.
    Returns list of groups (each group is a list of record indices).
    """
    N = len(valid_indices)
    if N == 0:
        return []

    # L2 normalize for cosine similarity
    feats = l2_normalize(features.astype(np.float32))

    # Build FAISS index (GPU if available)
    D = feats.shape[1]
    if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
        print("Using GPU FAISS index")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, D)
    else:
        print("Using CPU FAISS index")
        index = faiss.IndexFlatIP(D)

    index.add(feats)

    # Search k nearest neighbors
    k_eff = min(k, N)
    print(f"Searching top-{k_eff} neighbors...")
    similarities, neighbor_ids = index.search(feats, k_eff)

    # Union-Find clustering
    uf = UnionFind(N)

    for i in range(N):
        for j in range(k_eff):
            neighbor = neighbor_ids[i, j]
            if neighbor == i or neighbor < 0:
                continue
            if similarities[i, j] >= threshold:
                uf.union(i, neighbor)

    # Get groups and map back to original record indices
    local_groups = uf.get_groups()
    groups = [[valid_indices[i] for i in group] for group in local_groups]

    # Return only groups with duplicates
    return [g for g in groups if len(g) > 1]
