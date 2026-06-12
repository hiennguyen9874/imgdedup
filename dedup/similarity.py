"""
Shared similarity helpers for duplicate matching.
"""

from typing import List

import numpy as np


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize feature vectors for cosine similarity."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


class UnionFind:
    """Union-Find data structure with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def get_groups(self) -> List[List[int]]:
        """Return all connected components."""
        groups_map = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups_map.setdefault(root, []).append(i)
        return list(groups_map.values())
