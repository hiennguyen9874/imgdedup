"""Deterministic representative selection from image embeddings."""

from typing import List

import numpy as np


def _validate(features: np.ndarray, count: int) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if count < 1:
        raise ValueError("selection count must be at least 1")
    if count > len(features):
        raise ValueError("selection count cannot exceed the number of features")


def _nearest_unique(features: np.ndarray, centers: np.ndarray) -> List[int]:
    selected = []
    used = set()
    for center in centers:
        distances = np.sum((features - center) ** 2, axis=1)
        for idx in np.argsort(distances, kind="stable"):
            value = int(idx)
            if value not in used:
                used.add(value)
                selected.append(value)
                break
    return selected


def select_kmeans(features: np.ndarray, count: int, seed: int) -> List[int]:
    """Select the image nearest each MiniBatchKMeans centroid."""
    _validate(features, count)
    if count == len(features):
        return list(range(len(features)))

    from sklearn.cluster import MiniBatchKMeans

    model = MiniBatchKMeans(
        n_clusters=count,
        random_state=seed,
        batch_size=min(1024, len(features)),
        n_init=3,
    )
    model.fit(features)
    return _nearest_unique(features, model.cluster_centers_)


def select_farthest(features: np.ndarray, count: int, seed: int) -> List[int]:
    """Greedily maximize coverage using squared Euclidean distance."""
    _validate(features, count)
    if count == len(features):
        return list(range(len(features)))

    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(features)))
    selected = [first]
    min_distances = np.sum((features - features[first]) ** 2, axis=1)
    min_distances[first] = -1.0

    while len(selected) < count:
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        distances = np.sum((features - features[next_idx]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected] = -1.0
    return selected


def select_hybrid(
    features: np.ndarray, count: int, seed: int, oversample_factor: int = 3
) -> List[int]:
    """Build a centroid candidate pool, then maximize coverage within it."""
    _validate(features, count)
    if count == len(features):
        return list(range(len(features)))

    candidate_count = min(len(features), max(count, count * oversample_factor))
    candidates = select_kmeans(features, candidate_count, seed)
    local_indices = select_farthest(features[candidates], count, seed)
    return [candidates[idx] for idx in local_indices]


def select_representatives(
    features: np.ndarray, count: int, method: str = "hybrid", seed: int = 42
) -> List[int]:
    """Select feature-row indices with the requested strategy."""
    if method == "kmeans":
        return select_kmeans(features, count, seed)
    if method == "farthest":
        return select_farthest(features, count, seed)
    if method == "hybrid":
        return select_hybrid(features, count, seed)
    raise ValueError(f"Unknown selection method: {method}")
