"""
Duplicate pair generation and decision policy.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

try:
    import faiss
except ImportError:
    print("ERROR: FAISS is required. Install with: pip install faiss-cpu (or faiss-gpu)")
    raise ImportError("FAISS is required for similarity calculations")

from .filesystem import ImgRec
from .hashing import phash_distance
from .similarity import UnionFind, l2_normalize


@dataclass(frozen=True)
class MatchThresholds:
    """Duplicate decision thresholds."""

    cosine_auto: float = 0.97
    cosine_verify: float = 0.90
    cosine_review: float = 0.85
    phash_auto_distance: int = 4
    phash_verify_distance: int = 8


@dataclass
class DuplicatePair:
    """Scored relationship between two image records."""

    a: int
    b: int
    cosine: Optional[float]
    phash_distance: Optional[int]
    same_sha256: bool
    decision: str
    reason: str
    confidence: str


def decide_pair(
    same_sha256: bool,
    phash_dist: Optional[int],
    cosine: Optional[float],
    thresholds: MatchThresholds,
) -> Tuple[str, str, str]:
    """Apply the requested duplicate policy to one image pair."""
    if same_sha256:
        return "duplicate", "same_sha256", "exact"

    if phash_dist is not None and phash_dist <= thresholds.phash_auto_distance:
        return "duplicate", f"phash_distance<={thresholds.phash_auto_distance}", "high"

    if cosine is not None and cosine >= thresholds.cosine_auto:
        return "duplicate", f"cosine>={thresholds.cosine_auto:.2f}", "high"

    if (
        cosine is not None
        and cosine >= thresholds.cosine_verify
        and phash_dist is not None
        and phash_dist <= thresholds.phash_verify_distance
    ):
        return (
            "duplicate",
            f"cosine>={thresholds.cosine_verify:.2f}_and_phash<={thresholds.phash_verify_distance}",
            "verified",
        )

    if (
        cosine is not None
        and thresholds.cosine_review <= cosine < thresholds.cosine_verify
    ):
        return "review", f"{thresholds.cosine_review:.2f}<=cosine<{thresholds.cosine_verify:.2f}", "review"

    return "ignore", "not_confident", "ignored"


def _same_parent_folder(records: List[ImgRec], a: int, b: int) -> bool:
    return os.path.dirname(records[a].path) == os.path.dirname(records[b].path)


def _make_pair(
    records: List[ImgRec],
    a: int,
    b: int,
    cosine: Optional[float],
    thresholds: MatchThresholds,
    phash_dist: Optional[int] = None,
) -> DuplicatePair:
    if a > b:
        a, b = b, a

    left = records[a]
    right = records[b]
    same_sha = bool(left.sha256 and right.sha256 and left.sha256 == right.sha256)
    if phash_dist is None:
        phash_dist = phash_distance(left.phash, right.phash)
    decision, reason, confidence = decide_pair(same_sha, phash_dist, cosine, thresholds)

    return DuplicatePair(
        a=a,
        b=b,
        cosine=cosine,
        phash_distance=phash_dist,
        same_sha256=same_sha,
        decision=decision,
        reason=reason,
        confidence=confidence,
    )


def _upsert_pair(pairs: Dict[Tuple[int, int], DuplicatePair], pair: DuplicatePair) -> None:
    """Keep the strongest evidence when the same pair is found multiple ways."""
    key = (pair.a, pair.b)
    existing = pairs.get(key)
    if existing is None:
        pairs[key] = pair
        return

    rank = {"ignore": 0, "review": 1, "duplicate": 2}
    if rank[pair.decision] > rank[existing.decision]:
        pairs[key] = pair
    elif rank[pair.decision] == rank[existing.decision]:
        existing_score = existing.cosine if existing.cosine is not None else -1.0
        pair_score = pair.cosine if pair.cosine is not None else -1.0
        if pair_score > existing_score:
            pairs[key] = pair


def _iter_exact_sha_pairs(records: List[ImgRec]) -> Iterable[Tuple[int, int]]:
    by_hash: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        if record.sha256:
            by_hash.setdefault(record.sha256, []).append(idx)

    for indices in by_hash.values():
        if len(indices) < 2:
            continue
        for pos, left in enumerate(indices):
            for right in indices[pos + 1 :]:
                yield left, right


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _BKTreeNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: Dict[int, "_BKTreeNode"] = {}


class _BKTree:
    """Exact metric-radius index for pHash Hamming distance lookups."""

    def __init__(self) -> None:
        self.root: Optional[_BKTreeNode] = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKTreeNode(value, index)
            return

        node = self.root
        while True:
            distance = _hamming_distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return

            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKTreeNode(value, index)
                return
            node = child

    def query(self, value: int, max_distance: int) -> Iterable[Tuple[int, int]]:
        if self.root is None:
            return

        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = _hamming_distance(value, node.value)
            if distance <= max_distance:
                for index in node.indices:
                    yield index, distance

            lower = distance - max_distance
            upper = distance + max_distance
            for edge_distance, child in node.children.items():
                if lower <= edge_distance <= upper:
                    pending.append(child)


def _iter_phash_pairs(records: List[ImgRec], max_distance: int) -> Iterable[Tuple[int, int, int]]:
    """Yield all pHash pairs within max_distance using an exact BK-tree search."""
    tree = _BKTree()
    hashes: List[Tuple[int, int]] = []

    for idx, record in enumerate(records):
        if not record.phash:
            continue
        try:
            value = int(record.phash, 16)
        except ValueError:
            continue
        hashes.append((idx, value))
        tree.add(value, idx)

    seen: Set[Tuple[int, int]] = set()
    for left_idx, left_hash in hashes:
        for right_idx, distance in tree.query(left_hash, max_distance):
            if left_idx == right_idx:
                continue
            key = (left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx)
            if key in seen:
                continue
            seen.add(key)
            yield key[0], key[1], distance


def _split_group_agglomerative(
    group: List[int],
    feats: np.ndarray,
    feature_positions: Dict[int, int],
    cosine_threshold: float,
    linkage: str,
) -> List[List[int]]:
    if len(group) < 3:
        return [group]

    positions = []
    for record_idx in group:
        position = feature_positions.get(record_idx)
        if position is None:
            return [group]
        positions.append(position)

    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError(
            "--grouping agglomerative requires scikit-learn to be installed"
        ) from exc

    group_feats = feats[positions]
    distances = 1.0 - np.clip(group_feats @ group_feats.T, -1.0, 1.0)
    np.fill_diagonal(distances, 0.0)

    kwargs = {
        "n_clusters": None,
        "distance_threshold": 1.0 - cosine_threshold,
        "linkage": linkage,
    }
    try:
        clusterer = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        clusterer = AgglomerativeClustering(affinity="precomputed", **kwargs)

    labels = clusterer.fit_predict(distances)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for record_idx, label in zip(group, labels):
        clusters[int(label)].append(record_idx)

    return [cluster for cluster in clusters.values() if len(cluster) > 1]


def _split_groups_agglomerative(
    groups: List[List[int]],
    feats: np.ndarray,
    feature_positions: Dict[int, int],
    cosine_threshold: float,
    linkage: str,
) -> List[List[int]]:
    split_groups = []
    for group in groups:
        split_groups.extend(
            _split_group_agglomerative(
                group, feats, feature_positions, cosine_threshold, linkage
            )
        )
    return split_groups


def _build_faiss_index(feats: np.ndarray):
    dim = feats.shape[1]
    if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
        print("Using GPU FAISS index")
        res = faiss.StandardGpuResources()
        return faiss.GpuIndexFlatIP(res, dim)

    print("Using CPU FAISS index")
    return faiss.IndexFlatIP(dim)


def find_duplicate_pairs(
    records: List[ImgRec],
    features: np.ndarray,
    valid_indices: List[int],
    thresholds: MatchThresholds,
    k: int = 50,
    faiss_index_path: Optional[str] = None,
    show_timings: bool = True,
    show_progress: bool = False,
    cross_folder_only: bool = False,
    grouping_method: str = "connected",
    agglomerative_linkage: str = "complete",
    agglomerative_cosine_threshold: Optional[float] = None,
) -> Tuple[List[DuplicatePair], List[DuplicatePair], List[List[int]]]:
    """
    Find duplicate and review-only pairs, then group duplicates.

    Review-only pairs are intentionally excluded from union-find groups so they
    cannot be auto-deleted.
    """
    pairs: Dict[Tuple[int, int], DuplicatePair] = {}
    timings: Dict[str, float] = {}
    feats = np.empty((0, 0), dtype=np.float32)
    feature_positions: Dict[int, int] = {}

    started = time.perf_counter()
    for left, right in _iter_exact_sha_pairs(records):
        if cross_folder_only and _same_parent_folder(records, left, right):
            continue
        _upsert_pair(pairs, _make_pair(records, left, right, None, thresholds))
    timings["sha256 pairs"] = time.perf_counter() - started

    if len(valid_indices) > 0 and features.size > 0:
        started = time.perf_counter()
        feats = l2_normalize(features.astype(np.float32))
        feature_positions = {record_idx: pos for pos, record_idx in enumerate(valid_indices)}
        timings["feature normalize"] = time.perf_counter() - started

        started = time.perf_counter()
        for left, right, phash_dist in _iter_phash_pairs(records, thresholds.phash_verify_distance):
            if cross_folder_only and _same_parent_folder(records, left, right):
                continue
            left_pos = feature_positions.get(left)
            right_pos = feature_positions.get(right)
            cosine = None
            if left_pos is not None and right_pos is not None:
                cosine = float(np.dot(feats[left_pos], feats[right_pos]))
            _upsert_pair(pairs, _make_pair(records, left, right, cosine, thresholds, phash_dist))
        timings["pHash BK-tree"] = time.perf_counter() - started

        started = time.perf_counter()
        index = _build_faiss_index(feats)
        index.add(feats)
        timings["FAISS build"] = time.perf_counter() - started

        if faiss_index_path:
            started = time.perf_counter()
            os.makedirs(os.path.dirname(os.path.abspath(faiss_index_path)), exist_ok=True)
            try:
                index_to_write = faiss.index_gpu_to_cpu(index)
            except Exception:
                index_to_write = index
            faiss.write_index(index_to_write, faiss_index_path)
            timings["FAISS write"] = time.perf_counter() - started

        k_eff = min(k, len(valid_indices))
        print(f"Searching top-{k_eff} neighbors...")
        search_seconds = 0.0
        classification_seconds = 0.0
        batch_size = 1024
        batches = range(0, len(valid_indices), batch_size)
        if show_progress:
            batches = tqdm(
                batches,
                total=(len(valid_indices) + batch_size - 1) // batch_size,
                desc="FAISS search/classify",
                unit="batch",
            )

        for start in batches:
            end = min(start + batch_size, len(valid_indices))
            started = time.perf_counter()
            similarities, neighbor_ids = index.search(feats[start:end], k_eff)
            search_seconds += time.perf_counter() - started

            started = time.perf_counter()
            for offset, record_idx in enumerate(valid_indices[start:end]):
                local_idx = start + offset
                for score, neighbor_local_idx in zip(similarities[offset], neighbor_ids[offset]):
                    if neighbor_local_idx < 0 or neighbor_local_idx == local_idx:
                        continue

                    neighbor_record_idx = valid_indices[neighbor_local_idx]
                    if record_idx > neighbor_record_idx:
                        continue
                    if cross_folder_only and _same_parent_folder(
                        records, record_idx, neighbor_record_idx
                    ):
                        continue

                    score = float(score)
                    if score < thresholds.cosine_review:
                        break

                    pair = _make_pair(records, record_idx, neighbor_record_idx, score, thresholds)
                    if pair.decision != "ignore":
                        _upsert_pair(pairs, pair)
            classification_seconds += time.perf_counter() - started

        timings["FAISS search"] = search_seconds
        timings["pair classification"] = classification_seconds
    else:
        started = time.perf_counter()
        for left, right, phash_dist in _iter_phash_pairs(records, thresholds.phash_verify_distance):
            if cross_folder_only and _same_parent_folder(records, left, right):
                continue
            if phash_dist <= thresholds.phash_auto_distance:
                _upsert_pair(pairs, _make_pair(records, left, right, None, thresholds, phash_dist))
        timings["pHash BK-tree"] = time.perf_counter() - started

    started = time.perf_counter()
    duplicate_pairs = [pair for pair in pairs.values() if pair.decision == "duplicate"]
    review_pairs = [pair for pair in pairs.values() if pair.decision == "review"]

    uf = UnionFind(len(records))
    for pair in duplicate_pairs:
        uf.union(pair.a, pair.b)

    groups = [group for group in uf.get_groups() if len(group) > 1]
    timings["grouping"] = time.perf_counter() - started

    if grouping_method == "agglomerative":
        started = time.perf_counter()
        if feats.size == 0 or not feature_positions:
            raise ValueError("Agglomerative grouping requires valid image features")
        cosine_threshold = (
            thresholds.cosine_auto
            if agglomerative_cosine_threshold is None
            else agglomerative_cosine_threshold
        )
        groups = _split_groups_agglomerative(
            groups,
            feats,
            feature_positions,
            cosine_threshold,
            agglomerative_linkage,
        )
        timings["agglomerative split"] = time.perf_counter() - started

    if show_timings:
        print("Step 4 timings:")
        for name, elapsed in timings.items():
            print(f"  {name}: {elapsed:.2f}s")

    return duplicate_pairs, review_pairs, groups
