"""
Duplicate pair generation and decision policy.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

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


def decide_pair(
    same_sha256: bool,
    phash_dist: Optional[int],
    cosine: Optional[float],
    thresholds: MatchThresholds,
) -> Tuple[str, str]:
    """Apply the requested duplicate policy to one image pair."""
    if same_sha256:
        return "duplicate", "same_sha256"

    if phash_dist is not None and phash_dist <= thresholds.phash_auto_distance:
        return "duplicate", f"phash_distance<={thresholds.phash_auto_distance}"

    if cosine is not None and cosine >= thresholds.cosine_auto:
        return "duplicate", f"cosine>={thresholds.cosine_auto:.2f}"

    if (
        cosine is not None
        and cosine >= thresholds.cosine_verify
        and phash_dist is not None
        and phash_dist <= thresholds.phash_verify_distance
    ):
        return (
            "duplicate",
            f"cosine>={thresholds.cosine_verify:.2f}_and_phash<={thresholds.phash_verify_distance}",
        )

    if (
        cosine is not None
        and thresholds.cosine_review <= cosine < thresholds.cosine_verify
    ):
        return "review", f"{thresholds.cosine_review:.2f}<=cosine<{thresholds.cosine_verify:.2f}"

    return "ignore", "not_confident"


def _make_pair(
    records: List[ImgRec],
    a: int,
    b: int,
    cosine: Optional[float],
    thresholds: MatchThresholds,
) -> DuplicatePair:
    if a > b:
        a, b = b, a

    left = records[a]
    right = records[b]
    same_sha = bool(left.sha256 and right.sha256 and left.sha256 == right.sha256)
    phash_dist = phash_distance(left.phash, right.phash)
    decision, reason = decide_pair(same_sha, phash_dist, cosine, thresholds)

    return DuplicatePair(
        a=a,
        b=b,
        cosine=cosine,
        phash_distance=phash_dist,
        same_sha256=same_sha,
        decision=decision,
        reason=reason,
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


def _iter_phash_pairs(records: List[ImgRec], max_distance: int) -> Iterable[Tuple[int, int]]:
    hashed = [(idx, record.phash) for idx, record in enumerate(records) if record.phash]
    for pos, (left_idx, left_hash) in enumerate(hashed):
        for right_idx, right_hash in hashed[pos + 1 :]:
            distance = phash_distance(left_hash, right_hash)
            if distance is not None and distance <= max_distance:
                yield left_idx, right_idx


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
) -> Tuple[List[DuplicatePair], List[DuplicatePair], List[List[int]]]:
    """
    Find duplicate and review-only pairs, then group duplicates.

    Review-only pairs are intentionally excluded from union-find groups so they
    cannot be auto-deleted.
    """
    pairs: Dict[Tuple[int, int], DuplicatePair] = {}

    for left, right in _iter_exact_sha_pairs(records):
        _upsert_pair(pairs, _make_pair(records, left, right, None, thresholds))

    for left, right in _iter_phash_pairs(records, thresholds.phash_auto_distance):
        _upsert_pair(pairs, _make_pair(records, left, right, None, thresholds))

    if len(valid_indices) > 0 and features.size > 0:
        feats = l2_normalize(features.astype(np.float32))
        feature_positions = {record_idx: pos for pos, record_idx in enumerate(valid_indices)}

        for left, right in _iter_phash_pairs(records, thresholds.phash_verify_distance):
            left_pos = feature_positions.get(left)
            right_pos = feature_positions.get(right)
            cosine = None
            if left_pos is not None and right_pos is not None:
                cosine = float(np.dot(feats[left_pos], feats[right_pos]))
            _upsert_pair(pairs, _make_pair(records, left, right, cosine, thresholds))

        index = _build_faiss_index(feats)
        index.add(feats)

        k_eff = min(k, len(valid_indices))
        print(f"Searching top-{k_eff} neighbors...")
        similarities, neighbor_ids = index.search(feats, k_eff)

        for local_idx, record_idx in enumerate(valid_indices):
            for score, neighbor_local_idx in zip(similarities[local_idx], neighbor_ids[local_idx]):
                if neighbor_local_idx < 0 or neighbor_local_idx == local_idx:
                    continue

                neighbor_record_idx = valid_indices[neighbor_local_idx]
                if record_idx > neighbor_record_idx:
                    continue

                score = float(score)
                if score < thresholds.cosine_review:
                    continue

                pair = _make_pair(records, record_idx, neighbor_record_idx, score, thresholds)
                if pair.decision != "ignore":
                    _upsert_pair(pairs, pair)

    duplicate_pairs = [pair for pair in pairs.values() if pair.decision == "duplicate"]
    review_pairs = [pair for pair in pairs.values() if pair.decision == "review"]

    uf = UnionFind(len(records))
    for pair in duplicate_pairs:
        uf.union(pair.a, pair.b)

    groups = [group for group in uf.get_groups() if len(group) > 1]
    return duplicate_pairs, review_pairs, groups
