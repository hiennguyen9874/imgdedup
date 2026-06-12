"""
Report generation and representative selection functionality.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .filesystem import ImgRec
from .matching import DuplicatePair


def _pixel_count(record: ImgRec) -> int:
    if record.width is None or record.height is None:
        return 0
    return record.width * record.height


def pick_representative(group: List[int], records: List[ImgRec], policy: str) -> int:
    """
    Select which file to keep from a duplicate group based on policy.
    Returns the index of the file to keep.
    """
    if policy == "lexi":
        return min(group, key=lambda i: records[i].path)
    elif policy == "smallest":
        return min(group, key=lambda i: records[i].size)
    elif policy == "largest":
        return max(group, key=lambda i: records[i].size)
    elif policy == "highest-resolution":
        return max(group, key=lambda i: (_pixel_count(records[i]), records[i].size))
    elif policy == "newest":
        return max(group, key=lambda i: records[i].mtime)
    elif policy == "oldest":
        return min(group, key=lambda i: records[i].mtime)
    else:
        raise ValueError(f"Unknown keep policy: {policy}")


def _pair_key(left: int, right: int) -> Tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _build_pair_indexes(
    duplicate_pairs: List[DuplicatePair],
) -> Tuple[Dict[Tuple[int, int], DuplicatePair], Dict[int, List[DuplicatePair]]]:
    pair_by_key = {}
    pairs_by_image = defaultdict(list)
    for pair in duplicate_pairs:
        pair_by_key[_pair_key(pair.a, pair.b)] = pair
        pairs_by_image[pair.a].append(pair)
        pairs_by_image[pair.b].append(pair)
    return pair_by_key, pairs_by_image


def _best_pair_for_duplicate(
    keep_idx: int,
    duplicate_idx: int,
    pair_by_key: Dict[Tuple[int, int], DuplicatePair],
    pairs_by_image: Dict[int, List[DuplicatePair]],
) -> Optional[DuplicatePair]:
    direct = pair_by_key.get(_pair_key(keep_idx, duplicate_idx))
    if direct is not None:
        return direct

    related = pairs_by_image.get(duplicate_idx, [])
    if not related:
        return None

    return max(related, key=lambda pair: pair.cosine if pair.cosine is not None else -1.0)


def _pair_to_report(pair: DuplicatePair, records: List[ImgRec]) -> Dict:
    return {
        "a": records[pair.a].path,
        "b": records[pair.b].path,
        "cosine": pair.cosine,
        "phash_distance": pair.phash_distance,
        "same_sha256": pair.same_sha256,
        "decision": pair.decision,
        "reason": pair.reason,
        "confidence": pair.confidence,
    }


def make_report(
    records: List[ImgRec],
    groups: List[List[int]],
    keep_policy: str,
    duplicate_pairs: Optional[List[DuplicatePair]] = None,
    review_pairs: Optional[List[DuplicatePair]] = None,
) -> Dict:
    """
    Generate JSON report with duplicate groups, decisions, and review-only pairs.
    """
    duplicate_pairs = duplicate_pairs or []
    review_pairs = review_pairs or []
    report_groups = []
    total_duplicates = 0
    pair_by_key, pairs_by_image = _build_pair_indexes(duplicate_pairs)

    for group in groups:
        rep_idx = pick_representative(group, records, keep_policy)
        duplicates = []

        for idx in group:
            if idx == rep_idx:
                continue

            pair = _best_pair_for_duplicate(rep_idx, idx, pair_by_key, pairs_by_image)
            duplicates.append(
                {
                    "path": records[idx].path,
                    "reason": pair.reason if pair else "grouped_duplicate",
                    "cosine": pair.cosine if pair else None,
                    "phash_distance": pair.phash_distance if pair else None,
                    "same_sha256": pair.same_sha256 if pair else False,
                    "confidence": pair.confidence if pair else "grouped",
                }
            )

        total_duplicates += len(duplicates)
        report_groups.append({"keep": records[rep_idx].path, "duplicates": duplicates})

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(records),
        "duplicate_groups": len(groups),
        "total_duplicates": total_duplicates,
        "review_only_pairs": len(review_pairs),
        "keep_policy": keep_policy,
        "groups": report_groups,
        "review_only": [_pair_to_report(pair, records) for pair in review_pairs],
    }

    return report
