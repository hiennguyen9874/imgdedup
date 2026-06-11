"""
Report generation and representative selection functionality.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .filesystem import ImgRec
from .matching import DuplicatePair


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
    elif policy == "newest":
        return max(group, key=lambda i: records[i].mtime)
    elif policy == "oldest":
        return min(group, key=lambda i: records[i].mtime)
    else:
        raise ValueError(f"Unknown keep policy: {policy}")


def _pair_key(left: int, right: int) -> Tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _best_pair_for_duplicate(
    keep_idx: int,
    duplicate_idx: int,
    duplicate_pairs: List[DuplicatePair],
) -> Optional[DuplicatePair]:
    pair_by_key = {_pair_key(pair.a, pair.b): pair for pair in duplicate_pairs}
    direct = pair_by_key.get(_pair_key(keep_idx, duplicate_idx))
    if direct is not None:
        return direct

    related = [
        pair
        for pair in duplicate_pairs
        if pair.a == duplicate_idx or pair.b == duplicate_idx
    ]
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

    for group in groups:
        rep_idx = pick_representative(group, records, keep_policy)
        duplicates = []

        for idx in group:
            if idx == rep_idx:
                continue

            pair = _best_pair_for_duplicate(rep_idx, idx, duplicate_pairs)
            duplicates.append(
                {
                    "path": records[idx].path,
                    "reason": pair.reason if pair else "grouped_duplicate",
                    "cosine": pair.cosine if pair else None,
                    "phash_distance": pair.phash_distance if pair else None,
                    "same_sha256": pair.same_sha256 if pair else False,
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
