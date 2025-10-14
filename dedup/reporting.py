"""
Report generation and representative selection functionality.
"""

from datetime import datetime
from typing import List, Dict

from .filesystem import ImgRec


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


def make_report(
    records: List[ImgRec], groups: List[List[int]], keep_policy: str
) -> Dict:
    """
    Generate JSON report with duplicate groups and statistics.
    """
    report_groups = []
    total_duplicates = 0

    for group in groups:
        rep_idx = pick_representative(group, records, keep_policy)
        duplicates = [records[i].path for i in group if i != rep_idx]
        total_duplicates += len(duplicates)

        report_groups.append({"keep": records[rep_idx].path, "duplicates": duplicates})

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(records),
        "duplicate_groups": len(groups),
        "total_duplicates": total_duplicates,
        "keep_policy": keep_policy,
        "groups": report_groups,
    }

    return report
