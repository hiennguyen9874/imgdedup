"""
Safe file deletion operations with error handling and progress tracking.
"""

import os
from typing import Dict
from tqdm import tqdm


def delete_duplicates(report: Dict) -> Dict:
    """
    Delete duplicate files based on report.
    Returns dict with statistics and any actual errors encountered.
    """
    errors = []
    already_deleted = []
    successfully_deleted = 0
    total_to_delete = sum(len(g["duplicates"]) for g in report["groups"])

    with tqdm(total=total_to_delete, desc="Deleting duplicates") as pbar:
        for group in report["groups"]:
            for dup_path in group["duplicates"]:
                try:
                    # Check if file exists before trying to delete
                    if not os.path.exists(dup_path):
                        already_deleted.append(dup_path)
                    else:
                        os.remove(dup_path)
                        successfully_deleted += 1
                    pbar.update(1)
                except OSError as e:
                    # Handle specific OS errors
                    if e.errno == 2:  # No such file or directory
                        already_deleted.append(dup_path)
                    else:
                        errors.append(f"Failed to delete {dup_path}: {e}")
                    pbar.update(1)
                except Exception as e:
                    errors.append(f"Failed to delete {dup_path}: {e}")
                    pbar.update(1)

    # Return detailed statistics
    return {
        "total_attempted": total_to_delete,
        "successfully_deleted": successfully_deleted,
        "already_deleted": len(already_deleted),
        "actual_errors": len(errors),
        "error_details": errors[:10] if errors else [],  # Limit error details
        "has_more_errors": len(errors) > 10,
    }
