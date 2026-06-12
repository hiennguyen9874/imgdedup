"""
Safe file deletion operations with error handling and progress tracking.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Sequence, Union
from tqdm import tqdm


def _iter_duplicate_paths(report: Dict) -> Iterable[str]:
    for group in report["groups"]:
        for duplicate in group["duplicates"]:
            if isinstance(duplicate, dict):
                yield duplicate["path"]
            else:
                yield duplicate


def _root_for_path(roots: Sequence[str], original_path: str) -> Path:
    original = Path(original_path).resolve()
    for root in roots:
        root_path = Path(root).resolve()
        try:
            original.relative_to(root_path)
            return root_path
        except ValueError:
            continue
    return Path(roots[0]).resolve()


def _trash_path(roots: Sequence[str], run_id: str, original_path: str) -> str:
    original = Path(original_path)
    root = _root_for_path(roots, original_path)
    trash_root = root / ".imgdedup" / "trash" / run_id
    try:
        relative = original.resolve().relative_to(root)
    except ValueError:
        relative = Path(original.name)

    destination = trash_root / relative
    if not destination.exists():
        return str(destination)

    suffix = 1
    while True:
        candidate = destination.with_name(f"{destination.stem}.{suffix}{destination.suffix}")
        if not candidate.exists():
            return str(candidate)
        suffix += 1


def delete_duplicates(
    report: Dict,
    root: Union[str, Sequence[str]],
    mode: str = "move",
    run_id: str = None,
) -> Dict:
    """
    Apply duplicate removal based on report.

    mode="move" moves files into .imgdedup/trash/<run_id> and writes a restore
    manifest. mode="hard-delete" removes files permanently.
    """
    errors = []
    already_deleted = []
    actions = []
    successfully_processed = 0
    duplicate_paths = list(_iter_duplicate_paths(report))
    roots = [root] if isinstance(root, str) else list(root)
    run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    with tqdm(total=len(duplicate_paths), desc="Processing duplicates") as pbar:
        for duplicate_path in duplicate_paths:
            try:
                if not os.path.exists(duplicate_path):
                    already_deleted.append(duplicate_path)
                elif mode == "hard-delete":
                    os.remove(duplicate_path)
                    successfully_processed += 1
                else:
                    destination = _trash_path(roots, run_id, duplicate_path)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.move(duplicate_path, destination)
                    actions.append(
                        {
                            "original_path": duplicate_path,
                            "trash_path": destination,
                        }
                    )
                    successfully_processed += 1
                pbar.update(1)
            except OSError as e:
                if e.errno == 2:
                    already_deleted.append(duplicate_path)
                else:
                    errors.append(f"Failed to process {duplicate_path}: {e}")
                pbar.update(1)
            except Exception as e:
                errors.append(f"Failed to process {duplicate_path}: {e}")
                pbar.update(1)

    manifest_path = None
    if mode != "hard-delete" and actions:
        manifest_dir = Path(roots[0]).resolve() / ".imgdedup" / "trash" / run_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = str(manifest_dir / "restore_manifest.json")
        with open(manifest_path, "w") as file_obj:
            json.dump({"run_id": run_id, "actions": actions}, file_obj, indent=2)

    return {
        "run_id": run_id,
        "mode": mode,
        "manifest_path": manifest_path,
        "total_attempted": len(duplicate_paths),
        "successfully_deleted": successfully_processed if mode == "hard-delete" else 0,
        "successfully_moved": successfully_processed if mode != "hard-delete" else 0,
        "already_deleted": len(already_deleted),
        "actual_errors": len(errors),
        "error_details": errors[:10] if errors else [],
        "has_more_errors": len(errors) > 10,
    }
