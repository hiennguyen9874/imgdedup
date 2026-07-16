"""Reports for representative dataset selection."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from .quality import QualityMetrics


def build_selection_report(
    *,
    input_root: str,
    output_root: str,
    total_images: int,
    eligible_after_quality: int,
    duplicate_groups: int,
    duplicate_paths: Iterable[str],
    quality_rejected: Dict[str, QualityMetrics],
    embedding_failed: Iterable[str],
    not_selected: Iterable[str],
    exports: List[Dict],
    requested: int,
    method: str,
    seed: int,
    model: str,
    copy_mode: str,
) -> Dict:
    duplicates = sorted(set(duplicate_paths))
    embedding_failed = sorted(set(embedding_failed))
    not_selected = sorted(set(not_selected))
    selected = [item for item in exports if item["status"] == "exported"]
    export_failed = [item for item in exports if item["status"] != "exported"]
    rejected = []
    for path, metrics in sorted(quality_rejected.items()):
        rejected.append({"path": path, "reason": metrics.rejection_reason, "quality": metrics.to_dict()})
    rejected.extend({"path": path, "reason": "duplicate"} for path in duplicates)
    rejected.extend({"path": path, "reason": "embedding_failed"} for path in embedding_failed)
    rejected.extend({"path": path, "reason": "not_selected"} for path in not_selected)
    rejected.extend({"path": item["source"], "reason": "export_failed", "error": item.get("error")} for item in export_failed)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_root": input_root,
        "output_root": output_root,
        "configuration": {
            "requested": requested,
            "selection_method": method,
            "seed": seed,
            "model": model,
            "copy_mode": copy_mode,
        },
        "funnel": {
            "input": total_images,
            "quality_rejected": len(quality_rejected),
            "eligible_after_quality": eligible_after_quality,
            "duplicate_groups": duplicate_groups,
            "duplicates_removed": len(duplicates),
            "embedding_failed": len(embedding_failed),
            "eligible_after_dedup": len(not_selected) + len(exports),
            "selected": len(selected),
            "export_failed": len(export_failed),
        },
        "selected": exports,
        "rejected": rejected,
    }


def write_selection_reports(report: Dict, report_dir: str) -> Dict[str, str]:
    """Write JSON, CSV, path lists, and a compact funnel summary."""
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "selection_report.json",
        "csv": destination / "selected_images.csv",
        "selected": destination / "selected.txt",
        "selected_output": destination / "selected_output.txt",
        "rejected": destination / "rejected_all.txt",
        "stats": destination / "stats.txt",
    }
    paths["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")

    selected = [item for item in report["selected"] if item["status"] == "exported"]
    paths["selected"].write_text("".join(f"{item['source']}\n" for item in selected), encoding="utf-8")
    paths["selected_output"].write_text("".join(f"{item['output']}\n" for item in selected), encoding="utf-8")
    paths["rejected"].write_text(
        "".join(f"{item['reason']}\t{item['path']}\n" for item in report["rejected"]),
        encoding="utf-8",
    )
    paths["stats"].write_text(
        "".join(f"{key}: {value}\n" for key, value in report["funnel"].items()),
        encoding="utf-8",
    )

    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "output", "status", "error"])
        writer.writeheader()
        for item in report["selected"]:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    return {key: str(value) for key, value in paths.items()}
