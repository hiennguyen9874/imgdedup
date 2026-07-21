"""Deduplicate manifest-declared YOLO images and export a safe new dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from .exporting import transfer_file
from .filesystem import ImgRec
from .matching import DuplicatePair
from .quality import QualityMetrics
from .reporting import pick_representative


SPLITS = ("train", "val", "test")
OPTIONAL_SPLITS = frozenset({"val"})


@dataclass(frozen=True)
class YoloSample:
    """One image and its required YOLO label, as declared by a split manifest."""

    split: str
    image_path: Path
    label_path: Path
    relative_image_path: Path


def parse_split_priority(value: str) -> List[str]:
    """Validate and normalize a comma-separated highest-to-lowest split order."""
    priority = [part.strip() for part in value.split(",") if part.strip()]
    if len(priority) != len(SPLITS) or set(priority) != set(SPLITS):
        raise ValueError("--split-priority must contain train,val,test exactly once")
    return priority


def load_yolo_dataset(dataset_root: str) -> tuple[Dict, Dict[str, List[YoloSample]]]:
    """Load a local YOLO data.yaml and its train, val, and test text manifests."""
    root = Path(dataset_root).resolve()
    data_path = root / "data.yaml"
    try:
        data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load '{data_path}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("YOLO data.yaml must contain a YAML mapping")

    configured_root = data.get("path", ".")
    if not isinstance(configured_root, str):
        raise ValueError("YOLO data.yaml 'path' must be a string")
    image_root = (data_path.parent / configured_root).resolve()

    samples_by_split: Dict[str, List[YoloSample]] = {}
    seen_images = set()
    for split in SPLITS:
        manifest_value = data.get(split)
        if manifest_value is None and split in OPTIONAL_SPLITS:
            samples_by_split[split] = []
            continue
        if not isinstance(manifest_value, str) or not manifest_value.endswith(".txt"):
            raise ValueError(
                f"YOLO data.yaml '{split}' must reference one local .txt manifest"
            )
        manifest_path = (image_root / manifest_value).resolve()
        if not manifest_path.is_file():
            raise ValueError(f"missing {split} manifest: {manifest_path}")

        samples = []
        for line_number, raw_path in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            declared_path = raw_path.strip()
            if not declared_path:
                continue
            image_path = (image_root / declared_path).resolve()
            try:
                relative_image_path = image_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path}:{line_number}: image is outside dataset root: {declared_path}"
                ) from exc
            if not image_path.is_file():
                raise ValueError(f"{manifest_path}:{line_number}: missing image: {declared_path}")
            if relative_image_path in seen_images:
                raise ValueError(f"image is declared more than once: {relative_image_path}")
            seen_images.add(relative_image_path)

            label_relative_path = _label_path(relative_image_path)
            label_path = root / label_relative_path
            if not label_path.is_file():
                raise ValueError(
                    f"{manifest_path}:{line_number}: missing label: {label_relative_path}"
                )
            samples.append(YoloSample(split, image_path, label_path, relative_image_path))
        samples_by_split[split] = samples
    return data, samples_by_split


def flatten_samples(samples_by_split: Dict[str, List[YoloSample]]) -> List[YoloSample]:
    """Return manifest samples in a stable order matching the matching records."""
    return [sample for split in SPLITS for sample in samples_by_split[split]]


def build_cleanup_plan(
    samples: Sequence[YoloSample],
    records: Sequence[ImgRec],
    duplicate_groups: Sequence[Sequence[int]],
    priority: Sequence[str],
    keep_policy: str,
    quality_metrics: Optional[Dict[str, QualityMetrics]] = None,
) -> tuple[List[YoloSample], List[Dict]]:
    """Choose one YOLO sample per duplicate group and retain its image-label pair.

    Split priority prevents train/validation leakage into a higher-priority split.
    The normal dedup keep policy breaks ties among candidates in that split.
    """
    if len(samples) != len(records):
        raise ValueError("YOLO samples and image records must have the same length")

    rank = {split: position for position, split in enumerate(priority)}
    removed_indices = set()
    groups = []
    for group in duplicate_groups:
        group = list(group)
        highest_priority = min(rank[samples[index].split] for index in group)
        candidates = [
            index for index in group if rank[samples[index].split] == highest_priority
        ]
        kept_index = pick_representative(
            candidates, list(records), keep_policy, quality_metrics
        )
        removed_indices.update(index for index in group if index != kept_index)
        group_samples = [samples[index] for index in group]
        groups.append(
            {
                "kept": [_sample_report(samples[kept_index])],
                "removed": [_sample_report(samples[index]) for index in group if index != kept_index],
                "splits": sorted({sample.split for sample in group_samples}),
                "label_conflict": len(
                    {sample.label_path.read_text(encoding="utf-8") for sample in group_samples}
                ) > 1,
            }
        )

    retained = [sample for index, sample in enumerate(samples) if index not in removed_indices]
    return retained, groups


def export_yolo_dataset(
    source_root: str,
    output_root: Path,
    data: Dict,
    retained: Iterable[YoloSample],
    groups: List[Dict],
    copy_mode: str,
    priority: Sequence[str],
    matching_report: Dict,
    published_output_root: Path | None = None,
) -> Dict:
    """Export retained pairs, rewritten manifests, and the deduplication report."""
    retained_by_split: Dict[str, List[YoloSample]] = defaultdict(list)
    retained = list(retained)
    for sample in retained:
        retained_by_split[sample.split].append(sample)
        transfer_file(sample.image_path, output_root / sample.relative_image_path, copy_mode)
        transfer_file(
            sample.label_path,
            output_root / _label_path(sample.relative_image_path),
            copy_mode,
        )

    declared_splits = [split for split in SPLITS if split in data]
    for split in declared_splits:
        paths = sorted(
            f"./{sample.relative_image_path.as_posix()}"
            for sample in retained_by_split[split]
        )
        (output_root / f"{split}.txt").write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )

    output_data = dict(data)
    output_data["path"] = "."
    for split in declared_splits:
        output_data[split] = f"./{split}.txt"
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(output_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    source_classes = Path(source_root).resolve() / "classes.txt"
    if source_classes.is_file():
        transfer_file(source_classes, output_root / "classes.txt", copy_mode)

    report = {
        "input_root": str(Path(source_root).resolve()),
        "output_root": str(published_output_root or output_root),
        "method": "sha256_phash_embedding",
        "split_priority": list(priority),
        "total_input_images": len(retained) + sum(len(group["removed"]) for group in groups),
        "total_output_images": len(retained),
        "duplicate_groups": len(groups),
        "total_duplicates": sum(len(group["removed"]) for group in groups),
        "duplicate_pairs": matching_report["duplicate_pairs"],
        "review_only_pairs": matching_report["review_only_pairs"],
        "annotation_conflicts": sum(group["label_conflict"] for group in groups),
        "output_split_counts": {split: len(retained_by_split[split]) for split in SPLITS},
        "groups": groups,
        "review_only": matching_report["review_only"],
    }
    report_path = output_root / "reports" / "yolo_dedup_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _label_path(relative_image_path: Path) -> Path:
    parts = relative_image_path.parts
    try:
        images_position = parts.index("images")
    except ValueError as exc:
        raise ValueError(
            f"image path must be below an 'images' directory: {relative_image_path}"
        ) from exc
    return Path(*parts[:images_position], "labels", *parts[images_position + 1 :]).with_suffix(".txt")


def _sample_report(sample: YoloSample) -> Dict[str, str]:
    return {
        "split": sample.split,
        "image": str(sample.relative_image_path),
        "label": str(_label_path(sample.relative_image_path)),
    }
