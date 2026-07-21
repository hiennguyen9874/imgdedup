"""
Command line interface and main workflow orchestration.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .hardware import print_gpu_info

from .cache import DedupCache, build_feature_matrix
from .filesystem import ImgRec, scan_images
from .hashing import compute_image_metadata, compute_sha256
from .matching import MatchThresholds, find_duplicate_pairs, find_matches_to_reference
from .models import extract_clip_features_multigpu
from .reporting import make_report, pick_representative
from .deletion import delete_duplicates
from .exporting import export_records, staged_output, validate_output
from .preview import make_contact_sheet
from .quality import QualityThresholds, measure_records
from .selection import select_representatives
from .selection_reporting import build_selection_report, write_selection_reports
from .config import ConfigError, apply_config, cli_command, load_config
from .yolo import (
    build_cleanup_plan,
    export_yolo_dataset,
    flatten_samples,
    load_yolo_dataset,
    parse_split_priority,
)


def parse_args(argv=None):
    """Parse CLI arguments, then apply optional YAML config overrides."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        _config_path, config = load_config(argv)
    except ConfigError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    command = config.get("command", "dedup") if config else cli_command(argv)
    explicit_command = cli_command(argv)
    if config and explicit_command != "dedup" and explicit_command != command:
        raise SystemExit(
            f"Error: config command '{command}' conflicts with CLI command '{explicit_command}'"
        )

    if command == "yolo-dedup":
        parser = argparse.ArgumentParser(
            description="Export a YOLO dataset with duplicate images removed using the dedup matcher",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("command", choices=["yolo-dedup"], nargs="?" if config else None)
        parser.add_argument("dataset", nargs="?" if config else None, help="YOLO dataset root containing data.yaml")
        parser.add_argument("--config", help="YAML config file; its values override CLI options")
        parser.add_argument("--output", required=not bool(config), help="New output dataset directory")
        parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
        parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
        parser.add_argument(
            "--split-priority",
            default="test,val,train",
            help="Highest-to-lowest retained split order. Default: test,val,train",
        )
        parser.add_argument("--cosine-auto", type=float, default=0.97)
        parser.add_argument("--cosine-verify", type=float, default=0.90)
        parser.add_argument("--cosine-review", type=float, default=0.85)
        parser.add_argument("--phash-auto-distance", type=int, default=4)
        parser.add_argument("--phash-verify-distance", type=int, default=8)
        parser.add_argument("--k", type=int, default=50)
        parser.add_argument("--save-faiss-index", action="store_true")
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--metadata-workers", type=int, default=min(32, (os.cpu_count() or 1)))
        parser.add_argument("--loader-workers", type=int, default=0)
        parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
        parser.add_argument("--gpus", type=int, default=None)
        parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
        parser.add_argument(
            "--keep-policy",
            choices=["lexi", "smallest", "largest", "highest-resolution", "newest", "oldest", "best-quality"],
            default="highest-resolution",
        )
        parser.add_argument("--cross-folder-only", action="store_true")
        parser.add_argument("--grouping", choices=["connected", "agglomerative"], default="connected")
        parser.add_argument("--agglomerative-linkage", choices=["complete", "average"], default="complete")
        parser.add_argument("--agglomerative-cosine-threshold", type=float, default=None)
        args = apply_config(parser, parser.parse_args(argv), config, "yolo-dedup")
        if args.dataset is not None:
            args.dataset = os.path.expanduser(args.dataset)
        if args.output is not None:
            args.output = os.path.expanduser(args.output)
        args.yolo_dedup = True
        return args

    if command == "select":
        parser = argparse.ArgumentParser(
            description="Deduplicate and export a representative image subset",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("command", choices=["select"], nargs="?" if config else None)
        parser.add_argument("folder", nargs="?" if config else None, help="Folder to scan recursively")
        parser.add_argument("--config", help="YAML config file; its values override CLI options")
        parser.add_argument("--output", required=not bool(config), help="New output directory")
        parser.add_argument("--num", type=int, required=not bool(config), help="Exact number of images to select")
        parser.add_argument("--selection-method", choices=["kmeans", "farthest", "hybrid"], default="hybrid")
        parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
        parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--make-preview", action="store_true")
        parser.add_argument("--preview-columns", type=int, default=8)
        parser.add_argument("--preview-size", type=int, default=192)
        parser.add_argument("--reject-low-quality", action="store_true")
        parser.add_argument("--min-width", type=int, default=224)
        parser.add_argument("--min-height", type=int, default=224)
        parser.add_argument("--min-blur-score", type=float, default=20.0)
        parser.add_argument("--min-brightness", type=float, default=15.0)
        parser.add_argument("--max-brightness", type=float, default=240.0)
        parser.add_argument(
            "--keep-policy",
            choices=["lexi", "smallest", "largest", "highest-resolution", "newest", "oldest", "best-quality"],
            default="best-quality",
        )
        parser.add_argument("--cosine-auto", type=float, default=0.97)
        parser.add_argument("--cosine-verify", type=float, default=0.90)
        parser.add_argument("--cosine-review", type=float, default=0.85)
        parser.add_argument("--phash-auto-distance", type=int, default=4)
        parser.add_argument("--phash-verify-distance", type=int, default=8)
        parser.add_argument("--k", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--metadata-workers", type=int, default=min(32, (os.cpu_count() or 1)))
        parser.add_argument("--loader-workers", type=int, default=0)
        parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
        parser.add_argument("--gpus", type=int, default=None)
        parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
        parser.add_argument("--grouping", choices=["connected", "agglomerative"], default="connected")
        parser.add_argument("--agglomerative-linkage", choices=["complete", "average"], default="complete")
        parser.add_argument("--agglomerative-cosine-threshold", type=float, default=None)
        args = apply_config(parser, parser.parse_args(argv), config, "select")
        args.select = True
        args.folders = [args.folder]
        args.cross_folder_only = False
        args.save_faiss_index = False
        args.inplace = False
        args.hard_delete = False
        args.yes = False
        args.report = None
        args.no_report = True
        return args

    if command == "remove-like":
        parser = argparse.ArgumentParser(
            description="Remove images in a folder that match one input image",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("command", choices=["remove-like"], nargs="?" if config else None)
        parser.add_argument("folder", nargs="?" if config else None, help="Folder to scan recursively for images to remove")
        parser.add_argument("image", nargs="?" if config else None, help="Reference image to compare against the folder")
        parser.add_argument("--config", help="YAML config file; its values override CLI options")
        parser.add_argument("--cosine-auto", type=float, default=0.97, help="Auto-duplicate cosine threshold. Default: 0.97")
        parser.add_argument("--cosine-verify", type=float, default=0.90, help="Cosine threshold requiring pHash verification. Default: 0.90")
        parser.add_argument("--cosine-review", type=float, default=0.85, help="Review-only cosine threshold. Default: 0.85")
        parser.add_argument("--phash-auto-distance", type=int, default=4, help="Auto-duplicate pHash distance. Default: 4")
        parser.add_argument("--phash-verify-distance", type=int, default=8, help="pHash distance required with --cosine-verify. Default: 8")
        parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference. Default: 128")
        parser.add_argument("--metadata-workers", type=int, default=min(32, (os.cpu_count() or 1)), help="Parallel workers for sha256 and pHash metadata. Default: min(32, CPU count)")
        parser.add_argument("--loader-workers", type=int, default=0, help="PyTorch DataLoader workers for image loading. Default: 0")
        parser.add_argument("--model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="Hugging Face model name. Default: facebook/dinov3-vitb16-pretrain-lvd1689m")
        parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use for parallel processing. Default: all available")
        parser.add_argument("--gpu-memory-fraction", type=float, default=0.9, help="GPU memory fraction to use per GPU (0.1-1.0). Default: 0.9")
        parser.add_argument("--inplace", action="store_true", help="Move matching folder images to .imgdedup/trash immediately (default is dry run)")
        parser.add_argument("--hard-delete", action="store_true", help="Permanently delete matching folder images instead of moving to trash. Requires --yes.")
        parser.add_argument("--yes", action="store_true", help="Confirm destructive hard-delete mode.")
        parser.add_argument("--report", type=str, default=None, help="Path to output JSON report. Default: <folder>/remove_like_report.json")
        parser.add_argument("--no-report", action="store_true", help="Do not write a JSON report file.")
        args = apply_config(parser, parser.parse_args(argv), config, "remove-like")
        args.remove_like = True
        args.folders = [args.folder]
        return args

    parser = argparse.ArgumentParser(
        description="Image deduplication tool using CLIP semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "folders", nargs="*" if config else "+", help="Root folder(s) to scan recursively for images"
    )
    parser.add_argument(
        "--config", help="YAML config file; its values override CLI options"
    )

    parser.add_argument(
        "--cosine-auto",
        type=float,
        default=0.97,
        help="Auto-duplicate cosine threshold. Default: 0.97",
    )

    parser.add_argument(
        "--cosine-verify",
        type=float,
        default=0.90,
        help="Cosine threshold requiring pHash verification. Default: 0.90",
    )

    parser.add_argument(
        "--cosine-review",
        type=float,
        default=0.85,
        help="Review-only cosine threshold. Default: 0.85",
    )

    parser.add_argument(
        "--phash-auto-distance",
        type=int,
        default=4,
        help="Auto-duplicate pHash distance. Default: 4",
    )

    parser.add_argument(
        "--phash-verify-distance",
        type=int,
        default=8,
        help="pHash distance required with --cosine-verify. Default: 8",
    )

    parser.add_argument(
        "--k", type=int, default=50, help="Top-k neighbors to search. Default: 50"
    )

    parser.add_argument(
        "--save-faiss-index",
        action="store_true",
        help="Save .imgdedup/faiss.index after matching. Disabled by default for speed.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference. Default: 128",
    )

    parser.add_argument(
        "--metadata-workers",
        type=int,
        default=min(32, (os.cpu_count() or 1)),
        help="Parallel workers for sha256 and pHash metadata. Default: min(32, CPU count)",
    )

    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers for image loading. Default: 0",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="Hugging Face model name. Default: facebook/dinov3-vitb16-pretrain-lvd1689m",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for parallel processing. Default: all available",
    )

    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.9,
        help="GPU memory fraction to use per GPU (0.1-1.0). Default: 0.9",
    )

    parser.add_argument(
        "--keep-policy",
        type=str,
        choices=["lexi", "smallest", "largest", "highest-resolution", "newest", "oldest", "best-quality"],
        default="highest-resolution",
        help="Policy for selecting which file to keep. Default: highest-resolution",
    )

    parser.add_argument(
        "--cross-folder-only",
        action="store_true",
        help="Only compare images from different parent folders.",
    )

    parser.add_argument(
        "--grouping",
        type=str,
        choices=["connected", "agglomerative"],
        default="connected",
        help="Duplicate grouping method. Default: connected",
    )

    parser.add_argument(
        "--agglomerative-linkage",
        type=str,
        choices=["complete", "average"],
        default="complete",
        help="Linkage for --grouping agglomerative. Default: complete",
    )

    parser.add_argument(
        "--agglomerative-cosine-threshold",
        type=float,
        default=None,
        help="Cosine threshold for agglomerative splitting. Default: --cosine-auto",
    )

    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Move duplicates to .imgdedup/trash immediately (default is dry run)",
    )

    parser.add_argument(
        "--hard-delete",
        action="store_true",
        help="Permanently delete duplicates instead of moving to trash. Requires --yes.",
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive hard-delete mode.",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to output JSON report. Default: <folder>/dedup_report.json for one folder, ./dedup_report.json for multiple folders.",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write a JSON report file.",
    )

    return apply_config(parser, parser.parse_args(argv), config, "dedup")


def validate_args(args):
    """Validate merged command line and config arguments."""
    if getattr(args, "yolo_dedup", False):
        if args.dataset is None or not os.path.isdir(args.dataset):
            print(f"Error: {args.dataset} is not a valid YOLO dataset directory")
            sys.exit(1)
        if args.output is None:
            print("Error: yolo-dedup requires --output")
            sys.exit(1)
        try:
            parse_split_priority(args.split_priority)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        _validate_matching_options(args)
        return

    if not args.folders or any(folder is None for folder in args.folders):
        print("Error: at least one input folder is required")
        sys.exit(1)
    if getattr(args, "select", False):
        if args.output is None or args.num is None:
            print("Error: select requires 'output' and 'num'")
            sys.exit(1)
    if getattr(args, "remove_like", False) and args.image is None:
        print("Error: remove-like requires 'image'")
        sys.exit(1)

    # Validate folders
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"Error: {folder} is not a valid directory")
            sys.exit(1)

    if getattr(args, "remove_like", False) and not os.path.isfile(args.image):
        print(f"Error: {args.image} is not a valid file")
        sys.exit(1)

    if getattr(args, "select", False):
        if args.num < 1:
            print("Error: --num must be at least 1")
            sys.exit(1)
        if args.min_width < 1 or args.min_height < 1:
            print("Error: quality dimensions must be at least 1")
            sys.exit(1)
        if not 0 <= args.min_brightness <= args.max_brightness <= 255:
            print("Error: brightness thresholds must satisfy 0 <= min <= max <= 255")
            sys.exit(1)
        if args.preview_columns < 1 or args.preview_size < 32:
            print("Error: preview columns must be positive and preview size at least 32")
            sys.exit(1)

    _validate_matching_options(args)

    if args.hard_delete and not args.yes:
        print("Error: --hard-delete requires --yes")
        sys.exit(1)

    if args.no_report and args.report is not None:
        print("Error: --no-report cannot be used with --report")
        sys.exit(1)

    # Set default report path
    if not args.no_report and args.report is None:
        report_root = args.folders[0] if len(args.folders) == 1 else os.getcwd()
        report_name = "remove_like_report.json" if getattr(args, "remove_like", False) else "dedup_report.json"
        args.report = os.path.join(report_root, report_name)

    args.cache_root = args.folders[0] if len(args.folders) == 1 else os.getcwd()


def _validate_matching_options(args):
    if not 0.1 <= args.gpu_memory_fraction <= 1.0:
        print(
            f"Error: --gpu-memory-fraction must be between 0.1 and 1.0, got {args.gpu_memory_fraction}"
        )
        sys.exit(1)
    if args.cosine_review > args.cosine_verify or args.cosine_verify > args.cosine_auto:
        print("Error: cosine thresholds must satisfy review <= verify <= auto")
        sys.exit(1)
    if args.metadata_workers < 1:
        print("Error: --metadata-workers must be at least 1")
        sys.exit(1)
    if args.loader_workers < 0:
        print("Error: --loader-workers must be 0 or greater")
        sys.exit(1)
    agglomerative_threshold = getattr(args, "agglomerative_cosine_threshold", None)
    if (
        agglomerative_threshold is not None
        and not -1.0 <= agglomerative_threshold <= 1.0
    ):
        print("Error: --agglomerative-cosine-threshold must be between -1.0 and 1.0")
        sys.exit(1)


def print_config(args):
    """Print configuration summary"""
    print("=" * 60)
    print("Image Deduplication Tool - Multi-GPU CLIP Mode")
    print("=" * 60)
    print(f"Folders: {', '.join(args.folders)}")
    print(f"Model: {args.model}")
    print(f"Cosine auto threshold: {args.cosine_auto}")
    print(f"Cosine verify threshold: {args.cosine_verify}")
    print(f"Cosine review threshold: {args.cosine_review}")
    print(f"pHash auto distance: {args.phash_auto_distance}")
    print(f"pHash verify distance: {args.phash_verify_distance}")
    print(f"Top-k: {args.k}")
    print(f"Save FAISS index: {args.save_faiss_index}")
    print(f"Batch size: {args.batch_size}")
    print(f"Metadata workers: {args.metadata_workers}")
    print(f"Loader workers: {args.loader_workers}")
    print(f"GPUs to use: {args.gpus if args.gpus is not None else 'all available'}")
    print(f"GPU memory fraction: {args.gpu_memory_fraction}")
    print(f"Keep policy: {args.keep_policy}")
    print(f"Cross-folder only: {args.cross_folder_only}")
    print(f"Grouping: {args.grouping}")
    if args.grouping == "agglomerative":
        agglomerative_threshold = (
            args.cosine_auto
            if args.agglomerative_cosine_threshold is None
            else args.agglomerative_cosine_threshold
        )
        print(f"Agglomerative linkage: {args.agglomerative_linkage}")
        print(f"Agglomerative cosine threshold: {agglomerative_threshold}")
    if args.inplace and args.hard_delete:
        mode = "IN-PLACE HARD DELETE"
    elif args.inplace:
        mode = "IN-PLACE MOVE TO TRASH"
    else:
        mode = "DRY RUN (no deletion)"
    print(f"Mode: {mode}")
    print(f"Report: {'disabled' if args.no_report else args.report}")
    print("=" * 60)
    print()


def _compute_missing_metadata(record):
    sha256 = record.sha256
    phash = record.phash
    width = record.width
    height = record.height

    if sha256 is None:
        try:
            sha256 = compute_sha256(record.path)
        except OSError as e:
            print(f"Warning: Could not compute sha256 for {record.path}: {e}")

    if phash is None or width is None or height is None:
        phash, width, height = compute_image_metadata(record.path)

    return sha256, phash, width, height


def _file_record(path):
    stat = os.stat(path)
    return ImgRec(path=os.path.abspath(path), size=stat.st_size, mtime=stat.st_mtime)


def _prepare_records(records, args, cache):
    started = time.perf_counter()
    cached_metadata = cache.apply_cached_metadata(records)
    cache_apply_seconds = time.perf_counter() - started
    print(f"Computing sha256 and pHash... ({cached_metadata} metadata cache hits)")
    records_to_hash = [
        record
        for record in records
        if not (
            record.sha256 is not None
            and record.phash is not None
            and record.width is not None
            and record.height is not None
        )
    ]

    hash_started = time.perf_counter()
    if records_to_hash:
        worker_count = min(args.metadata_workers, len(records_to_hash))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_record = {
                executor.submit(_compute_missing_metadata, record): record
                for record in records_to_hash
            }
            for future in tqdm(
                as_completed(future_to_record),
                total=len(future_to_record),
                desc="Hashing images",
            ):
                record = future_to_record[future]
                try:
                    record.sha256, record.phash, record.width, record.height = future.result()
                    cache.update_metadata(record)
                except ImportError as e:
                    print(f"Error: {e}")
                    cache.close()
                    sys.exit(1)
                except OSError as e:
                    print(f"Warning: Could not compute metadata for {record.path}: {e}")
    hash_seconds = time.perf_counter() - hash_started

    commit_started = time.perf_counter()
    cache.conn.commit()
    commit_seconds = time.perf_counter() - commit_started
    print("Metadata timings:")
    print(f"  cache apply: {cache_apply_seconds:.2f}s")
    print(f"  metadata compute: {hash_seconds:.2f}s")
    print(f"  cache commit: {commit_seconds:.2f}s")
    print()

    feature_step_started = time.perf_counter()
    missing_embedding_indices = [
        idx for idx, record in enumerate(records) if cache.get_embedding(record, args.model) is None
    ]
    print(
        f"Extracting CLIP features... ({len(records) - len(missing_embedding_indices)} embedding cache hits)"
    )
    if missing_embedding_indices:
        records_to_extract = [records[idx] for idx in missing_embedding_indices]
        extracted_features, extracted_indices = extract_clip_features_multigpu(
            records_to_extract,
            model_name=args.model,
            batch_size=args.batch_size,
            num_gpus=args.gpus,
            gpu_memory_fraction=args.gpu_memory_fraction,
            loader_workers=args.loader_workers,
        )
        extracted_indices = [missing_embedding_indices[idx] for idx in extracted_indices]
    else:
        extracted_features = None
        extracted_indices = []

    if extracted_features is None:
        extracted_features = np.empty((0, 0), dtype=np.float32)

    matrix_started = time.perf_counter()
    features, valid_indices, embedding_cache_hits = build_feature_matrix(
        cache,
        records,
        args.model,
        extracted_features,
        extracted_indices,
    )
    matrix_seconds = time.perf_counter() - matrix_started

    save_started = time.perf_counter()
    cache.save_embeddings(records, features, valid_indices, args.model)
    embedding_save_seconds = time.perf_counter() - save_started
    feature_step_seconds = time.perf_counter() - feature_step_started
    print(f"Extracted/reused features: {features.shape} ({embedding_cache_hits} reused from cache)")
    print("Feature timings:")
    print(f"  feature matrix build: {matrix_seconds:.2f}s")
    print(f"  embedding cache save: {embedding_save_seconds:.2f}s")
    print(f"  total: {feature_step_seconds:.2f}s\n")
    return features, valid_indices


def _reference_report(records, reference_idx, duplicate_pairs, review_pairs):
    duplicates = []
    for pair in duplicate_pairs:
        duplicate_idx = pair.b if pair.a == reference_idx else pair.a
        duplicates.append(
            {
                "path": records[duplicate_idx].path,
                "reason": pair.reason,
                "cosine": pair.cosine,
                "phash_distance": pair.phash_distance,
                "same_sha256": pair.same_sha256,
                "confidence": pair.confidence,
            }
        )

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference_image": records[reference_idx].path,
        "total_images": len(records) - 1,
        "duplicate_groups": 1 if duplicates else 0,
        "total_duplicates": len(duplicates),
        "review_only_pairs": len(review_pairs),
        "groups": [{"keep": records[reference_idx].path, "duplicates": duplicates}] if duplicates else [],
        "review_only": [
            {
                "a": records[pair.a].path,
                "b": records[pair.b].path,
                "cosine": pair.cosine,
                "phash_distance": pair.phash_distance,
                "same_sha256": pair.same_sha256,
                "decision": pair.decision,
                "reason": pair.reason,
                "confidence": pair.confidence,
            }
            for pair in review_pairs
        ],
    }


def run_remove_like(args):
    print("=" * 60)
    print("Remove Images Matching Reference")
    print("=" * 60)
    print(f"Folder: {args.folder}")
    print(f"Reference image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Mode: {'IN-PLACE HARD DELETE' if args.inplace and args.hard_delete else 'IN-PLACE MOVE TO TRASH' if args.inplace else 'DRY RUN (no deletion)'}")
    print(f"Report: {'disabled' if args.no_report else args.report}")
    print("=" * 60)
    print()

    print_gpu_info()
    print()

    print("[1/5] Scanning folder and reference image...")
    folder_records = scan_images(args.folder)
    if not folder_records:
        print("No images found in folder. Exiting.")
        sys.exit(0)

    reference_idx = 0
    records = [_file_record(args.image)] + folder_records
    candidate_indices = list(range(1, len(records)))
    print(f"Found {len(folder_records)} folder images")

    cache = DedupCache(args.cache_root)
    print("[2/5] Preparing hashes and embeddings...")
    features, valid_indices = _prepare_records(records, args, cache)
    if reference_idx not in valid_indices:
        print("No valid reference image features to process. Exiting.")
        cache.close()
        sys.exit(1)

    print("[3/5] Comparing folder images to reference...")
    thresholds = MatchThresholds(
        cosine_auto=args.cosine_auto,
        cosine_verify=args.cosine_verify,
        cosine_review=args.cosine_review,
        phash_auto_distance=args.phash_auto_distance,
        phash_verify_distance=args.phash_verify_distance,
    )
    duplicate_pairs, review_pairs = find_matches_to_reference(
        records, features, valid_indices, reference_idx, candidate_indices, thresholds
    )
    report = _reference_report(records, reference_idx, duplicate_pairs, review_pairs)
    print(f"Matching folder images to remove: {report['total_duplicates']}")
    print(f"Review-only pairs: {report['review_only_pairs']}\n")

    print("[4/5] Generating report...")
    if args.no_report:
        print("Report writing disabled.")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.report}")

    if args.inplace:
        print("\n[5/5] Applying report...")
        delete_mode = "hard-delete" if args.hard_delete else "move"
        result = delete_duplicates(report, [args.folder], mode=delete_mode)
        print(f"Successfully moved: {result['successfully_moved']}")
        print(f"Successfully deleted: {result['successfully_deleted']}")
        print(f"Actual errors: {result['actual_errors']}")
        if result['manifest_path']:
            print(f"Restore manifest: {result['manifest_path']}")
    else:
        print("\nDry run complete. No files deleted.")
        print("To move matches to trash, run again with --inplace flag")

    cache.close()
    print("\nDone.")


def run_select(args):
    """Deduplicate, sample, and export an exact-size representative subset."""
    try:
        output_root = validate_output(args.folder, args.output, args.force)
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("=" * 60)
    print("Representative Image Selection")
    print("=" * 60)
    print(f"Folder: {args.folder}")
    print(f"Output: {output_root}")
    print(f"Requested images: {args.num}")
    print(f"Selection method: {args.selection_method} (seed {args.seed})")
    print(f"Quality rejection: {'enabled' if args.reject_low_quality else 'disabled'}")
    print("=" * 60)

    all_records = scan_images(args.folder)
    if not all_records:
        print("No images found. Exiting.")
        sys.exit(0)
    print(f"[1/7] Found {len(all_records)} images")

    print("[2/7] Measuring image quality...")
    quality_thresholds = QualityThresholds(
        min_width=args.min_width,
        min_height=args.min_height,
        min_blur_score=args.min_blur_score,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
    )
    quality_metrics = measure_records(
        all_records, quality_thresholds if args.reject_low_quality else None
    )
    quality_rejected = {
        path: metrics
        for path, metrics in quality_metrics.items()
        if args.reject_low_quality and metrics.rejection_reason is not None
    }
    records = [record for record in all_records if record.path not in quality_rejected]
    print(f"Quality rejected: {len(quality_rejected)}")
    if not records:
        print("Error: no images remain after quality filtering")
        sys.exit(1)

    cache = DedupCache(args.cache_root)
    try:
        print("[3/7] Preparing hashes and embeddings...")
        features, valid_indices = _prepare_records(records, args, cache)
        if not valid_indices:
            print("Error: no valid image embeddings available")
            sys.exit(1)

        print("[4/7] Removing duplicate candidates...")
        thresholds = MatchThresholds(
            cosine_auto=args.cosine_auto,
            cosine_verify=args.cosine_verify,
            cosine_review=args.cosine_review,
            phash_auto_distance=args.phash_auto_distance,
            phash_verify_distance=args.phash_verify_distance,
        )
        duplicate_pairs, review_pairs, groups = find_duplicate_pairs(
            records,
            features,
            valid_indices,
            thresholds=thresholds,
            k=args.k,
            show_progress=True,
            grouping_method=args.grouping,
            agglomerative_linkage=args.agglomerative_linkage,
            agglomerative_cosine_threshold=args.agglomerative_cosine_threshold,
        )

        valid_set = set(valid_indices)
        candidate_indices = set(valid_indices)
        duplicate_paths = set()
        for group in groups:
            valid_group = [idx for idx in group if idx in valid_set]
            if not valid_group:
                continue
            representative = pick_representative(
                valid_group, records, args.keep_policy, quality_metrics
            )
            candidate_indices.difference_update(group)
            candidate_indices.add(representative)
            duplicate_paths.update(records[idx].path for idx in group if idx != representative)

        ordered_candidates = sorted(candidate_indices, key=lambda idx: records[idx].path)
        if args.num > len(ordered_candidates):
            print(
                f"Error: requested {args.num} images, but only {len(ordered_candidates)} "
                "valid unique images remain"
            )
            sys.exit(1)

        feature_row_by_record = {
            record_idx: row_idx for row_idx, record_idx in enumerate(valid_indices)
        }
        candidate_features = np.asarray(
            [features[feature_row_by_record[idx]] for idx in ordered_candidates],
            dtype=np.float32,
        )

        print(f"[5/7] Selecting {args.num} of {len(ordered_candidates)} unique images...")
        selected_rows = select_representatives(
            candidate_features, args.num, args.selection_method, args.seed
        )
        selected_indices = [ordered_candidates[row] for row in selected_rows]
        selected_records = [records[idx] for idx in selected_indices]
        selected_paths = {record.path for record in selected_records}
        not_selected = [
            records[idx].path
            for idx in ordered_candidates
            if records[idx].path not in selected_paths
        ]
        embedding_failed = [
            record.path
            for idx, record in enumerate(records)
            if idx not in valid_set and record.path not in duplicate_paths
        ]

        print(f"[6/7] Exporting with mode '{args.copy_mode}'...")
        with staged_output(args.folder, args.output, args.force) as staging_root:
            exports = export_records(
                selected_records, args.folder, str(staging_root), args.copy_mode
            )
            for exported_record in exports:
                staged_path = Path(exported_record["output"])
                exported_record["output"] = str(
                    output_root / staged_path.relative_to(staging_root)
                )
            report = build_selection_report(
                input_root=os.path.abspath(args.folder),
                output_root=str(output_root),
                total_images=len(all_records),
                eligible_after_quality=len(records),
                duplicate_groups=len(groups),
                duplicate_paths=duplicate_paths,
                quality_rejected=quality_rejected,
                embedding_failed=embedding_failed,
                not_selected=not_selected,
                exports=exports,
                requested=args.num,
                method=args.selection_method,
                seed=args.seed,
                model=args.model,
                copy_mode=args.copy_mode,
            )

            print("[7/7] Writing reports and preview...")
            report_paths = write_selection_reports(report, str(staging_root / "reports"))
            if args.make_preview:
                preview_path = staging_root / "previews" / "selected.jpg"
                rendered = make_contact_sheet(
                    selected_records,
                    str(preview_path),
                    columns=args.preview_columns,
                    thumbnail_size=args.preview_size,
                )
                print(f"Preview: {output_root / 'previews' / 'selected.jpg'} ({rendered} images)")

            exported = report["funnel"]["selected"]
            failed = report["funnel"]["export_failed"]
            if failed:
                print(f"Selected and exported: {exported}/{args.num}")
                print(f"Export failures: {failed}")
                print(f"Report: {report_paths['json']}")
                sys.exit(1)

        print(f"Selected and exported: {exported}/{args.num}")
        print("Export failures: 0")
        print(f"Report: {output_root / 'reports' / 'selection_report.json'}")
    finally:
        cache.close()


def run_yolo_dedup(args):
    """Deduplicate manifest-declared YOLO images and export a new dataset."""
    try:
        priority = parse_split_priority(args.split_priority)
        data, samples_by_split = load_yolo_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    samples = flatten_samples(samples_by_split)
    records = [_file_record(str(sample.image_path)) for sample in samples]
    if not records:
        print("No images declared in YOLO manifests. Exiting.")
        sys.exit(0)

    print(f"Found {len(records)} images declared in YOLO manifests")
    print_gpu_info()
    cache = DedupCache(args.dataset)
    try:
        features, valid_indices = _prepare_records(records, args, cache)
        thresholds = MatchThresholds(
            cosine_auto=args.cosine_auto,
            cosine_verify=args.cosine_verify,
            cosine_review=args.cosine_review,
            phash_auto_distance=args.phash_auto_distance,
            phash_verify_distance=args.phash_verify_distance,
        )
        duplicate_pairs, review_pairs, duplicate_groups = find_duplicate_pairs(
            records,
            features,
            valid_indices,
            thresholds=thresholds,
            k=args.k,
            faiss_index_path=str(cache.faiss_index_path) if args.save_faiss_index else None,
            show_progress=True,
            cross_folder_only=args.cross_folder_only,
            grouping_method=args.grouping,
            agglomerative_linkage=args.agglomerative_linkage,
            agglomerative_cosine_threshold=args.agglomerative_cosine_threshold,
        )
        quality_metrics = measure_records(records) if args.keep_policy == "best-quality" else None
        retained, groups = build_cleanup_plan(
            samples,
            records,
            duplicate_groups,
            priority,
            args.keep_policy,
            quality_metrics,
        )
        matching_report = {
            "duplicate_pairs": len(duplicate_pairs),
            "review_only_pairs": len(review_pairs),
            "review_only": [
                {
                    "a": str(samples[pair.a].relative_image_path),
                    "b": str(samples[pair.b].relative_image_path),
                    "cosine": pair.cosine,
                    "phash_distance": pair.phash_distance,
                    "same_sha256": pair.same_sha256,
                    "reason": pair.reason,
                    "confidence": pair.confidence,
                }
                for pair in review_pairs
            ],
        }
        with staged_output(args.dataset, args.output, args.force) as output_root:
            report = export_yolo_dataset(
                args.dataset,
                output_root,
                data,
                retained,
                groups,
                args.copy_mode,
                priority,
                matching_report,
                Path(args.output).expanduser().absolute(),
            )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    finally:
        cache.close()

    print(f"Exported YOLO dataset: {args.output}")
    print(f"Input images: {report['total_input_images']}")
    print(f"Duplicate groups: {report['duplicate_groups']}")
    print(f"Images removed: {report['total_duplicates']}")
    print(f"Review-only pairs: {report['review_only_pairs']}")
    print(f"Annotation conflicts: {report['annotation_conflicts']}")
    print(f"Report: {Path(args.output) / 'reports' / 'yolo_dedup_report.json'}")


def main():
    """Main entry point for the CLI"""
    args = parse_args()
    validate_args(args)

    if getattr(args, "yolo_dedup", False):
        run_yolo_dedup(args)
        return

    if getattr(args, "select", False):
        run_select(args)
        return

    if getattr(args, "remove_like", False):
        run_remove_like(args)
        return

    print_config(args)

    # Print GPU information
    print_gpu_info()
    print()

    # Step 1: Scan images
    print("[1/6] Scanning for images...")
    started = time.perf_counter()
    records = []
    for folder in args.folders:
        records.extend(scan_images(folder))
    scan_seconds = time.perf_counter() - started

    if not records:
        print("No images found. Exiting.")
        sys.exit(0)

    print(f"Found {len(records)} images")
    print("Step 1 timings:")
    print(f"  scan: {scan_seconds:.2f}s\n")

    cache = DedupCache(args.cache_root)

    # Step 2: Compute exact and perceptual hashes
    started = time.perf_counter()
    cached_metadata = cache.apply_cached_metadata(records)
    cache_apply_seconds = time.perf_counter() - started
    print(f"[2/6] Computing sha256 and pHash... ({cached_metadata} metadata cache hits)")
    records_to_hash = [
        record
        for record in records
        if not (
            record.sha256 is not None
            and record.phash is not None
            and record.width is not None
            and record.height is not None
        )
    ]

    hash_started = time.perf_counter()
    if records_to_hash:
        worker_count = min(args.metadata_workers, len(records_to_hash))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_record = {
                executor.submit(_compute_missing_metadata, record): record
                for record in records_to_hash
            }
            for future in tqdm(
                as_completed(future_to_record),
                total=len(future_to_record),
                desc="Hashing images",
            ):
                record = future_to_record[future]
                try:
                    record.sha256, record.phash, record.width, record.height = future.result()
                    cache.update_metadata(record)
                except ImportError as e:
                    print(f"Error: {e}")
                    cache.close()
                    sys.exit(1)
                except OSError as e:
                    print(f"Warning: Could not compute metadata for {record.path}: {e}")
    hash_seconds = time.perf_counter() - hash_started

    commit_started = time.perf_counter()
    cache.conn.commit()
    commit_seconds = time.perf_counter() - commit_started

    print("Step 2 timings:")
    print(f"  cache apply: {cache_apply_seconds:.2f}s")
    print(f"  metadata compute: {hash_seconds:.2f}s")
    print(f"  cache commit: {commit_seconds:.2f}s")
    print()

    # Step 3: Extract features
    feature_step_started = time.perf_counter()
    missing_embedding_indices = [
        idx for idx, record in enumerate(records) if cache.get_embedding(record, args.model) is None
    ]
    print(
        f"[3/6] Extracting CLIP features... ({len(records) - len(missing_embedding_indices)} embedding cache hits)"
    )
    if missing_embedding_indices:
        records_to_extract = [records[idx] for idx in missing_embedding_indices]
        extracted_features, extracted_indices = extract_clip_features_multigpu(
            records_to_extract,
            model_name=args.model,
            batch_size=args.batch_size,
            num_gpus=args.gpus,
            gpu_memory_fraction=args.gpu_memory_fraction,
            loader_workers=args.loader_workers,
        )
        extracted_indices = [missing_embedding_indices[idx] for idx in extracted_indices]
    else:
        extracted_features = None
        extracted_indices = []

    if extracted_features is None:
        extracted_features = np.empty((0, 0), dtype=np.float32)

    matrix_started = time.perf_counter()
    features, valid_indices, embedding_cache_hits = build_feature_matrix(
        cache,
        records,
        args.model,
        extracted_features,
        extracted_indices,
    )
    matrix_seconds = time.perf_counter() - matrix_started

    if len(valid_indices) == 0:
        print("No valid images to process. Exiting.")
        cache.close()
        sys.exit(0)

    save_started = time.perf_counter()
    cache.save_embeddings(records, features, valid_indices, args.model)
    embedding_save_seconds = time.perf_counter() - save_started
    feature_step_seconds = time.perf_counter() - feature_step_started
    print(f"Extracted/reused features: {features.shape} ({embedding_cache_hits} reused from cache)")
    print("Step 3 timings:")
    print(f"  feature matrix build: {matrix_seconds:.2f}s")
    print(f"  embedding cache save: {embedding_save_seconds:.2f}s")
    print(f"  total: {feature_step_seconds:.2f}s\n")

    # Step 4: Build duplicate groups
    print("[4/6] Building duplicate groups with sha256, pHash, and FAISS...")
    thresholds = MatchThresholds(
        cosine_auto=args.cosine_auto,
        cosine_verify=args.cosine_verify,
        cosine_review=args.cosine_review,
        phash_auto_distance=args.phash_auto_distance,
        phash_verify_distance=args.phash_verify_distance,
    )
    duplicate_pairs, review_pairs, groups = find_duplicate_pairs(
        records,
        features,
        valid_indices,
        thresholds=thresholds,
        k=args.k,
        faiss_index_path=str(cache.faiss_index_path) if args.save_faiss_index else None,
        show_progress=True,
        cross_folder_only=args.cross_folder_only,
        grouping_method=args.grouping,
        agglomerative_linkage=args.agglomerative_linkage,
        agglomerative_cosine_threshold=args.agglomerative_cosine_threshold,
    )

    print(f"Found {len(groups)} duplicate groups")
    print(f"Duplicate pairs: {len(duplicate_pairs)}")
    print(f"Review-only pairs: {len(review_pairs)}")
    total_duplicates = sum(len(g) - 1 for g in groups)
    print(f"Total duplicates to remove: {total_duplicates}\n")

    # Step 5: Generate report
    print("[5/6] Generating report...")
    quality_metrics = None
    if args.keep_policy == "best-quality":
        print("Measuring quality for best-quality representative selection...")
        quality_metrics = measure_records(records)
    started = time.perf_counter()
    report = make_report(
        records,
        groups,
        args.keep_policy,
        duplicate_pairs,
        review_pairs,
        quality_metrics,
    )
    report_build_seconds = time.perf_counter() - started

    # Save report
    if args.no_report:
        print("Report writing disabled.")
        print("Step 5 timings:")
        print(f"  report build: {report_build_seconds:.2f}s")
        print()
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        started = time.perf_counter()
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        report_write_seconds = time.perf_counter() - started

        print("Step 5 timings:")
        print(f"  report build: {report_build_seconds:.2f}s")
        print(f"  report write: {report_write_seconds:.2f}s")
        print(f"Report saved to: {args.report}\n")

    # Print summary
    print("Summary:")
    print(f"  Total images scanned: {report['total_images']}")
    print(f"  Duplicate groups found: {report['duplicate_groups']}")
    print(f"  Files to be removed: {report['total_duplicates']}")
    print(f"  Review-only pairs: {report['review_only_pairs']}")

    # Step 6: Delete if inplace
    if args.inplace:
        action = "hard-deleting" if args.hard_delete else "moving duplicates to trash"
        print(f"\n[6/6] Applying report by {action}...")
        delete_mode = "hard-delete" if args.hard_delete else "move"
        result = delete_duplicates(report, args.folders, mode=delete_mode)

        # Print detailed cleanup summary
        print(f"\nCleanup Summary:")
        print(f"  Run ID: {result['run_id']}")
        print(f"  Total files processed: {result['total_attempted']}")
        print(f"  Successfully moved: {result['successfully_moved']}")
        print(f"  Successfully deleted: {result['successfully_deleted']}")
        print(f"  Already removed: {result['already_deleted']}")
        print(f"  Actual errors: {result['actual_errors']}")
        if result['manifest_path']:
            print(f"  Restore manifest: {result['manifest_path']}")

        if result["actual_errors"] > 0:
            print(f"\nEncountered {result['actual_errors']} real errors:")
            for err in result["error_details"]:
                print(f"  - {err}")
            if result["has_more_errors"]:
                print(f"  ... and {result['actual_errors'] - 10} more")
        else:
            if result["successfully_deleted"] > 0:
                print(
                    f"✓ Successfully deleted {result['successfully_deleted']} duplicate files"
                )
            if result["successfully_moved"] > 0:
                print(
                    f"✓ Successfully moved {result['successfully_moved']} duplicate files to trash"
                )
            if result["already_deleted"] > 0:
                print(f"✓ Found {result['already_deleted']} files already removed")
    else:
        print("\nDry run complete. No files deleted.")
        print(f"To move duplicates to trash, run again with --inplace flag")

    cache.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
