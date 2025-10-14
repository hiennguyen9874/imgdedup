"""
Command line interface and main workflow orchestration.
"""

import argparse
import json
import os
import sys
from datetime import datetime

from .hardware import print_gpu_info
from .filesystem import scan_images
from .models import extract_clip_features_multigpu
from .similarity import build_groups_clip
from .reporting import make_report
from .deletion import delete_duplicates


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Image deduplication tool using CLIP semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "folder", type=str, help="Root folder to scan recursively for images"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.985,
        help="Cosine similarity threshold (0.0-1.0). Default: 0.985",
    )

    parser.add_argument(
        "--k", type=int, default=10, help="Top-k neighbors to search. Default: 10"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference. Default: 128",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip2-base-patch16-naflex",
        help="Hugging Face model name. Default: google/siglip2-base-patch16-naflex",
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
        choices=["lexi", "smallest", "largest", "newest", "oldest"],
        default="lexi",
        help="Policy for selecting which file to keep. Default: lexi",
    )

    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Delete duplicates immediately (default is dry run)",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to output JSON report. Default: <folder>/dedup_report.json",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments"""
    # Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        sys.exit(1)

    # Validate GPU memory fraction
    if not 0.1 <= args.gpu_memory_fraction <= 1.0:
        print(
            f"Error: --gpu-memory-fraction must be between 0.1 and 1.0, got {args.gpu_memory_fraction}"
        )
        sys.exit(1)

    # Set default report path
    if args.report is None:
        args.report = os.path.join(args.folder, "dedup_report.json")


def print_config(args):
    """Print configuration summary"""
    print("=" * 60)
    print("Image Deduplication Tool - Multi-GPU CLIP Mode")
    print("=" * 60)
    print(f"Folder: {args.folder}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Top-k: {args.k}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs to use: {args.gpus if args.gpus is not None else 'all available'}")
    print(f"GPU memory fraction: {args.gpu_memory_fraction}")
    print(f"Keep policy: {args.keep_policy}")
    print(
        f"Mode: {'IN-PLACE (will delete)' if args.inplace else 'DRY RUN (no deletion)'}"
    )
    print(f"Report: {args.report}")
    print("=" * 60)
    print()


def main():
    """Main entry point for the CLI"""
    args = parse_args()
    validate_args(args)

    print_config(args)

    # Print GPU information
    print_gpu_info()
    print()

    # Step 1: Scan images
    print("[1/5] Scanning for images...")
    records = scan_images(args.folder)

    if not records:
        print("No images found. Exiting.")
        sys.exit(0)

    print(f"Found {len(records)} images\n")

    # Step 2: Extract features
    print("[2/5] Extracting CLIP features...")
    features, valid_indices = extract_clip_features_multigpu(
        records,
        model_name=args.model,
        batch_size=args.batch_size,
        num_gpus=args.gpus,
        gpu_memory_fraction=args.gpu_memory_fraction,
    )

    if len(valid_indices) == 0:
        print("No valid images to process. Exiting.")
        sys.exit(0)

    print(f"Extracted features: {features.shape}\n")

    # Step 3: Build duplicate groups
    print("[3/5] Building duplicate groups with FAISS...")
    groups = build_groups_clip(
        records, features, valid_indices, threshold=args.threshold, k=args.k
    )

    print(f"Found {len(groups)} duplicate groups")
    total_duplicates = sum(len(g) - 1 for g in groups)
    print(f"Total duplicates to remove: {total_duplicates}\n")

    # Step 4: Generate report
    print("[4/5] Generating report...")
    report = make_report(records, groups, args.keep_policy)

    # Save report
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {args.report}\n")

    # Print summary
    print("Summary:")
    print(f"  Total images scanned: {report['total_images']}")
    print(f"  Duplicate groups found: {report['duplicate_groups']}")
    print(f"  Files to be removed: {report['total_duplicates']}")

    # Step 5: Delete if inplace
    if args.inplace:
        print("\n[5/5] Deleting duplicates...")
        result = delete_duplicates(report)

        # Print detailed cleanup summary
        print(f"\nCleanup Summary:")
        print(f"  Total files processed: {result['total_attempted']}")
        print(f"  Successfully deleted: {result['successfully_deleted']}")
        print(f"  Already removed: {result['already_deleted']}")
        print(f"  Actual errors: {result['actual_errors']}")

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
            if result["already_deleted"] > 0:
                print(f"✓ Found {result['already_deleted']} files already removed")
    else:
        print("\nDry run complete. No files deleted.")
        print(f"To delete duplicates, run again with --inplace flag")

    print("\nDone.")


if __name__ == "__main__":
    main()
