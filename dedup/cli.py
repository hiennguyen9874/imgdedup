"""
Command line interface and main workflow orchestration.
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from .hardware import print_gpu_info

from .cache import DedupCache, build_feature_matrix
from .filesystem import scan_images
from .hashing import compute_image_metadata, compute_sha256
from .matching import MatchThresholds, find_duplicate_pairs
from .models import extract_clip_features_multigpu
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
        "--model",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
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
        choices=["lexi", "smallest", "largest", "highest-resolution", "newest", "oldest"],
        default="highest-resolution",
        help="Policy for selecting which file to keep. Default: highest-resolution",
    )

    parser.add_argument(
        "--cross-folder-only",
        action="store_true",
        help="Only compare images from different parent folders.",
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

    if args.cosine_review > args.cosine_verify or args.cosine_verify > args.cosine_auto:
        print("Error: cosine thresholds must satisfy review <= verify <= auto")
        sys.exit(1)

    if args.hard_delete and not args.yes:
        print("Error: --hard-delete requires --yes")
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
    print(f"Cosine auto threshold: {args.cosine_auto}")
    print(f"Cosine verify threshold: {args.cosine_verify}")
    print(f"Cosine review threshold: {args.cosine_review}")
    print(f"pHash auto distance: {args.phash_auto_distance}")
    print(f"pHash verify distance: {args.phash_verify_distance}")
    print(f"Top-k: {args.k}")
    print(f"Save FAISS index: {args.save_faiss_index}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs to use: {args.gpus if args.gpus is not None else 'all available'}")
    print(f"GPU memory fraction: {args.gpu_memory_fraction}")
    print(f"Keep policy: {args.keep_policy}")
    print(f"Cross-folder only: {args.cross_folder_only}")
    if args.inplace and args.hard_delete:
        mode = "IN-PLACE HARD DELETE"
    elif args.inplace:
        mode = "IN-PLACE MOVE TO TRASH"
    else:
        mode = "DRY RUN (no deletion)"
    print(f"Mode: {mode}")
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
    print("[1/6] Scanning for images...")
    records = scan_images(args.folder)

    if not records:
        print("No images found. Exiting.")
        sys.exit(0)

    print(f"Found {len(records)} images\n")

    cache = DedupCache(args.folder)

    # Step 2: Compute exact and perceptual hashes
    cached_metadata = cache.apply_cached_metadata(records)
    print(f"[2/6] Computing sha256 and pHash... ({cached_metadata} metadata cache hits)")
    for record in tqdm(records, desc="Hashing images"):
        metadata_cached = (
            record.sha256 is not None
            and record.phash is not None
            and record.width is not None
            and record.height is not None
        )
        if record.sha256 is None:
            try:
                record.sha256 = compute_sha256(record.path)
            except OSError as e:
                print(f"Warning: Could not compute sha256 for {record.path}: {e}")

        if record.phash is None or record.width is None or record.height is None:
            try:
                record.phash, record.width, record.height = compute_image_metadata(record.path)
            except ImportError as e:
                print(f"Error: {e}")
                cache.close()
                sys.exit(1)

        if not metadata_cached:
            cache.update_metadata(record)
    cache.conn.commit()

    print()

    # Step 3: Extract features
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
        )
        extracted_indices = [missing_embedding_indices[idx] for idx in extracted_indices]
    else:
        extracted_features = None
        extracted_indices = []

    if extracted_features is None:
        extracted_features = np.empty((0, 0), dtype=np.float32)

    features, valid_indices, embedding_cache_hits = build_feature_matrix(
        cache,
        records,
        args.model,
        extracted_features,
        extracted_indices,
    )

    if len(valid_indices) == 0:
        print("No valid images to process. Exiting.")
        cache.close()
        sys.exit(0)

    cache.save_embeddings(records, features, valid_indices, args.model)
    print(f"Extracted/reused features: {features.shape} ({embedding_cache_hits} reused from cache)\n")

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
        cross_folder_only=args.cross_folder_only,
    )

    print(f"Found {len(groups)} duplicate groups")
    print(f"Duplicate pairs: {len(duplicate_pairs)}")
    print(f"Review-only pairs: {len(review_pairs)}")
    total_duplicates = sum(len(g) - 1 for g in groups)
    print(f"Total duplicates to remove: {total_duplicates}\n")

    # Step 5: Generate report
    print("[5/6] Generating report...")
    report = make_report(records, groups, args.keep_policy, duplicate_pairs, review_pairs)

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
    print(f"  Review-only pairs: {report['review_only_pairs']}")

    # Step 6: Delete if inplace
    if args.inplace:
        action = "hard-deleting" if args.hard_delete else "moving duplicates to trash"
        print(f"\n[6/6] Applying report by {action}...")
        delete_mode = "hard-delete" if args.hard_delete else "move"
        result = delete_duplicates(report, args.folder, mode=delete_mode)

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
