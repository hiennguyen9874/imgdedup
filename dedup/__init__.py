"""
Image Deduplication Tool using CLIP (SigLIP2) Semantic Similarity

A modular Python package for finding and removing duplicate images using
semantic similarity analysis with CLIP models and FAISS indexing.
"""

__version__ = "1.0.0"
__author__ = "Image Deduplication Tool"

from .hardware import (
    check_flash_attention_available,
    check_bfloat16_support,
    get_optimal_dtype,
    get_optimal_attention_implementation,
    detect_available_gpus,
    get_gpu_memory_info,
    print_gpu_info,
)

from .filesystem import (
    ImgRec,
    scan_images,
    split_records_for_gpus,
    get_optimal_batch_size,
)

from .models import (
    ImageDataset,
    collate_fn,
    load_image_for_embedding,
    extract_clip_features,
    extract_features_on_gpu,
    extract_clip_features_multigpu,
)

from .cache import (
    DedupCache,
    build_feature_matrix,
)

from .similarity import (
    l2_normalize,
    UnionFind,
)

from .hashing import (
    compute_sha256,
    compute_image_metadata,
    phash_distance,
)

from .matching import (
    MatchThresholds,
    DuplicatePair,
    decide_pair,
    find_duplicate_pairs,
)

from .reporting import (
    pick_representative,
    make_report,
)

from .deletion import (
    delete_duplicates,
)

from .quality import QualityMetrics, QualityThresholds, measure_quality
from .selection import select_representatives
from .exporting import export_records, prepare_output
from .preview import make_contact_sheet

__all__ = [
    # Hardware detection
    "check_flash_attention_available",
    "check_bfloat16_support",
    "get_optimal_dtype",
    "get_optimal_attention_implementation",
    "detect_available_gpus",
    "get_gpu_memory_info",
    "print_gpu_info",
    # Filesystem operations
    "ImgRec",
    "scan_images",
    "split_records_for_gpus",
    "get_optimal_batch_size",
    # Model operations
    "ImageDataset",
    "collate_fn",
    "load_image_for_embedding",
    "DedupCache",
    "build_feature_matrix",
    "extract_clip_features",
    "extract_features_on_gpu",
    "extract_clip_features_multigpu",
    # Similarity and clustering
    "l2_normalize",
    "UnionFind",
    # Hashing and matching
    "compute_sha256",
    "compute_image_metadata",
    "phash_distance",
    "MatchThresholds",
    "DuplicatePair",
    "decide_pair",
    "find_duplicate_pairs",
    # Reporting
    "pick_representative",
    "make_report",
    # Selection and export
    "QualityMetrics",
    "QualityThresholds",
    "measure_quality",
    "select_representatives",
    "export_records",
    "prepare_output",
    "make_contact_sheet",
    # Deletion
    "delete_duplicates",
]
