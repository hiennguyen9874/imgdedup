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
    extract_clip_features,
    extract_features_on_gpu,
    extract_clip_features_multigpu,
)

from .similarity import (
    l2_normalize,
    UnionFind,
    build_groups_clip,
)

from .reporting import (
    pick_representative,
    make_report,
)

from .deletion import (
    delete_duplicates,
)

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
    "extract_clip_features",
    "extract_features_on_gpu",
    "extract_clip_features_multigpu",
    # Similarity and clustering
    "l2_normalize",
    "UnionFind",
    "build_groups_clip",
    # Reporting
    "pick_representative",
    "make_report",
    # Deletion
    "delete_duplicates",
]
