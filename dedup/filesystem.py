"""
File system operations for image scanning and workload distribution.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch
from .hardware import get_gpu_memory_info


@dataclass
class ImgRec:
    """Image file record with metadata"""

    path: str
    size: int
    mtime: float


def scan_images(folder: str) -> List[ImgRec]:
    """
    Recursively scan directory for supported image files.
    Returns list of ImgRec objects with metadata.
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    records = []

    folder_path = Path(folder).resolve()
    print(f"Scanning directory: {folder_path}")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if Path(file).suffix.lower() in supported_exts:
                full_path = os.path.join(root, file)
                try:
                    stat = os.stat(full_path)
                    records.append(
                        ImgRec(path=full_path, size=stat.st_size, mtime=stat.st_mtime)
                    )
                except Exception as e:
                    print(f"Warning: Could not stat {full_path}: {e}")

    return records


def split_records_for_gpus(records: List[ImgRec], num_gpus: int) -> List[List[ImgRec]]:
    """
    Split image records into roughly equal chunks for multiple GPUs.
    Ensures balanced workload distribution.
    """
    if num_gpus <= 1:
        return [records]

    total_records = len(records)

    # Calculate base chunk size and remainder
    base_chunk_size = total_records // num_gpus
    remainder = total_records % num_gpus

    chunks = []
    start_idx = 0

    for gpu_id in range(num_gpus):
        # Add one extra record to early chunks to handle remainder
        chunk_size = base_chunk_size + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + chunk_size

        if start_idx < total_records:
            chunk = records[start_idx:end_idx]
            chunks.append(chunk)
            print(f"GPU {gpu_id}: {len(chunk)} images")
        else:
            chunks.append([])

        start_idx = end_idx

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]

    print(f"Split {total_records} images into {len(chunks)} GPU chunks")
    return chunks


def get_optimal_batch_size(gpu_id: int, base_batch_size: int = 128) -> int:
    """
    Adjust batch size based on GPU memory constraints.
    Returns optimal batch size for the specific GPU.
    """
    if not torch.cuda.is_available():
        return min(base_batch_size, 32)  # Conservative for CPU

    memory_info = get_gpu_memory_info(gpu_id)
    total_memory_gb = memory_info["total_memory"]

    # Adjust batch size based on available memory
    # These are heuristic values that work well for SigLIP models
    if total_memory_gb >= 40:  # A100, RTX 4090, etc.
        return base_batch_size
    elif total_memory_gb >= 24:  # RTX 3090, RTX 4080, etc.
        return min(base_batch_size, 96)
    elif total_memory_gb >= 16:  # RTX 3080, etc.
        return min(base_batch_size, 64)
    elif total_memory_gb >= 12:  # RTX 3060, etc.
        return min(base_batch_size, 48)
    else:  # Smaller GPUs
        return min(base_batch_size, 32)
