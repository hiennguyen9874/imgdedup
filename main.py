#!/usr/bin/env python3
"""
Image Deduplication Tool using CLIP (SigLIP2) Semantic Similarity
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor

try:
    import faiss
except ImportError:
    print(
        "ERROR: FAISS is required. Install with: pip install faiss-cpu (or faiss-gpu)"
    )
    sys.exit(1)


# Global lock for synchronized model loading across GPUs
_model_loading_lock = threading.Lock()


# ---------------------------
# Hardware Capability Detection
# ---------------------------


def check_flash_attention_available() -> bool:
    """Check if flash attention 2 is available and properly installed"""
    try:
        import flash_attn

        # Check if flash_attn is properly installed and has required version
        if hasattr(flash_attn, '__version__') and flash_attn.__version__ >= "2.0.0":
            # Test if flash attention actually works by importing a key function
            from flash_attn import flash_attn_func
            return True
        else:
            print("Warning: flash_attn version < 2.0.0 detected, may cause issues")
            return False
    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: flash_attn detected but not properly installed: {e}")
        return False


def check_bfloat16_support() -> bool:
    """Check if bfloat16 is supported on current hardware"""
    if not torch.cuda.is_available():
        # bfloat16 is available on CPU but typically slower
        # Check if CPU supports it properly
        try:
            return (
                torch.cuda.is_bf16_supported()
                if hasattr(torch.cuda, "is_bf16_supported")
                else False
            )
        except:
            return False

    # For CUDA devices, check compute capability
    # bfloat16 is well supported on Ampere (8.0+) and newer
    try:
        major, minor = torch.cuda.get_device_capability()
        # Ampere (A100, RTX 30xx) and newer have good bfloat16 support
        return major >= 8
    except:
        return False


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Determine optimal dtype based on hardware.
    Priority: bfloat16 > float16 > float32
    """
    if device.type == "cuda":
        if check_bfloat16_support():
            print("Using bfloat16 (hardware supported)")
            return torch.bfloat16
        else:
            print("Using float16 (bfloat16 not supported on this GPU)")
            return torch.float16
    else:
        # CPU: bfloat16 can work but float32 is safer
        print("Using float32 (CPU mode)")
        return torch.float32


def get_optimal_attention_implementation() -> str:
    """
    Determine optimal attention implementation.
    Priority: flash_attention_2 > sdpa > default (eager)
    """
    if check_flash_attention_available():
        print("Using flash_attention_2 (detected flash-attn package)")
        return "flash_attention_2"
    else:
        # SDPA (Scaled Dot Product Attention) is available in PyTorch 2.0+
        try:
            # Check if PyTorch version supports SDPA
            import torch.nn.functional as F

            if hasattr(F, "scaled_dot_product_attention"):
                print("Using sdpa attention (PyTorch 2.0+ native)")
                return "sdpa"
        except:
            pass

        print("Using default (eager) attention")
        return "eager"


def detect_available_gpus() -> List[int]:
    """
    Detect available CUDA GPUs and return their device IDs.
    Returns list of GPU device IDs that can be used.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, will use CPU")
        return []

    num_gpus = torch.cuda.device_count()
    available_gpus = []

    for i in range(num_gpus):
        try:
            # Test if GPU is accessible
            torch.cuda.set_device(i)
            # Simple test to verify GPU functionality
            test_tensor = torch.tensor([1.0], device=f"cuda:{i}")
            available_gpus.append(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Available")
        except Exception as e:
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Not available ({e})")

    return available_gpus


def get_gpu_memory_info(gpu_id: int) -> Dict[str, float]:
    """
    Get memory information for a specific GPU.
    Returns dict with total_memory, free_memory, used_memory in GB.
    """
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        return {"total_memory": 0.0, "free_memory": 0.0, "used_memory": 0.0}

    torch.cuda.set_device(gpu_id)
    props = torch.cuda.get_device_properties(gpu_id)
    total_memory = props.total_memory / (1024**3)  # Convert to GB

    # Get current memory usage
    memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
    memory_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)

    return {
        "total_memory": total_memory,
        "free_memory": total_memory - memory_reserved,
        "used_memory": memory_allocated
    }


def print_gpu_info():
    """Print detailed information about available GPUs"""
    available_gpus = detect_available_gpus()

    if not available_gpus:
        print("No CUDA GPUs available")
        return

    print(f"Found {len(available_gpus)} available GPU(s):")
    for gpu_id in available_gpus:
        memory_info = get_gpu_memory_info(gpu_id)
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        print(f"    Total Memory: {memory_info['total_memory']:.1f} GB")
        print(f"    Free Memory:  {memory_info['free_memory']:.1f} GB")


@dataclass
class ImgRec:
    """Image file record with metadata"""

    path: str
    size: int
    mtime: float


# ---------------------------
# File System Scanner
# ---------------------------


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


# ---------------------------
# CLIP Feature Extraction
# ---------------------------


class ImageDataset(Dataset):
    """Dataset wrapper for batch image loading"""

    def __init__(self, records: List[ImgRec]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec.path).convert("RGB")
            return img, idx
        except (UnidentifiedImageError, OSError) as e:
            print(f"Warning: Could not load {rec.path}: {e}")
            return None


def collate_fn(batch):
    """Custom collate to handle failed image loads"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], []
    imgs, indices = zip(*batch)
    return list(imgs), list(indices)


def extract_clip_features(
    records: List[ImgRec],
    model_name: str = "google/siglip2-base-patch16-naflex",
    batch_size: int = 128,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract CLIP features for all images using batched inference.
    Automatically detects and uses optimal dtype and attention implementation.
    Returns: (features [N, D], valid_indices)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine optimal dtype and attention implementation
    optimal_dtype = get_optimal_dtype(device)
    attn_implementation = get_optimal_attention_implementation()

    print(f"Loading model: {model_name}")

    # Try to load with optimal settings, fallback if needed
    model = None
    load_kwargs = {
        "dtype": optimal_dtype,
    }

    # Only add attn_implementation if not using default
    if attn_implementation != "eager":
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModel.from_pretrained(model_name, **load_kwargs).to(device).eval()
    except Exception as e:
        print(f"Warning: Failed to load with {attn_implementation} attention: {e}")
        print("Falling back to default attention implementation...")
        # Fallback: try without attention implementation specification
        try:
            load_kwargs.pop("attn_implementation", None)
            model = (
                AutoModel.from_pretrained(model_name, **load_kwargs).to(device).eval()
            )
        except Exception as e2:
            print(f"Warning: Failed with {optimal_dtype}, falling back to float32...")
            # Last resort: use float32
            load_kwargs["dtype"] = torch.float32
            model = (
                AutoModel.from_pretrained(model_name, **load_kwargs).to(device).eval()
            )

    processor = AutoProcessor.from_pretrained(model_name)

    dataset = ImageDataset(records)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    all_features = []
    all_indices = []

    # Determine autocast dtype based on model's dtype
    autocast_dtype = optimal_dtype if device.type == "cuda" else torch.float32

    with torch.no_grad():
        for imgs, indices in tqdm(dataloader, desc="Extracting features"):
            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=(device.type == "cuda"),
            ):
                features = model.get_image_features(**inputs)

            features = features.float().cpu().numpy()
            all_features.append(features)
            all_indices.extend(indices)

    if all_features:
        features = np.concatenate(all_features, axis=0)
    else:
        features = np.empty((0, 0), dtype=np.float32)

    return features, all_indices


def extract_features_on_gpu(
    records: List[ImgRec],
    gpu_id: int,
    model_name: str = "google/siglip2-base-patch16-naflex",
    batch_size: int = 128,
    gpu_memory_fraction: float = 0.9,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract CLIP features on a specific GPU.
    This function runs on a single GPU and returns features for the assigned subset of images.
    Returns: (features [N, D], valid_indices)
    """
    device = torch.device(f"cuda:{gpu_id}")
    print(f"GPU {gpu_id}: Using device {device}")

    # Set memory fraction for this GPU
    torch.cuda.set_device(gpu_id)
    try:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, gpu_id)
    except:
        # Ignore if memory fraction setting fails
        pass

    # Determine optimal dtype and attention implementation for this GPU
    optimal_dtype = get_optimal_dtype(device)
    attn_implementation = get_optimal_attention_implementation()

    # Adjust batch size based on GPU memory
    optimal_batch_size = get_optimal_batch_size(gpu_id, batch_size)

    print(f"GPU {gpu_id}: Loading model with {optimal_dtype} and {attn_implementation}")
    print(f"GPU {gpu_id}: Using batch size {optimal_batch_size}")

    # Use global lock to prevent concurrent model loading issues across GPUs
    with _model_loading_lock:
        print(f"GPU {gpu_id}: Acquired model loading lock")

        # Load model directly on target device without meta device to avoid conflicts
        model = None
        processor = None

        # Try different loading strategies in order of preference
        loading_strategies = [
            {
                "name": f"{optimal_dtype} with {attn_implementation}",
                "kwargs": {
                    "torch_dtype": optimal_dtype,
                    "attn_implementation": attn_implementation if attn_implementation != "eager" else None,
                }
            },
            {
                "name": f"{optimal_dtype} with eager attention",
                "kwargs": {
                    "torch_dtype": optimal_dtype,
                }
            },
            {
                "name": "float32 with eager attention",
                "kwargs": {
                    "torch_dtype": torch.float32,
                }
            }
        ]

        for strategy in loading_strategies:
            try:
                print(f"GPU {gpu_id}: Trying to load model with {strategy['name']}...")
                # Remove None values from kwargs
                load_kwargs = {k: v for k, v in strategy["kwargs"].items() if v is not None}

                # Clear GPU cache before loading
                torch.cuda.empty_cache()
                time.sleep(0.1)  # Brief pause to allow GPU cleanup

                # Load model directly to device without intermediate meta device
                model = AutoModel.from_pretrained(model_name, **load_kwargs)
                model = model.to(device).eval()

                # Load processor (can be done after model loading)
                processor = AutoProcessor.from_pretrained(model_name)

                print(f"GPU {gpu_id}: Successfully loaded model with {strategy['name']}")
                break

            except Exception as e:
                error_msg = str(e)
                print(f"GPU {gpu_id}: Failed to load with {strategy['name']}: {error_msg}")

                # Enhanced error handling for specific errors
                if "meta tensor" in error_msg:
                    print(f"GPU {gpu_id}: Meta tensor error detected, ensuring proper cleanup...")
                elif "out of memory" in error_msg:
                    print(f"GPU {gpu_id}: OOM during loading, attempting aggressive cleanup...")
                    torch.cuda.empty_cache()
                    time.sleep(0.5)  # Longer pause for OOM recovery

                if model is not None:
                    del model
                    model = None
                torch.cuda.empty_cache()
                continue

        print(f"GPU {gpu_id}: Releasing model loading lock")

    if model is None:
        raise RuntimeError(f"GPU {gpu_id}: Failed to load model with any strategy")

    dataset = ImageDataset(records)
    dataloader = DataLoader(
        dataset,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_workers=0,  # As requested to prevent init process issues
        collate_fn=collate_fn,
        pin_memory=True,  # Always use pin_memory for faster GPU transfer
    )

    all_features = []
    all_indices = []

    # Determine autocast dtype based on model's dtype
    autocast_dtype = optimal_dtype if device.type == "cuda" else torch.float32

    # Clear GPU cache before processing
    torch.cuda.empty_cache()

    with torch.no_grad():
        for batch_idx, (imgs, indices) in enumerate(tqdm(dataloader, desc=f"GPU {gpu_id} extracting", position=gpu_id, leave=False)):
            if not imgs:
                continue

            try:
                inputs = processor(images=imgs, return_tensors="pt").to(device)

                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=(device.type == "cuda"),
                ):
                    features = model.get_image_features(**inputs)

                features = features.float().cpu().numpy()
                all_features.append(features)
                all_indices.extend(indices)

                # Clear cache periodically to prevent memory buildup
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU {gpu_id}: OOM error, reducing batch size...")
                    # Reduce batch size and retry this batch
                    torch.cuda.empty_cache()
                    try:
                        # Retry with smaller batch
                        smaller_batch_size = len(imgs) // 2
                        if smaller_batch_size > 0:
                            smaller_imgs = imgs[:smaller_batch_size]
                            smaller_indices = indices[:smaller_batch_size]

                            inputs = processor(images=smaller_imgs, return_tensors="pt").to(device)
                            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")):
                                features = model.get_image_features(**inputs)

                            features = features.float().cpu().numpy()
                            all_features.append(features)
                            all_indices.extend(smaller_indices)
                    except:
                        print(f"GPU {gpu_id}: Failed to process batch even with reduced size, skipping...")
                        continue
                else:
                    print(f"GPU {gpu_id}: Error processing batch: {e}")
                    continue

    # Final cleanup
    torch.cuda.empty_cache()

    if all_features:
        features = np.concatenate(all_features, axis=0)
    else:
        features = np.empty((0, 0), dtype=np.float32)

    print(f"GPU {gpu_id}: Completed processing {len(all_indices)} images")
    return features, all_indices


def extract_clip_features_multigpu(
    records: List[ImgRec],
    model_name: str = "google/siglip2-base-patch16-naflex",
    batch_size: int = 128,
    num_gpus: int = None,
    gpu_memory_fraction: float = 0.9,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract CLIP features using multiple GPUs in parallel.
    Automatically detects available GPUs and distributes the workload.
    Falls back to single GPU or CPU if multi-GPU setup fails.
    Returns: (features [N, D], valid_indices)
    """
    # Detect available GPUs
    available_gpus = detect_available_gpus()

    if not available_gpus:
        print("No GPUs available, falling back to CPU")
        return extract_clip_features(records, model_name, batch_size)

    # Determine number of GPUs to use
    if num_gpus is None:
        num_gpus_to_use = len(available_gpus)
    else:
        num_gpus_to_use = min(num_gpus, len(available_gpus))

    if num_gpus_to_use <= 1:
        print(f"Using single GPU {available_gpus[0]}")
        return extract_features_on_gpu(
            records, available_gpus[0], model_name, batch_size, gpu_memory_fraction
        )

    print(f"Using {num_gpus_to_use} GPUs for parallel processing: {available_gpus[:num_gpus_to_use]}")

    # Split records across GPUs
    gpu_chunks = split_records_for_gpus(records, num_gpus_to_use)
    gpu_ids_to_use = available_gpus[:num_gpus_to_use]

    # Use ThreadPoolExecutor for parallel processing
    all_features = []
    all_indices = []
    completed_gpus = set()

    def process_gpu_chunk(gpu_idx_chunk):
        gpu_idx, chunk = gpu_idx_chunk
        try:
            if not chunk:
                print(f"GPU {gpu_idx}: No images to process")
                return gpu_idx, np.empty((0, 0), dtype=np.float32), [], None

            features, indices = extract_features_on_gpu(
                chunk,
                gpu_idx,
                model_name,
                batch_size,
                gpu_memory_fraction,
            )
            return gpu_idx, features, indices, None
        except Exception as e:
            error_msg = str(e)
            print(f"GPU {gpu_idx}: Error during processing: {error_msg}")

            # Enhanced error recovery for specific errors
            if "meta tensor" in error_msg:
                print(f"GPU {gpu_idx}: Meta tensor error during processing, attempting GPU cleanup...")
                try:
                    torch.cuda.set_device(gpu_idx)
                    torch.cuda.empty_cache()
                    time.sleep(1.0)  # Pause for recovery
                except:
                    pass
            elif "out of memory" in error_msg:
                print(f"GPU {gpu_idx}: OOM error during processing, attempting aggressive cleanup...")
                try:
                    torch.cuda.set_device(gpu_idx)
                    torch.cuda.empty_cache()
                    time.sleep(2.0)  # Longer pause for OOM recovery
                except:
                    pass

            return gpu_idx, np.empty((0, 0), dtype=np.float32), [], error_msg

    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=num_gpus_to_use) as executor:
        # Submit all GPU tasks
        future_to_gpu = {
            executor.submit(process_gpu_chunk, (gpu_id, chunk)): gpu_id
            for gpu_id, chunk in zip(gpu_ids_to_use, gpu_chunks)
        }

        # Create progress bar for overall progress
        pbar = tqdm(total=num_gpus_to_use, desc="Multi-GPU processing")

        # Collect results as they complete
        for future in as_completed(future_to_gpu):
            gpu_id = future_to_gpu[future]
            try:
                gpu_idx, features, indices, error = future.result()

                if error:
                    print(f"GPU {gpu_id}: Processing failed with error: {error}")
                else:
                    all_features.append(features)
                    all_indices.extend(indices)
                    completed_gpus.add(gpu_id)
                    print(f"GPU {gpu_id}: Successfully processed {len(indices)} images")

            except Exception as e:
                print(f"GPU {gpu_id}: Future failed: {e}")

            pbar.update(1)

        pbar.close()

    # Check if any GPUs completed successfully
    if not completed_gpus:
        print("All GPUs failed, falling back to CPU")
        return extract_clip_features(records, model_name, batch_size)

    # Combine results from all successful GPUs
    if all_features:
        combined_features = np.concatenate(all_features, axis=0)
    else:
        combined_features = np.empty((0, 0), dtype=np.float32)

    # Final cleanup of all GPUs
    try:
        for gpu_id in available_gpus[:num_gpus_to_use]:
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        time.sleep(0.5)  # Brief pause for final cleanup
    except:
        pass

    print(f"Multi-GPU processing completed:")
    print(f"  Successfully processed on GPUs: {sorted(completed_gpus)}")
    print(f"  Total features extracted: {combined_features.shape}")
    print(f"  Total valid indices: {len(all_indices)}")

    return combined_features, all_indices


# ---------------------------
# Similarity & Grouping
# ---------------------------


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize feature vectors for cosine similarity"""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return features / norms


class UnionFind:
    """Union-Find data structure with path compression"""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def get_groups(self) -> List[List[int]]:
        """Return all connected components"""
        groups_map = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            groups_map.setdefault(root, []).append(i)
        return list(groups_map.values())


def build_groups_clip(
    records: List[ImgRec],
    features: np.ndarray,
    valid_indices: List[int],
    threshold: float = 0.985,
    k: int = 10,
) -> List[List[int]]:
    """
    Build duplicate groups using CLIP features and FAISS k-NN search.
    Returns list of groups (each group is a list of record indices).
    """
    N = len(valid_indices)
    if N == 0:
        return []

    # L2 normalize for cosine similarity
    feats = l2_normalize(features.astype(np.float32))

    # Build FAISS index (GPU if available)
    D = feats.shape[1]
    if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
        print("Using GPU FAISS index")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, D)
    else:
        print("Using CPU FAISS index")
        index = faiss.IndexFlatIP(D)

    index.add(feats)

    # Search k nearest neighbors
    k_eff = min(k, N)
    print(f"Searching top-{k_eff} neighbors...")
    similarities, neighbor_ids = index.search(feats, k_eff)

    # Union-Find clustering
    uf = UnionFind(N)

    for i in range(N):
        for j in range(k_eff):
            neighbor = neighbor_ids[i, j]
            if neighbor == i or neighbor < 0:
                continue
            if similarities[i, j] >= threshold:
                uf.union(i, neighbor)

    # Get groups and map back to original record indices
    local_groups = uf.get_groups()
    groups = [[valid_indices[i] for i in group] for group in local_groups]

    # Return only groups with duplicates
    return [g for g in groups if len(g) > 1]


# ---------------------------
# Representative Selection
# ---------------------------


def pick_representative(group: List[int], records: List[ImgRec], policy: str) -> int:
    """
    Select which file to keep from a duplicate group based on policy.
    Returns the index of the file to keep.
    """
    if policy == "lexi":
        return min(group, key=lambda i: records[i].path)
    elif policy == "smallest":
        return min(group, key=lambda i: records[i].size)
    elif policy == "largest":
        return max(group, key=lambda i: records[i].size)
    elif policy == "newest":
        return max(group, key=lambda i: records[i].mtime)
    elif policy == "oldest":
        return min(group, key=lambda i: records[i].mtime)
    else:
        raise ValueError(f"Unknown keep policy: {policy}")


# ---------------------------
# Report Generation
# ---------------------------


def make_report(
    records: List[ImgRec], groups: List[List[int]], keep_policy: str
) -> Dict:
    """
    Generate JSON report with duplicate groups and statistics.
    """
    report_groups = []
    total_duplicates = 0

    for group in groups:
        rep_idx = pick_representative(group, records, keep_policy)
        duplicates = [records[i].path for i in group if i != rep_idx]
        total_duplicates += len(duplicates)

        report_groups.append({"keep": records[rep_idx].path, "duplicates": duplicates})

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(records),
        "duplicate_groups": len(groups),
        "total_duplicates": total_duplicates,
        "keep_policy": keep_policy,
        "groups": report_groups,
    }

    return report


# ---------------------------
# File Deletion
# ---------------------------


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
        "has_more_errors": len(errors) > 10
    }


# ---------------------------
# Main Entry Point
# ---------------------------


def main():
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

    args = parser.parse_args()

    # Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory")
        sys.exit(1)

    # Validate GPU memory fraction
    if not 0.1 <= args.gpu_memory_fraction <= 1.0:
        print(f"Error: --gpu-memory-fraction must be between 0.1 and 1.0, got {args.gpu_memory_fraction}")
        sys.exit(1)

    # Set default report path
    if args.report is None:
        args.report = os.path.join(args.folder, "dedup_report.json")

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
        gpu_memory_fraction=args.gpu_memory_fraction
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

        if result['actual_errors'] > 0:
            print(f"\nEncountered {result['actual_errors']} real errors:")
            for err in result['error_details']:
                print(f"  - {err}")
            if result['has_more_errors']:
                print(f"  ... and {result['actual_errors'] - 10} more")
        else:
            if result['successfully_deleted'] > 0:
                print(f"✓ Successfully deleted {result['successfully_deleted']} duplicate files")
            if result['already_deleted'] > 0:
                print(f"✓ Found {result['already_deleted']} files already removed")
    else:
        print("\nDry run complete. No files deleted.")
        print(f"To delete duplicates, run again with --inplace flag")

    print("\nDone.")


if __name__ == "__main__":
    main()
