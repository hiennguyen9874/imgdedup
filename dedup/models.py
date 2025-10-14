"""
CLIP model loading and feature extraction with multi-GPU support.
"""

import threading
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

from .hardware import (
    get_optimal_dtype,
    get_optimal_attention_implementation,
    detect_available_gpus,
)
from .filesystem import ImgRec, get_optimal_batch_size


# Global lock for synchronized model loading across GPUs
_model_loading_lock = threading.Lock()


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
                    "attn_implementation": (
                        attn_implementation if attn_implementation != "eager" else None
                    ),
                },
            },
            {
                "name": f"{optimal_dtype} with eager attention",
                "kwargs": {
                    "torch_dtype": optimal_dtype,
                },
            },
            {
                "name": "float32 with eager attention",
                "kwargs": {
                    "torch_dtype": torch.float32,
                },
            },
        ]

        for strategy in loading_strategies:
            try:
                print(f"GPU {gpu_id}: Trying to load model with {strategy['name']}...")
                # Remove None values from kwargs
                load_kwargs = {
                    k: v for k, v in strategy["kwargs"].items() if v is not None
                }

                # Clear GPU cache before loading
                torch.cuda.empty_cache()
                time.sleep(0.1)  # Brief pause to allow GPU cleanup

                # Load model directly to device without intermediate meta device
                model = AutoModel.from_pretrained(model_name, **load_kwargs)
                model = model.to(device).eval()

                # Load processor (can be done after model loading)
                processor = AutoProcessor.from_pretrained(model_name)

                print(
                    f"GPU {gpu_id}: Successfully loaded model with {strategy['name']}"
                )
                break

            except Exception as e:
                error_msg = str(e)
                print(
                    f"GPU {gpu_id}: Failed to load with {strategy['name']}: {error_msg}"
                )

                # Enhanced error handling for specific errors
                if "meta tensor" in error_msg:
                    print(
                        f"GPU {gpu_id}: Meta tensor error detected, ensuring proper cleanup..."
                    )
                elif "out of memory" in error_msg:
                    print(
                        f"GPU {gpu_id}: OOM during loading, attempting aggressive cleanup..."
                    )
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
        for batch_idx, (imgs, indices) in enumerate(
            tqdm(
                dataloader,
                desc=f"GPU {gpu_id} extracting",
                position=gpu_id,
                leave=False,
            )
        ):
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

                            inputs = processor(
                                images=smaller_imgs, return_tensors="pt"
                            ).to(device)
                            with torch.autocast(
                                device_type=device.type,
                                dtype=autocast_dtype,
                                enabled=(device.type == "cuda"),
                            ):
                                features = model.get_image_features(**inputs)

                            features = features.float().cpu().numpy()
                            all_features.append(features)
                            all_indices.extend(smaller_indices)
                    except:
                        print(
                            f"GPU {gpu_id}: Failed to process batch even with reduced size, skipping..."
                        )
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
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .filesystem import split_records_for_gpus

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

    print(
        f"Using {num_gpus_to_use} GPUs for parallel processing: {available_gpus[:num_gpus_to_use]}"
    )

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
                print(
                    f"GPU {gpu_idx}: Meta tensor error during processing, attempting GPU cleanup..."
                )
                try:
                    torch.cuda.set_device(gpu_idx)
                    torch.cuda.empty_cache()
                    time.sleep(1.0)  # Pause for recovery
                except:
                    pass
            elif "out of memory" in error_msg:
                print(
                    f"GPU {gpu_idx}: OOM error during processing, attempting aggressive cleanup..."
                )
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
