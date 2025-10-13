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


# ---------------------------
# Hardware Capability Detection
# ---------------------------


def check_flash_attention_available() -> bool:
    """Check if flash attention 2 is available"""
    try:
        import flash_attn

        return True
    except ImportError:
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


def delete_duplicates(report: Dict) -> List[str]:
    """
    Delete duplicate files based on report.
    Returns list of errors encountered.
    """
    errors = []
    total_to_delete = sum(len(g["duplicates"]) for g in report["groups"])

    with tqdm(total=total_to_delete, desc="Deleting duplicates") as pbar:
        for group in report["groups"]:
            for dup_path in group["duplicates"]:
                try:
                    os.remove(dup_path)
                    pbar.update(1)
                except Exception as e:
                    errors.append(f"Failed to delete {dup_path}: {e}")
                    pbar.update(1)

    return errors


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

    # Set default report path
    if args.report is None:
        args.report = os.path.join(args.folder, "dedup_report.json")

    print("=" * 60)
    print("Image Deduplication Tool - CLIP Mode")
    print("=" * 60)
    print(f"Folder: {args.folder}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Top-k: {args.k}")
    print(f"Batch size: {args.batch_size}")
    print(f"Keep policy: {args.keep_policy}")
    print(
        f"Mode: {'IN-PLACE (will delete)' if args.inplace else 'DRY RUN (no deletion)'}"
    )
    print(f"Report: {args.report}")
    print("=" * 60)
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
    features, valid_indices = extract_clip_features(
        records, model_name=args.model, batch_size=args.batch_size
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
        errors = delete_duplicates(report)

        if errors:
            print(f"\nEncountered {len(errors)} errors:")
            for err in errors[:10]:  # Show first 10
                print(f"  - {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        else:
            print(f"Successfully deleted {report['total_duplicates']} duplicate files")
    else:
        print("\nDry run complete. No files deleted.")
        print(f"To delete duplicates, run again with --inplace flag")

    print("\nDone.")


if __name__ == "__main__":
    main()
