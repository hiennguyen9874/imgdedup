# Image Deduplication Tool

A powerful and flexible image duplicate detection and removal tool supporting multiple detection algorithms from exact binary matches to semantic similarity.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Configuration Options](#configuration-options)
- [Output Format](#output-format)
- [Performance Considerations](#performance-considerations)
- [Examples](#examples)

---

## Overview

`imgdedup` is a Python-based tool designed to scan directories recursively, identify duplicate or near-duplicate images based on semantic similarity, and optionally remove them based on configurable policies. It uses CLIP (SigLIP2) vision transformers for intelligent image comparison.

The tool can operate in two modes:
- **Dry run** (default): Generate a JSON report without deleting files
- **In-place** (`--inplace`): Delete duplicates immediately after detection

---

## Features

✅ **Semantic Similarity Detection**
- Deep learning-based detection using CLIP (SigLIP2) vision transformers
- Detects visually and semantically similar content
- Robust to transformations (resize, crop, rotation, color adjustments)

✅ **Flexible Keep Policies**
- Lexicographic (alphabetical path)
- File size (smallest or largest)
- Modification time (newest or oldest)

✅ **Safe by Default**
- Dry run mode prevents accidental deletions
- Comprehensive JSON reports for review
- Progress bars for long operations

✅ **Performance Optimized**
- GPU acceleration support for CLIP mode
- Batch processing for neural network inference
- Efficient file scanning with metadata caching

---

## Installation

### CPU Version

```bash
pip install numpy pillow tqdm torch transformers faiss-cpu
```

### GPU Version (Recommended)

```bash
# For CUDA 12.1
pip install tqdm "numpy<2" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45" faiss-gpu
```

### Dependencies

**Required:**
- `numpy` - Array operations
- `Pillow` - Image loading and processing
- `tqdm` - Progress bars
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face model loading
- `faiss-cpu` or `faiss-gpu` - Fast similarity search

---

## Quick Start

```bash
# Scan for duplicates (dry run)
python main.py /path/to/images

# Find semantically similar images with custom threshold
python main.py /path/to/images --threshold 0.98 --report my_report.json

# Remove duplicates, keeping newest files
python main.py /path/to/images --keep-policy newest --inplace
```

---

## Usage

### Basic Syntax

```bash
python main.py <folder> [options]
```

### Arguments

| Argument        | Type   | Default                              | Description                                               |
| --------------- | ------ | ------------------------------------ | --------------------------------------------------------- |
| `folder`        | str    | Required                             | Root folder to scan recursively                           |
| `--threshold`   | float  | 0.985                                | Cosine similarity threshold (0.0-1.0)                     |
| `--k`           | int    | 10                                   | Top-k neighbors to search                                 |
| `--batch-size`  | int    | 128                                  | Batch size for inference                                  |
| `--model`       | str    | `google/siglip2-base-patch16-naflex` | Hugging Face model name                                   |
| `--keep-policy` | choice | `lexi`                               | Policy: `lexi`, `smallest`, `largest`, `newest`, `oldest` |
| `--inplace`     | flag   | False                                | Delete duplicates (otherwise dry run)                     |
| `--report`      | str    | `<folder>/dedup_report.json`         | Path to output JSON report                                |

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Command Line Interface                  │
│                        (argparse)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    File System Scanner                      │
│  • Recursive directory traversal                            │
│  • Image file filtering (jpg, png, bmp, webp, tif)          │
│  • Metadata extraction (size, mtime)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Hardware Capability Detection                  │
│  • Check flash attention availability                       │
│  • Check bfloat16 GPU support                               │
│  • Determine optimal dtype and attention                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  CLIP Feature Extraction                    │
│  • SigLIP2 vision transformer model                         │
│  • Batched GPU/CPU inference                                │
│  • Flash Attention 2 / SDPA / Eager attention               │
│  • bfloat16 / float16 / float32 dtype                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 FAISS Similarity Search                     │
│  • L2 normalization for cosine similarity                   │
│  • GPU/CPU k-NN search                                      │
│  • Top-k neighbor retrieval                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Grouping Engine                          │
│  • Union-Find algorithm for clustering                      │
│  • Threshold-based duplicate group formation                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Representative Selection                       │
│  • Apply keep policy to each group                          │
│  • Mark files for retention/deletion                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Report Generation                        │
│  • JSON output with metadata                                │
│  • Statistics summary                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Optional File Deletion                         │
│  • Only if --inplace flag is set                            │
│  • Error handling and logging                               │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **File Scanner (`scan_images`)**
- **Purpose**: Discovers all image files in the target directory tree
- **Process**:
  1. Recursively traverses directory structure
  2. Filters by supported image extensions
  3. Extracts file metadata (size, modification time)
  4. Returns list of `ImgRec` objects
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`

#### 2. **Hardware Detection**
- **Flash Attention**: Automatically detects if `flash-attn` package is installed
- **bfloat16 Support**: Checks GPU compute capability (Ampere/8.0+ for optimal support)
- **Attention Priority**: flash_attention_2 > sdpa > eager
- **Dtype Priority**: bfloat16 > float16 > float32

#### 3. **CLIP Feature Extraction (`extract_clip_features`)**
- Loads SigLIP2 vision transformer model with optimal settings
- Automatic fallback: tries flash attention → SDPA → eager
- Automatic fallback: tries bfloat16 → float16 → float32
- Batched GPU/CPU inference with progress tracking
- Returns L2-normalized feature vectors

#### 4. **FAISS Similarity Search (`build_groups_clip`)**
- L2-normalizes features for cosine similarity
- Creates GPU or CPU FAISS index (IndexFlatIP)
- Performs k-NN search for each image
- Groups images with similarity ≥ threshold
- O(n·k·log n) complexity

#### 5. **Grouping Engine (Union-Find)**
- **Data structure**: Disjoint Set Union (DSU) with path compression
- **Purpose**: Efficiently merge similar images into groups
- **Optimization**: Path halving during `find()` operations
- **Output**: List of duplicate groups (indices into original records)

#### 6. **Representative Selector (`pick_representative`)**
- **Purpose**: Choose one file to keep from each duplicate group
- **Policies**:
  - `lexi`: Alphabetically first path
  - `smallest`: Smallest file size
  - `largest`: Largest file size
  - `newest`: Most recent modification time
  - `oldest`: Oldest modification time

#### 7. **Report Generator (`make_report`)**
- Creates structured JSON output
- Includes metadata: timestamp, counts, policy
- Lists all groups with keep/delete decisions

#### 8. **Deletion Engine (`delete_duplicates`)**
- Only executes if `--inplace` flag is set
- Attempts to remove each duplicate file
- Collects and reports errors without stopping

---

## Algorithm

### CLIP Semantic Similarity (SigLIP2)

**Use Case**: Finding semantically similar images even if visually different (e.g., different photos of same object, similar scenes)

**How it works**:
1. **Hardware optimization**:
   - Automatically detects flash attention support
   - Checks GPU compute capability for bfloat16
   - Selects optimal attention: flash_attention_2 > sdpa > eager
   - Selects optimal dtype: bfloat16 > float16 > float32
   - Automatic fallback on errors

2. **Feature extraction**:
   - Load SigLIP2 vision transformer model with optimal settings
   - Process images in batches (GPU acceleration if available)
   - Extract high-dimensional embedding vectors (typically 768-D)
   - L2-normalize features for cosine similarity

3. **Index building**:
   - Create FAISS index (GPU or CPU)
   - Add all feature vectors to index
   - Index type: Flat Inner Product (exact search)

4. **Similarity search**:
   - For each image, find k nearest neighbors
   - Compute cosine similarity (dot product of normalized vectors)
   - Similarity range: -1 (opposite) to 1 (identical)

5. **Clustering**:
   - Use Union-Find to group images
   - Link pairs with similarity ≥ threshold
   - Form transitive closure of similar images

**Strengths**:
- ✅ Understands semantic content (cats, dogs, buildings, etc.)
- ✅ Robust to extreme transformations (crop, rotate, flip, color)
- ✅ Can find "similar but not duplicate" images
- ✅ Efficient k-NN search with FAISS
- ✅ GPU acceleration with flash attention and bfloat16
- ✅ Automatic hardware optimization
- ✅ Memory-efficient attention mechanisms

**Limitations**:
- ❌ Requires large dependencies (torch, transformers, faiss)
- ❌ First run downloads ~1GB model
- ❌ High memory usage (stores all features in RAM)
- ❌ May group conceptually similar but distinct images

**Parameters**:
- `--threshold`: Cosine similarity (0.0-1.0)
  - `0.99`: Very strict (near-identical images)
  - `0.985`: Default (good balance)
  - `0.95`: Loose (semantically similar)
- `--k`: Number of neighbors to check (default: 10)
- `--batch-size`: Inference batch size (default: 128)
- `--model`: Hugging Face model name

**Complexity**: O(n·k·log n) time with FAISS, O(n·d) space (d=embedding dimension)

**Performance Enhancements**:
- **Flash Attention 2**: Up to 2-4x faster inference, lower memory usage
- **bfloat16**: Up to 2x faster on Ampere GPUs (A100, RTX 30xx/40xx)
- **SDPA**: PyTorch 2.0+ optimized attention (fallback for no flash-attn)

**Example**:
```bash
# Default settings (balanced, auto-optimized)
python main.py /data/photos

# Strict semantic matching
python main.py /data/photos --threshold 0.99 --k 20

# GPU-accelerated batch processing
python main.py /data/photos --batch-size 256
```

---

## Configuration Options

### Keep Policies

Determines which file to keep when duplicates are found:

| Policy     | Keeps                     | Use Case                   |
| ---------- | ------------------------- | -------------------------- |
| `lexi`     | Alphabetically first path | Predictable, deterministic |
| `smallest` | Smallest file size        | Save disk space            |
| `largest`  | Largest file size         | Keep highest quality       |
| `newest`   | Most recent modification  | Keep latest version        |
| `oldest`   | Oldest modification       | Keep original version      |

### Threshold Tuning

**Cosine Similarity Threshold**:
```bash
# Conservative (fewer false positives)
--threshold 0.99-1.0   # Near-identical content

# Moderate (recommended)
--threshold 0.97-0.99  # Clearly similar images

# Aggressive (semantic similarity)
--threshold 0.90-0.97  # Same category/concept
```

### Optimization Options

**Flash Attention** (Optional, for faster inference):
```bash
# Install flash attention for 2-4x speedup
pip install -U flash-attn --no-build-isolation

# Tool automatically detects and uses it
python main.py /data/photos
```

**GPU Selection** (for multi-GPU systems):
```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python main.py /data/photos

# Use multiple GPUs (feature extraction is single-GPU, but you can run multiple instances)
CUDA_VISIBLE_DEVICES=1 python main.py /data/photos2
```

---

## Output Format

### JSON Report Structure

```json
{
  "generated_at": "2025-10-13 14:30:00",
  "total_images": 1523,
  "duplicate_groups": 47,
  "total_duplicates": 112,
  "keep_policy": "lexi",
  "groups": [
    {
      "keep": "/data/photos/IMG_001.jpg",
      "duplicates": [
        "/data/photos/IMG_001_copy.jpg",
        "/data/photos/backup/IMG_001.jpg"
      ]
    },
    {
      "keep": "/data/photos/vacation/beach.png",
      "duplicates": [
        "/data/photos/vacation/beach_resized.jpg"
      ]
    }
  ]
}
```

### Report Fields

- `generated_at`: Timestamp of scan completion
- `total_images`: Number of image files scanned
- `duplicate_groups`: Number of duplicate clusters found
- `total_duplicates`: Total files marked for deletion
- `keep_policy`: Policy used for representative selection
- `groups`: Array of duplicate groups
  - `keep`: Path to file being kept
  - `duplicates`: Array of paths to be deleted

---

## Performance Considerations

### Hardware Configurations

| Configuration                          | Speed | Memory | Best For                     |
| -------------------------------------- | ----- | ------ | ---------------------------- |
| **Ampere GPU + Flash Attn + bfloat16** | ⚡⚡⚡⚡  | Low    | Large datasets (>10k images) |
| **GPU + float16**                      | ⚡⚡⚡   | Medium | Most use cases               |
| **CPU + float32**                      | ⚡     | Low    | Small datasets (<1k images)  |

### Scaling Characteristics

**GPU with Flash Attention (A100, RTX 40xx)**:
- 1,000 images: ~30-60 seconds
- 10,000 images: ~5-10 minutes
- 100,000 images: ~45-90 minutes

**GPU without Flash Attention (older GPUs)**:
- 1,000 images: ~1-2 minutes
- 10,000 images: ~10-20 minutes
- 100,000 images: ~1.5-3 hours

**CPU Mode**:
- 1,000 images: ~5-15 minutes
- 10,000 images: ~1-3 hours
- 100,000 images: ~10-20 hours

### Optimization Tips

1. **Install Flash Attention** (highly recommended for GPU):
   ```bash
   pip install -U flash-attn --no-build-isolation
   ```
   - 2-4x faster inference
   - Lower memory usage
   - Automatic detection and fallback

2. **Use Ampere or newer GPUs** for best performance:
   - A100, A6000, RTX 30xx/40xx series
   - Native bfloat16 support
   - Better flash attention performance

3. **Optimize batch size**:
   - Increase `--batch-size` if GPU memory allows (try 256 or 512)
   - Reduce to 32 or 64 if you encounter OOM errors
   - Default 128 works well for most GPUs

4. **Adjust k-neighbors** for speed:
   - Decrease `--k` for faster search (try 5 for very large datasets)
   - Increase `--k` for more thorough duplicate detection
   - Default 10 is a good balance

5. **Large datasets** (>100k images):
   - Process in subdirectories if possible
   - Use higher thresholds (0.99) to reduce groups
   - Consider using faiss-gpu for faster indexing

6. **Memory constraints**:
   - Total RAM needed: ~4-8 GB + model (~2 GB) + features (~N × 3KB)
   - Reduce batch size if OOM errors occur
   - Use CPU mode if GPU memory insufficient (slower but works)

---

## Examples

### Example 1: Safe Duplicate Detection

```bash
# Scan without deleting anything (dry run)
python main.py ~/Pictures --threshold 0.985 --report ~/duplicates.json

# Review the report
cat ~/duplicates.json

# If satisfied, run with --inplace
python main.py ~/Pictures --threshold 0.985 --inplace
```

### Example 2: Photo Library Deduplication

```bash
# Find near-duplicates using GPU-accelerated semantic analysis
python main.py ~/PhotoLibrary \
  --threshold 0.98 \
  --batch-size 256 \
  --keep-policy largest \
  --report ~/photo_dedup.json
```

### Example 3: Strict Duplicate Detection

```bash
# Only find very similar images (fewer false positives)
python main.py ~/Downloads/images \
  --threshold 0.99 \
  --keep-policy newest \
  --inplace
```

### Example 4: Custom Model

```bash
# Use different vision model
python main.py ~/images \
  --model google/siglip2-so400m-patch14-384 \
  --threshold 0.99 \
  --k 5
```

### Example 5: Large Dataset Processing

```bash
# Optimize for large datasets with flash attention
pip install -U flash-attn --no-build-isolation

python main.py ~/large_photo_archive \
  --threshold 0.985 \
  --batch-size 512 \
  --k 5 \
  --keep-policy largest \
  --report ~/large_dedup.json
```

### Example 6: Aggressive Semantic Matching

```bash
# Find conceptually similar images (more aggressive)
python main.py ~/art_collection \
  --threshold 0.92 \
  --k 20 \
  --keep-policy largest \
  --report ~/review_before_delete.json
```

---

## Safety Features

### Built-in Safeguards

1. **Dry run by default**: Files are never deleted unless `--inplace` is explicitly set
2. **Comprehensive reporting**: JSON report generated before any deletions
3. **Error isolation**: Deletion errors don't stop the process
4. **Progress feedback**: tqdm progress bars for all long operations
5. **Empty folder handling**: Gracefully handles directories with no images

### Best Practices

✅ **DO**:
- Always run without `--inplace` first (dry run)
- Review the JSON report before deleting
- Back up important files before using `--inplace`
- Start with conservative thresholds (0.985-0.99)
- Test on small directory subset first
- Install flash-attn for optimal performance

❌ **DON'T**:
- Use `--inplace` on the first run without reviewing
- Use very low thresholds (<0.90) without careful review
- Run on directories without backups
- Assume semantic similarity means exact duplicates

---

## Troubleshooting

### Common Issues

**Issue**: `ImportError: cannot import name 'faiss'`
- **Solution**: Install FAISS with `pip install faiss-cpu` (or `faiss-gpu` for GPU support)

**Issue**: `CUDA out of memory` error
- **Solution**: Reduce `--batch-size` to 32 or 64, or use CPU mode

**Issue**: Too many false positives
- **Solution**: Increase `--threshold` to 0.99 or higher for stricter matching

**Issue**: Missing obvious duplicates
- **Solution**: Decrease `--threshold` to 0.95-0.97, or increase `--k` to 20

**Issue**: Slow performance on GPU
- **Solution**: 
  1. Install flash attention: `pip install -U flash-attn --no-build-isolation`
  2. Increase batch size if you have memory
  3. Check that GPU is actually being used (tool will print "Using device: cuda")

**Issue**: Model download fails
- **Solution**: Check internet connection, or manually download model and specify local path

**Issue**: Flash attention import error
- **Solution**: This is normal if not installed. Tool will fallback to SDPA or eager attention automatically

**Issue**: "bfloat16 not supported" warnings
- **Solution**: Normal for older GPUs. Tool automatically falls back to float16

---

## License

This tool is provided as-is for duplicate image detection and removal.

## Contributing

Suggestions for improvements:
- Support for video duplicate detection
- Add interactive mode for manual review
- Multi-GPU support for parallel processing
- Support for additional vision transformer models
- Incremental indexing for very large datasets
- Web UI for reviewing duplicates before deletion

---

## Technical Details

### Supported Image Formats
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- WebP (`.webp`)
- TIFF (`.tif`, `.tiff`)

### System Requirements

**Minimum**:
- Python 3.8+
- 8 GB RAM
- Any CPU (slow but works)

**Recommended**:
- Python 3.9+
- 16 GB RAM
- NVIDIA GPU with 8+ GB VRAM (RTX 30xx/40xx or better)
- CUDA 11.8 or 12.1
- Flash Attention 2 installed

**Optimal** (for large datasets):
- Python 3.10+
- 32 GB RAM
- NVIDIA Ampere or newer GPU (A100, RTX 40xx)
- CUDA 12.1
- Flash Attention 2
- faiss-gpu installed

### Model Information

Default model: `google/siglip2-base-patch16-naflex`
- Architecture: Vision Transformer (ViT)
- Parameters: ~150M
- Download size: ~600 MB
- Feature dimension: 768
- Input resolution: Flexible (naflex variant)
- Training: Contrastive learning on image-text pairs

**Performance Optimizations**:
- **Flash Attention 2**: Memory-efficient attention with 2-4x speedup
- **bfloat16**: Native support on Ampere GPUs for 2x faster inference
- **SDPA**: PyTorch 2.0+ optimized attention (automatic fallback)

---

## Changelog

### Version 2.0 (Current)
- **CLIP-only implementation** with SigLIP2 model
- **Flash Attention 2 support** for 2-4x faster inference
- **Automatic bfloat16 detection** for Ampere GPUs
- **Intelligent fallback system** for dtype and attention
- Automatic hardware capability detection
- Optimized for large-scale datasets
- Improved error handling and user feedback

### Version 1.0 (Legacy)
- Initial release with three detection algorithms (exact, pHash, CLIP)
- Union-Find based clustering
- JSON reporting
- Multiple keep policies


