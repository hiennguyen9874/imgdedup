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

- Multi-GPU support for parallel processing
- GPU acceleration support for CLIP mode
- Batch processing for neural network inference
- Efficient file scanning with metadata caching
- Automatic workload distribution across available GPUs

---

## Installation

```bash
uv sync --no-group dev

MAX_JOBS=$(nproc) uv sync --group dev
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

| Argument                | Type   | Default                              | Description                                               |
| ----------------------- | ------ | ------------------------------------ | --------------------------------------------------------- |
| `folder`                | str    | Required                             | Root folder to scan recursively                           |
| `--threshold`           | float  | 0.985                                | Cosine similarity threshold (0.0-1.0)                     |
| `--k`                   | int    | 10                                   | Top-k neighbors to search                                 |
| `--batch-size`          | int    | 128                                  | Batch size for inference                                  |
| `--model`               | str    | `google/siglip2-base-patch16-naflex` | Hugging Face model name                                   |
| `--gpus`                | int    | all available                        | Number of GPUs to use for parallel processing             |
| `--gpu-memory-fraction` | float  | 0.9                                  | GPU memory fraction to use per GPU (0.1-1.0)              |
| `--keep-policy`         | choice | `lexi`                               | Policy: `lexi`, `smallest`, `largest`, `newest`, `oldest` |
| `--inplace`             | flag   | False                                | Delete duplicates (otherwise dry run)                     |
| `--report`              | str    | `<folder>/dedup_report.json`         | Path to output JSON report                                |

---

## Project Structure

```
imgdedup/
├── main.py                 # Entry point (13 lines)
├── dedup/                  # Main package
│   ├── __init__.py         # Package exports and metadata
│   ├── cli.py              # Command-line interface and orchestration
│   ├── hardware.py         # Hardware detection and optimization
│   ├── filesystem.py       # File system operations and scanning
│   ├── models.py           # CLIP model operations and feature extraction
│   ├── similarity.py       # FAISS indexing and clustering algorithms
│   ├── reporting.py        # Report generation and analysis
│   └── deletion.py         # Safe file deletion operations
├── README.md
├── pyproject.toml
└── .gitignore
```

## Architecture

The tool follows a clean modular architecture with clear separation of concerns:

### Core Modules

- **`dedup/cli.py`**: Command-line interface and main workflow orchestration (225 lines)
- **`dedup/hardware.py`**: Hardware detection, GPU optimization, and capability testing (156 lines)
- **`dedup/filesystem.py`**: File scanning, metadata handling, and workload distribution (108 lines)
- **`dedup/models.py`**: CLIP model operations and multi-GPU feature extraction (476 lines)
- **`dedup/similarity.py`**: FAISS indexing and Union-Find clustering algorithms (103 lines)
- **`dedup/reporting.py`**: Report generation and duplicate analysis (54 lines)
- **`dedup/deletion.py`**: Safe file deletion with error handling and progress tracking (49 lines)

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Command Line Interface                  │
│                        (dedup/cli.py)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    File System Scanner                      │
│  • Recursive directory traversal                            │
│  • Image file filtering (jpg, png, bmp, webp, tif)          │
│  • Metadata extraction (size, mtime)                        │
│  • Workload distribution for multi-GPU                      │
│                    (dedup/filesystem.py)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Hardware Capability Detection                  │
│  • Check flash attention availability                       │
│  • Check bfloat16 GPU support                               │
│  • Determine optimal dtype and attention                    │
│  • Detect available GPUs and memory                         │
│                     (dedup/hardware.py)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Multi-GPU Orchestration                     │
│  • Automatic GPU detection and selection                    │
│  • Intelligent workload distribution                       │
│  • Parallel processing with ThreadPoolExecutor             │
│  • Per-GPU batch size optimization                         │
│                    (dedup/models.py)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Multi-GPU CLIP Feature Extraction             │
│  • SigLIP2 vision transformer on each GPU                  │
│  • Independent batched inference per GPU                   │
│  • Memory management and OOM handling                       │
│  • Automatic fallback to single GPU/CPU                    │
│  • Flash Attention 2 / SDPA / Eager per GPU                │
│  • bfloat16 / float16 / float32 per GPU                    │
│                    (dedup/models.py)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 FAISS Similarity Search                     │
│  • L2 normalization for cosine similarity                   │
│  • GPU/CPU k-NN search                                      │
│  • Top-k neighbor retrieval                                 │
│                   (dedup/similarity.py)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Grouping Engine                          │
│  • Union-Find algorithm for clustering                      │
│  • Threshold-based duplicate group formation                │
│                   (dedup/similarity.py)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Representative Selection                       │
│  • Apply keep policy to each group                          │
│  • Mark files for retention/deletion                        │
│                   (dedup/reporting.py)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Report Generation                        │
│  • JSON output with metadata                                │
│  • Statistics summary                                       │
│                   (dedup/reporting.py)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Optional File Deletion                         │
│  • Only if --inplace flag is set                            │
│  • Error handling and logging                               │
│                    (dedup/deletion.py)                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Functions by Module

#### **dedup/filesystem.py**

- `scan_images()`: Recursive image scanning with metadata extraction
- `split_records_for_gpus()`: Intelligent workload distribution across GPUs
- `get_optimal_batch_size()`: Per-GPU batch size optimization
- `ImgRec`: Dataclass for file metadata

#### **dedup/hardware.py**

- `check_flash_attention_available()`: Flash Attention detection
- `detect_available_gpus()`: GPU capability and memory detection
- `print_gpu_info()`: Hardware status reporting
- `get_optimal_dtype_and_attention()`: Automatic hardware optimization

#### **dedup/models.py**

- `extract_clip_features()`: Single-GPU CLIP feature extraction
- `extract_clip_features_multigpu()`: Multi-GPU parallel processing
- `ImageDataset`: PyTorch dataset for batched image loading
- Comprehensive error handling and OOM recovery

#### **dedup/similarity.py**

- `build_groups_clip()`: FAISS-based similarity search and clustering
- `l2_normalize()`: Feature vector normalization
- `UnionFind`: Efficient clustering data structure

#### **dedup/reporting.py**

- `make_report()`: Duplicate analysis and structured report generation
- `pick_representative()`: Policy-based file selection logic

#### **dedup/deletion.py**

- `delete_duplicates()`: Safe file deletion with progress tracking
- Comprehensive error handling and detailed statistics

#### **dedup/cli.py**

- `main()`: Complete workflow orchestration and user interface
- `parse_args()`: Command-line argument processing
- `validate_args()`: Input validation and error checking

---

## Development Benefits

### Modular Architecture Advantages

The refactored modular structure provides significant benefits for development and maintenance:

**✅ Clean Separation of Concerns**

- Each module has a single, well-defined responsibility
- Hardware detection is isolated from file operations
- Model operations are separated from similarity calculations
- Clear interfaces between components

**✅ Maintainability**

- Easy to locate and modify specific functionality
- Reduced risk of introducing bugs when making changes
- Clear dependency chains between modules
- Self-documenting code structure

**✅ Testability**

- Individual modules can be unit tested in isolation
- Mock dependencies easily for focused testing
- Clear interfaces make testing straightforward
- Better code coverage potential

**✅ Extensibility**

- New similarity algorithms can be added to `similarity.py`
- Additional file formats supported in `filesystem.py`
- New models integrated in `models.py`
- Custom report formats in `reporting.py`

**✅ Reusability**

- Modules can be imported and used independently
- Hardware detection utilities useful for other projects
- CLIP feature extraction reusable for other vision tasks
- FAISS clustering applicable to other domains

### Code Organization

The refactoring transformed a monolithic 1,148-line file into:

- **9 well-structured files** with clear responsibilities
- **1,263 total lines** including documentation and imports
- **22 exported functions** with clean APIs
- **Full backward compatibility** maintained

### Import Structure

```python
# Main entry point
from dedup.cli import main

# Individual modules can be imported as needed
from dedup.hardware import detect_available_gpus, print_gpu_info
from dedup.filesystem import scan_images, split_records_for_gpus
from dedup.models import extract_clip_features_multigpu
from dedup.similarity import build_groups_clip
from dedup.reporting import make_report
from dedup.deletion import delete_duplicates
```

---

## Algorithm

### CLIP Semantic Similarity (SigLIP2)

**Use Case**: Finding semantically similar images even if visually different (e.g., different photos of same object, similar scenes)

**How it works**:

1. **Multi-GPU detection and setup**:

   - Automatically detects all available CUDA GPUs
   - Tests GPU functionality and memory capabilities
   - Selects optimal number of GPUs (or uses all available)
   - Graceful fallback to single GPU/CPU if multi-GPU fails

2. **Workload distribution**:

   - Splits image records into balanced chunks across GPUs
   - Optimizes batch size per GPU based on available memory
   - Uses `num_workers=0` to prevent subprocess initialization issues
   - Handles remainder images intelligently

3. **Parallel feature extraction**:

   - Independent model loading on each GPU with optimal settings
   - Thread-based parallel processing using ThreadPoolExecutor
   - Per-GPU memory management and OOM recovery
   - Flash attention and bfloat16 optimization per GPU
   - Automatic fallback: flash_attention_2 > sdpa > eager
   - Automatic fallback: bfloat16 > float16 > float32

4. **Index building**:

   - Create FAISS index (GPU or CPU)
   - Add all feature vectors to index
   - Index type: Flat Inner Product (exact search)

5. **Similarity search**:

   - For each image, find k nearest neighbors
   - Compute cosine similarity (dot product of normalized vectors)
   - Similarity range: -1 (opposite) to 1 (identical)

6. **Clustering**:
   - Use Union-Find to group images
   - Link pairs with similarity ≥ threshold
   - Form transitive closure of similar images

**Strengths**:

- ✅ Understands semantic content (cats, dogs, buildings, etc.)
- ✅ Robust to extreme transformations (crop, rotate, flip, color)
- ✅ Can find "similar but not duplicate" images
- ✅ Efficient k-NN search with FAISS
- ✅ Multi-GPU support for near-linear speedup
- ✅ GPU acceleration with flash attention and bfloat16
- ✅ Automatic hardware optimization and workload distribution
- ✅ Memory-efficient attention mechanisms
- ✅ Intelligent fallback to single GPU/CPU if multi-GPU fails

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
- `--batch-size`: Base inference batch size (default: 128, auto-adjusted per GPU)
- `--model`: Hugging Face model name
- `--gpus`: Number of GPUs to use (default: all available)
- `--gpu-memory-fraction`: GPU memory allocation per GPU (0.1-1.0, default: 0.9)

**Complexity**: O(n·k·log n) time with FAISS, O(n·d) space (d=embedding dimension)

**Performance Enhancements**:

- **Multi-GPU Processing**: Near-linear speedup with number of GPUs
- **Flash Attention 2**: Up to 2-4x faster inference per GPU, lower memory usage
- **bfloat16**: Up to 2x faster on Ampere GPUs (A100, RTX 30xx/40xx)
- **SDPA**: PyTorch 2.0+ optimized attention (fallback for no flash-attn)
- **Per-GPU Batch Optimization**: Automatic batch size adjustment based on GPU memory

**Example**:

```bash
# Default settings (balanced, auto-optimized, uses all available GPUs)
python main.py /data/photos

# Strict semantic matching
python main.py /data/photos --threshold 0.99 --k 20

# Multi-GPU accelerated processing
python main.py /data/photos --batch-size 256 --gpus 4

# Single GPU with memory limit
python main.py /data/photos --gpu-memory-fraction 0.7
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
      "duplicates": ["/data/photos/vacation/beach_resized.jpg"]
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

| Configuration                                          | Speed      | Memory | Best For                          |
| ------------------------------------------------------ | ---------- | ------ | --------------------------------- |
| **4x Ampere GPUs + Flash Attn + bfloat16 + Multi-GPU** | ⚡⚡⚡⚡⚡ | Low    | Very large datasets (>50k images) |
| **2x Ampere GPUs + Flash Attn + bfloat16 + Multi-GPU** | ⚡⚡⚡⚡   | Low    | Large datasets (>10k images)      |
| **Single Ampere GPU + Flash Attn + bfloat16**          | ⚡⚡⚡     | Low    | Medium datasets (1k-10k images)   |
| **GPU + float16**                                      | ⚡⚡       | Medium | Small datasets (<1k images)       |
| **CPU + float32**                                      | ⚡         | Low    | Tiny datasets (<100 images)       |

### Scaling Characteristics

**Multi-GPU with Flash Attention (2x-4x A100, RTX 40xx)**:

- 1,000 images: ~15-30 seconds
- 10,000 images: ~2.5-5 minutes
- 100,000 images: ~15-30 minutes
- 1,000,000 images: ~2.5-3 hours

**Single GPU with Flash Attention (A100, RTX 40xx)**:

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

1. **Multi-GPU Setup** (for maximum performance):

   - Use `--gpus` to specify number of GPUs (default: all available)
   - Adjust `--gpu-memory-fraction` if encountering OOM errors (try 0.7-0.8)
   - Ensure all GPUs have similar memory for balanced processing
   - Multi-GPU provides near-linear speedup for large datasets

2. **Install Flash Attention** (highly recommended for GPU):

   ```bash
   pip install -U flash-attn --no-build-isolation
   ```

   - 2-4x faster inference per GPU
   - Lower memory usage
   - Automatic detection and fallback

3. **Use Ampere or newer GPUs** for best performance:

   - A100, A6000, RTX 30xx/40xx series
   - Native bfloat16 support
   - Better flash attention performance
   - Automatic per-GPU optimization

4. **Optimize batch size**:

   - Base `--batch-size` is auto-adjusted per GPU
   - Increase if GPU memory allows (try 256 or 512)
   - Reduce to 32 or 64 if you encounter OOM errors
   - Tool automatically optimizes per GPU based on available memory

5. **Adjust k-neighbors** for speed:

   - Decrease `--k` for faster search (try 5 for very large datasets)
   - Increase `--k` for more thorough duplicate detection
   - Default 10 is a good balance

6. **Large datasets** (>100k images):

   - Multi-GPU is highly recommended for best performance
   - Process in subdirectories if possible
   - Use higher thresholds (0.99) to reduce groups
   - Consider using faiss-gpu for faster indexing

7. **Memory constraints**:
   - Total RAM needed: ~4-8 GB + model (~2 GB) + features (~N × 3KB)
   - Use `--gpu-memory-fraction` to limit GPU memory per process
   - Tool automatically handles OOM errors and retries with smaller batches
   - Falls back to single GPU/CPU if all GPUs fail

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

### Example 6: Multi-GPU High-Performance Processing

```bash
# Use multiple GPUs for maximum speed on large datasets
python main.py ~/large_photo_archive \
  --gpus 4 \
  --gpu-memory-fraction 0.8 \
  --batch-size 256 \
  --threshold 0.985 \
  --keep-policy largest \
  --report ~/multi_gpu_dedup.json
```

### Example 7: Aggressive Semantic Matching

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

**Issue**: Multi-GPU processing fails

- **Solution**: Check if all GPUs are working and have sufficient memory. Try reducing `--gpus` or `--gpu-memory-fraction`

**Issue**: "CUDA out of memory" on multi-GPU

- **Solution**: Reduce `--gpu-memory-fraction` to 0.7 or lower, or use fewer GPUs with `--gpus`

**Issue**: Uneven GPU utilization

- **Solution**: Normal if dataset size isn't perfectly divisible by number of GPUs. Tool distributes as evenly as possible

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
- Multiple NVIDIA Ampere or newer GPUs (A100, RTX 40xx)
- CUDA 12.1
- Flash Attention 2
- faiss-gpu installed
- Similar GPU memory for balanced multi-GPU processing

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

### Version 2.2 (Current) - **Modular Architecture Refactoring**

- **Complete code restructuring** from monolithic 1,148-line file to clean modular architecture
- **9 separate modules** with clear separation of concerns and single responsibilities
- **22 exported functions** with clean, well-documented APIs
- **Improved maintainability** through modular design and clear interfaces
- **Enhanced testability** with isolated modules that can be unit tested independently
- **Better extensibility** for adding new algorithms, models, and features
- **Full backward compatibility** maintained - all existing functionality preserved
- **Clean dependency management** between modules
- **Comprehensive documentation** of module responsibilities and functions

### Version 2.1 (Previous)

- **Multi-GPU support** for parallel processing across multiple GPUs
- **Automatic GPU detection** and workload distribution
- **Per-GPU batch size optimization** based on available memory
- **Thread-based parallel processing** with `num_workers=0` to prevent init process issues
- **GPU memory fraction control** for OOM prevention
- **Intelligent fallback system** to single GPU/CPU if multi-GPU fails
- Enhanced error handling and recovery per GPU

### Version 2.0 (Previous)

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
