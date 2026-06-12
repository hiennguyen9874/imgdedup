# imgdedup

Local CLI tool for finding duplicate images using exact hash, perceptual hash, and image embeddings.

The tool is accuracy-focused and safe by default:

- no API server
- no database server required
- dry-run by default
- review report before deleting
- `--inplace` moves duplicates to trash, not hard delete
- hard delete requires explicit `--hard-delete --yes`

## Duplicate rules

A pair is treated as an auto duplicate when any rule matches:

| Rule | Condition |
| --- | --- |
| Same SHA-256 | `sha256` are identical |
| pHash auto | Hamming distance ≤ `--phash-auto-distance` (default 4) |
| Cosine auto | Cosine similarity ≥ `--cosine-auto` (default 0.97) |
| Cosine + pHash verify | Cosine ≥ `--cosine-verify` (default 0.90) **and** pHash distance ≤ `--phash-verify-distance` (default 8) |

Review-only rule (reported but never auto-deleted):

```text
--cosine-review (0.85) ≤ cosine < --cosine-verify (0.90)
  => report only, not auto delete
```

Review-only pairs are written to the JSON report under `review_only` and are never passed to deletion.

## Install

This project uses `uv`.

```bash
uv sync
```

If dependencies are already installed, you can run directly with:

```bash
uv run python main.py --help
```

## Basic usage

Dry run scan:

```bash
uv run python main.py ./photos
```

This will:

1. scan image files
2. compute `sha256` and pHash
3. extract image embeddings using a vision transformer model
4. search candidates with FAISS
5. apply duplicate rules
6. write a JSON report

Default report path:

```text
./photos/dedup_report.json
```

No files are deleted in dry-run mode.

## Write report to custom path

```bash
uv run python main.py ./photos --report ./dedup_report.json
```

## Move duplicates to trash

```bash
uv run python main.py ./photos --inplace
```

This moves duplicate files to:

```text
./photos/.imgdedup/trash/<run_id>/
```

A restore manifest is written to:

```text
./photos/.imgdedup/trash/<run_id>/restore_manifest.json
```

## Hard delete

Only use this after reviewing the report.

```bash
uv run python main.py ./photos --inplace --hard-delete --yes
```

Without `--yes`, hard delete is rejected.

## Useful options

```bash
uv run python main.py ./photos \
  --cosine-auto 0.97 \
  --cosine-verify 0.90 \
  --cosine-review 0.85 \
  --phash-auto-distance 4 \
  --phash-verify-distance 8 \
  --k 50
```

## Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--cosine-auto` | `0.97` | Auto duplicate if cosine is at least this value |
| `--cosine-verify` | `0.90` | Duplicate if cosine is at least this value and pHash distance is low enough |
| `--cosine-review` | `0.85` | Lower bound for report-only pairs |
| `--phash-auto-distance` | `4` | Auto duplicate if pHash distance is at most this value |
| `--phash-verify-distance` | `8` | Max pHash distance for cosine + pHash verified duplicates |
| `--k` | `50` | Number of FAISS nearest neighbors to search |
| `--batch-size` | `128` | Embedding inference batch size |
| `--model` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Hugging Face model for embedding extraction |
| `--gpus` | all available | Number of GPUs to use for parallel processing |
| `--gpu-memory-fraction` | `0.9` | GPU memory fraction per device (0.1–1.0) |
| `--keep-policy` | `highest-resolution` | Which file to keep in each duplicate group |
| `--report` | `<folder>/dedup_report.json` | Output path for the JSON report |

### Keep policies

```text
lexi                — lexicographically smallest filename
smallest            — smallest file size
largest             — largest file size
highest-resolution  — most pixels (width × height); tie-break by file size
newest              — most recent modification time
oldest              — oldest modification time
```

Example:

```bash
uv run python main.py ./photos --keep-policy largest
```

## Report format

Example report with one duplicate group:

```json
{
  "generated_at": "2026-06-12 10:30:00",
  "total_images": 100,
  "duplicate_groups": 2,
  "total_duplicates": 3,
  "review_only_pairs": 1,
  "keep_policy": "highest-resolution",
  "groups": [
    {
      "keep": "/path/photo.jpg",
      "duplicates": [
        {
          "path": "/path/photo-copy.jpg",
          "reason": "same_sha256",
          "cosine": null,
          "phash_distance": 0,
          "same_sha256": true,
          "confidence": "exact"
        }
      ]
    }
  ],
  "review_only": [
    {
      "a": "/path/a.jpg",
      "b": "/path/b.jpg",
      "cosine": 0.87,
      "phash_distance": 12,
      "same_sha256": false,
      "decision": "review",
      "reason": "0.85<=cosine<0.90",
      "confidence": "review"
    }
  ]
}
```

### Report fields

| Field | Description |
| --- | --- |
| `generated_at` | Timestamp of report generation |
| `total_images` | Number of images scanned |
| `duplicate_groups` | Number of groups with duplicates |
| `total_duplicates` | Individual duplicate files to remove |
| `review_only_pairs` | Count of pairs needing manual review |
| `keep_policy` | Policy used to pick the kept file |
| `groups` | Array of duplicate groups, each with a `keep` file and `duplicates` |
| `review_only` | Array of pairs flagged for review (never auto-deleted) |

## Supported image types

The scanner currently includes:

```text
.jpg  .jpeg  .png  .bmp  .webp  .tif  .tiff
```

It skips common internal folders such as:

```text
.git  .imgdedup  __pycache__  node_modules
```

## Caching

Metadata and embeddings are cached inside a `.imgdedup/` directory next to the scanned folder:

| File | Purpose |
| --- | --- |
| `db.sqlite` | SQLite database storing file paths, hashes, dimensions, and embedding offsets |
| `embeddings.npy` | Memory-mapped array of all computed embeddings |
| `faiss.index` | Serialized FAISS index for nearest-neighbor search |
| `trash/<run_id>/` | Moved duplicate files and `restore_manifest.json` |

On subsequent runs, only new or changed files are processed — cached results are reused.

## Architecture notes

- **Default model**: `facebook/dinov3-vitb16-pretrain-lvd1689m` (vision-only transformer). Any Hugging Face vision embedding model can be used via `--model`.
- **FAISS** is used for nearest-neighbor search (GPU-accelerated when available).
- **pHash search** uses an in-memory BK-tree for exact Hamming-distance queries.
- **Embeddings** are L2-normalized, so FAISS inner product equals cosine similarity.
- **GPU support**: multi-GPU parallel extraction via `ThreadPoolExecutor`, with automatic fallback to single GPU or CPU.
- **Flash Attention 2** is used when available, falling back to PyTorch SDPA or eager attention.
- **`bfloat16`** is used on Ampere (compute capability ≥ 8.0) GPUs; `float16` on older GPUs; `float32` on CPU.
