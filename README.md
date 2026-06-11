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

```text
same sha256
  => duplicate

pHash distance <= 4
  => duplicate

cosine >= 0.97
  => duplicate

cosine >= 0.90 and pHash distance <= 8
  => duplicate
```

Review-only rule:

```text
0.85 <= cosine < 0.90
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
3. extract image embeddings
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

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--cosine-auto` | `0.97` | Auto duplicate if cosine is at least this value |
| `--cosine-verify` | `0.90` | Duplicate if cosine is at least this value and pHash distance is low enough |
| `--cosine-review` | `0.85` | Lower bound for report-only pairs |
| `--phash-auto-distance` | `4` | Auto duplicate if pHash distance is at most this value |
| `--phash-verify-distance` | `8` | Max pHash distance for cosine + pHash verified duplicates |
| `--k` | `50` | Number of FAISS nearest neighbors to search |
| `--batch-size` | `128` | Embedding inference batch size |
| `--gpus` | all available | Number of GPUs to use |
| `--keep-policy` | `lexi` | Which image to keep in each duplicate group |

Keep policies:

```text
lexi
smallest
largest
newest
oldest
```

Example:

```bash
uv run python main.py ./photos --keep-policy largest
```

## Report format

The JSON report contains confirmed duplicate groups:

```json
{
  "groups": [
    {
      "keep": "/path/photo.jpg",
      "duplicates": [
        {
          "path": "/path/photo-copy.jpg",
          "reason": "same_sha256",
          "cosine": null,
          "phash_distance": 0,
          "same_sha256": true
        }
      ]
    }
  ],
  "review_only": []
}
```

Review-only pairs look like:

```json
{
  "a": "/path/a.jpg",
  "b": "/path/b.jpg",
  "cosine": 0.87,
  "phash_distance": 12,
  "decision": "review",
  "reason": "0.85<=cosine<0.90"
}
```

## Supported image types

The scanner currently includes:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

It skips common internal folders such as:

```text
.git
.imgdedup
__pycache__
node_modules
```

## Notes

- The default embedding model is `google/siglip2-base-patch16-naflex`.
- FAISS is used for nearest-neighbor search.
- Embeddings are L2-normalized, so FAISS inner product is cosine similarity.
- The current implementation does not yet persist an embedding cache; repeated runs recompute embeddings.
