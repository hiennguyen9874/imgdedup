# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python CLI for local image deduplication. The entry point is `main.py`, which delegates to `dedup.cli`. Core modules live in `dedup/`: scanning in `filesystem.py`, hashing in `hashing.py`, embedding extraction in `models.py`, similarity search in `similarity.py`, duplicate grouping in `matching.py`, reporting in `reporting.py`, and deletion/trash handling in `deletion.py`. Example image inputs are under `images/examples1/` and `images/examples2/`. Generated reports, caches, and `.imgdedup/` trash folders should not be committed.

## Build, Test, and Development Commands

- `uv sync` installs the pinned runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python main.py --help` verifies the CLI starts and lists available options.
- `uv run python main.py images/examples1 --report /tmp/dedup_report.json` runs a safe dry-run scan against sample images.
- `uv run python main.py <folder> --inplace` moves duplicates to `<folder>/.imgdedup/trash/`; use only after reviewing the report.
