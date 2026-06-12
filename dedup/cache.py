"""
SQLite-backed cache for image metadata and embeddings.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .filesystem import ImgRec


class DedupCache:
    """Cache metadata and per-model embedding offsets under .imgdedup."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.cache_dir = self.root / ".imgdedup"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "db.sqlite"
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.faiss_index_path = self.cache_dir / "faiss.index"
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()
        self._metadata_by_key = self._load_metadata_index()
        self._existing_embeddings = self._load_existing_embeddings()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                sha256 TEXT,
                phash TEXT,
                width INTEGER,
                height INTEGER,
                embedding_offset INTEGER,
                model_version TEXT
            )
            """
        )
        self.conn.commit()

    def _load_existing_embeddings(self) -> Optional[np.ndarray]:
        if not self.embeddings_path.exists():
            return None
        try:
            return np.load(self.embeddings_path, mmap_mode="r")
        except (OSError, ValueError) as exc:
            print(f"Warning: Could not load cached embeddings: {exc}")
            return None

    def _load_metadata_index(self) -> Dict[Tuple[str, int, float], Dict]:
        rows = self.conn.execute(
            """
            SELECT path, size, mtime, sha256, phash, width, height,
                   embedding_offset, model_version
            FROM images
            """
        ).fetchall()
        return {
            (row[0], row[1], row[2]): {
                "sha256": row[3],
                "phash": row[4],
                "width": row[5],
                "height": row[6],
                "embedding_offset": row[7],
                "model_version": row[8],
            }
            for row in rows
        }

    def _metadata_key(self, record: ImgRec) -> Tuple[str, int, float]:
        return (record.path, record.size, record.mtime)

    def get_metadata(self, record: ImgRec) -> Optional[Dict]:
        return self._metadata_by_key.get(self._metadata_key(record))

    def apply_cached_metadata(self, records: List[ImgRec]) -> int:
        hits = 0
        for record in records:
            cached = self.get_metadata(record)
            if cached is None:
                continue
            record.sha256 = cached["sha256"]
            record.phash = cached["phash"]
            record.width = cached["width"]
            record.height = cached["height"]
            hits += 1
        return hits

    def get_embedding(self, record: ImgRec, model_name: str) -> Optional[np.ndarray]:
        cached = self.get_metadata(record)
        if cached is None or cached["model_version"] != model_name:
            return None
        offset = cached["embedding_offset"]
        if offset is None or self._existing_embeddings is None:
            return None
        if offset < 0 or offset >= len(self._existing_embeddings):
            return None
        return np.asarray(self._existing_embeddings[offset], dtype=np.float32)

    def update_metadata(self, record: ImgRec) -> None:
        self.conn.execute(
            """
            INSERT INTO images(path, size, mtime, sha256, phash, width, height)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size,
                mtime = excluded.mtime,
                sha256 = excluded.sha256,
                phash = excluded.phash,
                width = excluded.width,
                height = excluded.height,
                embedding_offset = NULL,
                model_version = NULL
            """,
            (
                record.path,
                record.size,
                record.mtime,
                record.sha256,
                record.phash,
                record.width,
                record.height,
            ),
        )
        self._metadata_by_key[self._metadata_key(record)] = {
            "sha256": record.sha256,
            "phash": record.phash,
            "width": record.width,
            "height": record.height,
            "embedding_offset": None,
            "model_version": None,
        }

    def save_embeddings(
        self,
        records: List[ImgRec],
        features: np.ndarray,
        valid_indices: List[int],
        model_name: str,
    ) -> None:
        if features.size == 0:
            return

        np.save(self.embeddings_path, features.astype(np.float32))
        for offset, record_idx in enumerate(valid_indices):
            record = records[record_idx]
            self.conn.execute(
                """
                INSERT INTO images(
                    path, size, mtime, sha256, phash, width, height,
                    embedding_offset, model_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    size = excluded.size,
                    mtime = excluded.mtime,
                    sha256 = excluded.sha256,
                    phash = excluded.phash,
                    width = excluded.width,
                    height = excluded.height,
                    embedding_offset = excluded.embedding_offset,
                    model_version = excluded.model_version
                """,
                (
                    record.path,
                    record.size,
                    record.mtime,
                    record.sha256,
                    record.phash,
                    record.width,
                    record.height,
                    offset,
                    model_name,
                ),
            )
            self._metadata_by_key[self._metadata_key(record)] = {
                "sha256": record.sha256,
                "phash": record.phash,
                "width": record.width,
                "height": record.height,
                "embedding_offset": offset,
                "model_version": model_name,
            }
        self.conn.commit()
        self._existing_embeddings = self._load_existing_embeddings()


def build_feature_matrix(
    cache: DedupCache,
    records: List[ImgRec],
    model_name: str,
    extracted_features: np.ndarray,
    extracted_indices: List[int],
) -> Tuple[np.ndarray, List[int], int]:
    """Merge cached and newly extracted embeddings into record-order arrays."""
    by_index = {
        record_idx: extracted_features[pos]
        for pos, record_idx in enumerate(extracted_indices)
    }
    features = []
    valid_indices = []
    cache_hits = 0

    for record_idx, record in enumerate(records):
        feature = by_index.get(record_idx)
        if feature is None:
            feature = cache.get_embedding(record, model_name)
            if feature is not None:
                cache_hits += 1
        if feature is None:
            continue
        features.append(np.asarray(feature, dtype=np.float32))
        valid_indices.append(record_idx)

    if not features:
        return np.empty((0, 0), dtype=np.float32), [], cache_hits

    return np.stack(features).astype(np.float32), valid_indices, cache_hits
