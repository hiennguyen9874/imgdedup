import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from dedup.cache import DedupCache
from dedup.exporting import export_records, prepare_output, staged_output, validate_output
from dedup.filesystem import ImgRec
from dedup.quality import QualityThresholds, measure_quality
from dedup.selection import select_representatives


class SelectionTests(unittest.TestCase):
    def test_selection_methods_return_unique_exact_count(self):
        features = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0]],
            dtype=np.float32,
        )
        for method in ("kmeans", "farthest", "hybrid"):
            with self.subTest(method=method):
                selected = select_representatives(features, 3, method, seed=7)
                self.assertEqual(3, len(selected))
                self.assertEqual(3, len(set(selected)))
                self.assertEqual(selected, select_representatives(features, 3, method, seed=7))

    def test_selection_is_invariant_to_per_row_magnitude(self):
        features = np.asarray(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        scaled = features * np.asarray([[100.0], [0.1], [25.0], [3.0], [0.5]])
        for method in ("kmeans", "farthest", "hybrid"):
            with self.subTest(method=method):
                self.assertEqual(
                    select_representatives(features, 3, method, seed=7),
                    select_representatives(scaled, 3, method, seed=7),
                )

    def test_quality_rejects_small_solid_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.jpg"
            Image.new("RGB", (32, 32), "black").save(path)
            metrics = measure_quality(path, QualityThresholds(min_width=64, min_height=64))
            self.assertEqual("too_small", metrics.rejection_reason)
            self.assertEqual((32, 32), (metrics.width, metrics.height))

    def test_export_preserves_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            source = root / "nested" / "image.jpg"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), "red").save(source)
            output = Path(directory) / "output"
            stat = source.stat()
            record = ImgRec(str(source.resolve()), stat.st_size, stat.st_mtime)

            with staged_output(str(root), str(output)) as staging:
                result = export_records([record], str(root), str(staging), "copy")
                self.assertEqual("exported", result[0]["status"])

            self.assertTrue((output / "images" / "nested" / "image.jpg").is_file())

    def test_output_requires_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            with self.assertRaises(FileExistsError):
                validate_output(str(root), str(output))

    def test_prepare_output_never_deletes_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep")

            with self.assertRaises(FileExistsError):
                prepare_output(str(root), str(output), force=True)

            self.assertEqual("keep", marker.read_text())

    def test_output_rejects_symlink_even_with_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            target = Path(directory) / "target"
            output = Path(directory) / "output-link"
            root.mkdir()
            target.mkdir()
            output.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output(str(root), str(output), force=True)
            self.assertTrue(target.is_dir())

    def test_output_rejects_overlap_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            root.mkdir()
            with self.assertRaises(ValueError):
                validate_output(str(root), str(root / "export"), force=True)
            with self.assertRaises(ValueError):
                validate_output(str(root), directory, force=True)
            self.assertTrue(root.is_dir())

    def test_staged_output_preserves_existing_output_until_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            root.mkdir()
            output.mkdir()
            (output / "old.txt").write_text("old")

            with self.assertRaises(RuntimeError):
                with staged_output(str(root), str(output), force=True) as staging:
                    (staging / "new.txt").write_text("new")
                    raise RuntimeError("processing failed")

            self.assertEqual("old", (output / "old.txt").read_text())
            self.assertFalse((output / "new.txt").exists())

            with staged_output(str(root), str(output), force=True) as staging:
                (staging / "new.txt").write_text("new")

            self.assertFalse((output / "old.txt").exists())
            self.assertEqual("new", (output / "new.txt").read_text())

    def test_embedding_subset_save_invalidates_stale_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DedupCache(directory)
            try:
                records = [
                    ImgRec(str(Path(directory) / "a.jpg"), 1, 1.0),
                    ImgRec(str(Path(directory) / "b.jpg"), 1, 1.0),
                ]
                cache.save_embeddings(
                    records,
                    np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    [0, 1],
                    "model",
                )
                cache.save_embeddings(
                    records,
                    np.asarray([[0.5, 0.5]], dtype=np.float32),
                    [1],
                    "model",
                )

                self.assertIsNone(cache.get_embedding(records[0], "model"))
                np.testing.assert_array_equal(
                    np.asarray([0.5, 0.5], dtype=np.float32),
                    cache.get_embedding(records[1], "model"),
                )
            finally:
                cache.close()

    def test_select_workflow_deduplicates_and_exports_exact_count(self):
        from dedup.cli import run_select

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            output = Path(directory) / "output"
            root.mkdir()
            for name, color in (("a.jpg", "red"), ("b.jpg", "red"), ("c.jpg", "blue")):
                Image.new("RGB", (32, 32), color).save(root / name)

            args = Namespace(
                folder=str(root), output=str(output), force=False, num=2,
                selection_method="hybrid", copy_mode="copy", seed=11,
                make_preview=True, preview_columns=2, preview_size=64,
                reject_low_quality=False, min_width=1, min_height=1,
                min_blur_score=0.0, min_brightness=0.0, max_brightness=255.0,
                cache_root=str(root), model="test-model", metadata_workers=1,
                batch_size=2, gpus=None, gpu_memory_fraction=0.9,
                loader_workers=0, cosine_auto=0.97, cosine_verify=0.9,
                cosine_review=0.85, phash_auto_distance=4,
                phash_verify_distance=8, k=5, grouping="connected",
                agglomerative_linkage="complete",
                agglomerative_cosine_threshold=None, keep_policy="best-quality",
            )
            features = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            with patch("dedup.cli._prepare_records", return_value=(features, [0, 1, 2])), patch(
                "dedup.cli.find_duplicate_pairs", return_value=([], [], [[0, 1]])
            ):
                run_select(args)

            report = json.loads((output / "reports" / "selection_report.json").read_text())
            self.assertEqual(2, report["funnel"]["selected"])
            self.assertEqual(1, report["funnel"]["duplicates_removed"])
            self.assertNotIn(".staging-", json.dumps(report))
            self.assertEqual(2, len(list((output / "images").glob("*.jpg"))))
            self.assertTrue((output / "previews" / "selected.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
