import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from dedup.cli import parse_args, run_yolo_dedup
from dedup.yolo import export_yolo_dataset, flatten_samples, load_yolo_dataset


class YoloDedupTests(unittest.TestCase):
    def test_cli_exports_pairs_and_removes_phash_duplicates_using_shared_matcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)

            train_image = root / "images" / "train" / "duplicate.jpg"
            Image.new("RGB", (16, 16), "red").save(train_image)
            test_image = root / "images" / "test" / "duplicate.jpg"
            shutil.copy2(train_image, test_image)
            Image.new("RGB", (16, 16), "blue").save(root / "images" / "val" / "unique.jpg")

            (root / "labels" / "train" / "duplicate.txt").write_text("0 0.1 0.1 0.2 0.2\n")
            (root / "labels" / "test" / "duplicate.txt").write_text("0 0.2 0.2 0.3 0.3\n")
            (root / "labels" / "val" / "unique.txt").write_text("1 0.5 0.5 0.2 0.2\n")
            (root / "train.txt").write_text("./images/train/duplicate.jpg\n")
            (root / "val.txt").write_text("./images/val/unique.jpg\n")
            (root / "test.txt").write_text("./images/test/duplicate.jpg\n")
            (root / "classes.txt").write_text("person\n")
            (root / "data.yaml").write_text(
                "path: .\ntrain: ./train.txt\nval: ./val.txt\ntest: ./test.txt\nnc: 2\n"
            )

            args = parse_args(
                ["yolo-dedup", str(root), "--output", str(output), "--split-priority", "test,val,train"]
            )

            def prepare_phash_duplicate_records(records, _args, _cache):
                records[0].sha256 = "train-copy"
                records[2].sha256 = "test-copy"
                records[0].phash = records[2].phash = "0000000000000000"
                return np.empty((0, 0), dtype=np.float32), []

            with patch("dedup.cli._prepare_records", prepare_phash_duplicate_records):
                run_yolo_dedup(args)

            self.assertFalse((output / "images" / "train" / "duplicate.jpg").exists())
            self.assertFalse((output / "labels" / "train" / "duplicate.txt").exists())
            self.assertTrue((output / "images" / "test" / "duplicate.jpg").is_file())
            self.assertTrue((output / "labels" / "test" / "duplicate.txt").is_file())
            self.assertEqual("", (output / "train.txt").read_text())
            self.assertEqual("./images/test/duplicate.jpg\n", (output / "test.txt").read_text())
            self.assertEqual("person\n", (output / "classes.txt").read_text())

            saved_report = json.loads(
                (output / "reports" / "yolo_dedup_report.json").read_text()
            )
            self.assertEqual("sha256_phash_embedding", saved_report["method"])
            self.assertEqual(1, saved_report["duplicate_groups"])
            self.assertEqual(1, saved_report["duplicate_pairs"])
            self.assertEqual(1, saved_report["total_duplicates"])
            self.assertTrue(saved_report["groups"][0]["label_conflict"])
            self.assertEqual("test", saved_report["groups"][0]["kept"][0]["split"])
            self.assertEqual("train", saved_report["groups"][0]["removed"][0]["split"])

    def test_dataset_without_validation_split_is_loaded_and_exported_without_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            output = Path(directory) / "output"
            for split in ("train", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                Image.new("RGB", (16, 16), "red").save(root / "images" / split / "image.jpg")
                (root / "labels" / split / "image.txt").write_text("0 0.1 0.1 0.2 0.2\n")
                (root / f"{split}.txt").write_text(f"./images/{split}/image.jpg\n")
            (root / "data.yaml").write_text(
                "path: .\ntrain: ./train.txt\ntest: ./test.txt\nnc: 1\n"
            )

            data, samples_by_split = load_yolo_dataset(str(root))
            self.assertEqual([], samples_by_split["val"])

            export_yolo_dataset(
                str(root),
                output,
                data,
                flatten_samples(samples_by_split),
                [],
                "copy",
                ["test", "val", "train"],
                {"duplicate_pairs": 0, "review_only_pairs": 0, "review_only": []},
            )

            output_data = (output / "data.yaml").read_text()
            self.assertNotIn("val:", output_data)
            self.assertFalse((output / "val.txt").exists())
            self.assertTrue((output / "train.txt").is_file())
            self.assertTrue((output / "test.txt").is_file())


if __name__ == "__main__":
    unittest.main()
