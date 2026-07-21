import os
import tempfile
import unittest
from pathlib import Path

from dedup.cli import parse_args


class ConfigTests(unittest.TestCase):
    def _config(self, content):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "config.yaml"
        path.write_text(content, encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return str(path)

    def test_dedup_can_run_with_only_config(self):
        path = self._config(
            """
command: dedup
folders: [./photos]
batch_size: 64
cross_folder_only: true
"""
        )

        args = parse_args(["--config", path])

        self.assertEqual(["./photos"], args.folders)
        self.assertEqual(64, args.batch_size)
        self.assertTrue(args.cross_folder_only)

    def test_config_overrides_cli(self):
        path = self._config(
            """
command: dedup
folders: [./from-config]
batch_size: 32
no_report: false
"""
        )

        args = parse_args(
            ["./from-cli", "--batch-size", "128", "--no-report", "--config", path]
        )

        self.assertEqual(["./from-config"], args.folders)
        self.assertEqual(32, args.batch_size)
        self.assertFalse(args.no_report)

    def test_yolo_dedup_can_run_with_only_config(self):
        path = self._config(
            """
command: yolo-dedup
dataset: ~/dataset
output: ~/dataset-deduped
copy_mode: hardlink
split_priority: test,val,train
cosine_auto: 0.98
batch_size: 64
grouping: agglomerative
agglomerative_linkage: average
"""
        )

        args = parse_args(["--config", path])

        self.assertTrue(args.yolo_dedup)
        self.assertEqual(os.path.expanduser("~/dataset"), args.dataset)
        self.assertEqual(os.path.expanduser("~/dataset-deduped"), args.output)
        self.assertEqual("hardlink", args.copy_mode)
        self.assertEqual(0.98, args.cosine_auto)
        self.assertEqual(64, args.batch_size)
        self.assertEqual("agglomerative", args.grouping)
        self.assertEqual("average", args.agglomerative_linkage)

    def test_select_required_values_can_come_from_config(self):
        path = self._config(
            """
command: select
folder: ./photos
output: ./selected
num: 10
"""
        )

        args = parse_args(["--config", path])

        self.assertTrue(args.select)
        self.assertEqual("./photos", args.folder)
        self.assertEqual("./selected", args.output)
        self.assertEqual(10, args.num)

    def test_yes_is_an_option_name_not_a_yaml_boolean(self):
        path = self._config(
            """
command: dedup
folders: [./photos]
yes: true
"""
        )

        args = parse_args(["--config", path])

        self.assertTrue(args.yes)

    def test_unknown_option_is_rejected(self):
        path = self._config(
            """
command: dedup
folders: [./photos]
not_an_option: true
"""
        )

        with self.assertRaises(SystemExit):
            parse_args(["--config", path])


if __name__ == "__main__":
    unittest.main()
