"""Safe, non-destructive export of selected images."""

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from .filesystem import ImgRec


def prepare_output(input_root: str, output_root: str, force: bool = False) -> Path:
    """Validate and prepare an output directory without risking the input tree."""
    source = Path(input_root).resolve()
    output = Path(output_root).expanduser().absolute()
    if output.is_symlink():
        raise ValueError(f"refusing to use symlink as output root: {output}")
    resolved_output = output.resolve()
    if resolved_output == source or resolved_output in source.parents:
        raise ValueError("output must not be the input directory or one of its parents")

    if output.exists():
        if not force:
            raise FileExistsError(f"output already exists: {output}; use --force to replace it")
        if not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def _transfer(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source)
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def export_records(
    records: Iterable[ImgRec], input_root: str, output_root: str, mode: str
) -> List[Dict]:
    """Export records while preserving paths relative to the input root."""
    source_root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    results = []
    for record in records:
        source = Path(record.path).resolve()
        try:
            relative = source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"image is outside input root: {source}") from exc
        destination = output / "images" / relative
        try:
            _transfer(source, destination, mode)
            results.append(
                {"source": str(source), "output": str(destination), "status": "exported"}
            )
        except OSError as exc:
            results.append(
                {
                    "source": str(source),
                    "output": str(destination),
                    "status": "export_failed",
                    "error": str(exc),
                }
            )
    return results
