"""Safe, non-destructive export of selected images."""

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from .filesystem import ImgRec


def validate_output(input_root: str, output_root: str, force: bool = False) -> Path:
    """Validate an output path without changing its current contents."""
    source = Path(input_root).resolve()
    output = Path(output_root).expanduser().absolute()
    if output.is_symlink():
        raise ValueError(f"refusing to use symlink as output root: {output}")
    resolved_output = output.resolve()
    if (
        resolved_output == source
        or resolved_output in source.parents
        or source in resolved_output.parents
    ):
        raise ValueError("input and output directories must not overlap")

    if output.exists():
        if not force:
            raise FileExistsError(f"output already exists: {output}; use --force to replace it")
        if not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
    return output


def prepare_output(input_root: str, output_root: str, force: bool = False) -> Path:
    """Create a new output directory; use staged_output when replacing one."""
    output = validate_output(input_root, output_root, force)
    if output.exists():
        raise FileExistsError(
            f"refusing to replace output in place: {output}; use staged_output"
        )
    output.mkdir(parents=True)
    return output


@contextmanager
def staged_output(
    input_root: str, output_root: str, force: bool = False
) -> Iterator[Path]:
    """Build output in staging and replace the destination only on success."""
    output = validate_output(input_root, output_root, force)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"

    try:
        yield staging

        # Revalidate in case the destination changed while processing.
        validate_output(input_root, output_root, force)
        had_existing_output = output.exists()
        if had_existing_output:
            output.rename(backup)
        try:
            staging.rename(output)
        except BaseException:
            if had_existing_output and backup.exists() and not output.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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
