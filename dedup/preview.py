"""Contact-sheet generation for selected images."""

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageOps

from .filesystem import ImgRec


def make_contact_sheet(
    records: Iterable[ImgRec],
    output_path: str,
    columns: int = 8,
    thumbnail_size: int = 192,
) -> int:
    """Write a labeled contact sheet and return the number of rendered images."""
    records = list(records)
    if not records:
        return 0
    if columns < 1 or thumbnail_size < 32:
        raise ValueError("preview columns must be positive and thumbnail size at least 32")

    label_height = 24
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * thumbnail_size, rows * (thumbnail_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    rendered = 0
    for position, record in enumerate(records):
        try:
            with Image.open(record.path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumbnail_size, thumbnail_size))
                x = (position % columns) * thumbnail_size
                y = (position // columns) * (thumbnail_size + label_height)
                offset_x = x + (thumbnail_size - image.width) // 2
                offset_y = y + (thumbnail_size - image.height) // 2
                sheet.paste(image, (offset_x, offset_y))
                label = f"{position + 1}. {Path(record.path).name}"
                draw.text((x + 3, y + thumbnail_size + 4), label[:28], fill="black")
                rendered += 1
        except OSError:
            continue

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)
    return rendered
