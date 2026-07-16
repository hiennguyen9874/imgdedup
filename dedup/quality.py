"""Image quality measurement used for representative selection."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from .filesystem import ImgRec


@dataclass(frozen=True)
class QualityThresholds:
    min_width: int = 224
    min_height: int = 224
    min_blur_score: float = 20.0
    min_brightness: float = 15.0
    max_brightness: float = 240.0


@dataclass(frozen=True)
class QualityMetrics:
    width: int
    height: int
    blur_score: float
    brightness: float
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def measure_quality(path: str, thresholds: Optional[QualityThresholds] = None) -> QualityMetrics:
    """Measure dimensions, brightness, and edge variance without extra dependencies."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            gray = image.convert("L")
            brightness = float(np.asarray(gray, dtype=np.float32).mean())
            edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
            if width > 2 and height > 2:
                edges = edges[1:-1, 1:-1]
            blur_score = float(edges.var()) if edges.size else 0.0
    except (OSError, UnidentifiedImageError):
        return QualityMetrics(0, 0, 0.0, 0.0, "unreadable")

    reason = None
    if thresholds is not None:
        if width < thresholds.min_width or height < thresholds.min_height:
            reason = "too_small"
        elif blur_score < thresholds.min_blur_score:
            reason = "blurry"
        elif brightness < thresholds.min_brightness:
            reason = "too_dark"
        elif brightness > thresholds.max_brightness:
            reason = "too_bright"

    return QualityMetrics(width, height, blur_score, brightness, reason)


def measure_records(
    records: Iterable[ImgRec], thresholds: Optional[QualityThresholds] = None
) -> Dict[str, QualityMetrics]:
    """Measure records and key results by absolute image path."""
    return {record.path: measure_quality(record.path, thresholds) for record in records}


def quality_sort_key(record: ImgRec, metrics: Optional[QualityMetrics]):
    """Return a deterministic key where larger means a better representative."""
    if metrics is None:
        return (0, 0.0, -255.0, 0, record.size, record.path)
    pixels = metrics.width * metrics.height
    return (
        0 if metrics.rejection_reason else 1,
        metrics.blur_score,
        -abs(metrics.brightness - 127.5),
        pixels,
        record.size,
        record.path,
    )
