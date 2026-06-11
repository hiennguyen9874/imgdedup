"""
Image hashing helpers for exact and perceptual duplicate detection.
"""

import hashlib
from typing import Optional

from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import imagehash
except ImportError:  # pragma: no cover - runtime dependency guard
    imagehash = None


def compute_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 for a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_image_metadata(path: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Compute perceptual hash and image dimensions.

    Returns (phash, width, height). If the image cannot be decoded, returns
    (None, None, None) so the rest of the pipeline can still use SHA-256.
    """
    if imagehash is None:
        raise ImportError("imagehash is required for pHash. Install with: pip install imagehash")

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            phash = str(imagehash.phash(image))
            return phash, width, height
    except (UnidentifiedImageError, OSError) as exc:
        print(f"Warning: Could not compute pHash for {path}: {exc}")
        return None, None, None


def phash_distance(left: Optional[str], right: Optional[str]) -> Optional[int]:
    """Return Hamming distance between two hexadecimal perceptual hashes."""
    if not left or not right:
        return None

    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return None
