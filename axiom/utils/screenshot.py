"""Screenshot capture, storage, and encoding utilities.

Screenshots are a core part of the observation pipeline for vision-capable
agents. These utilities handle the base64 encode/decode and file I/O
so other modules don't need to deal with raw bytes.
"""

from __future__ import annotations

import base64
from pathlib import Path


def save_screenshot(base64_data: str, path: Path) -> Path:
    """Decode a base64 PNG string and save to disk.

    Args:
        base64_data: Base64-encoded PNG image.
        path: Destination file path.

    Returns:
        The path where the file was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(base64_data))
    return path


def encode_screenshot(path: Path) -> str:
    """Read a PNG file and return its base64 encoding.

    Args:
        path: Path to a PNG file.

    Returns:
        Base64-encoded string of the file contents.
    """
    return base64.b64encode(path.read_bytes()).decode()
