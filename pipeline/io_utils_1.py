"""Read photos from a folder, save step images."""
from pathlib import Path

import cv2
import numpy as np

from pipeline import config


def list_images(folder: Path) -> list[Path]:
    """Sorted photo paths in folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Map bestaat niet: {folder}")
    # lower(): iPhone sometimes writes .JPEG
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"Geen foto's in map: {folder}")
    return paths


def load_image(path: Path) -> np.ndarray:
    """Photo as BGR array."""
    image = cv2.imread(str(path))
    # imread does not raise on a bad file, it returns None
    if image is None:
        raise OSError(f"Kan foto niet lezen: {path}")
    return image


def save_debug_image(image: np.ndarray, card_name: str, step_name: str) -> None:
    """Save to output/debug/<card_name>/<step_name>.png."""
    folder = config.DEBUG_DIR / card_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{step_name}.png"  # png: lossless, so no jpeg artefacts that look like scratches
    # imwrite also fails silently, it returns False
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Kan debugplaatje niet opslaan: {path}")
