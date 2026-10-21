"""Read photos from a folder, save step images."""
from pathlib import Path

import numpy as np

from pipeline import config


def list_images(folder: Path) -> list[Path]:
    """Sorted photo paths in folder."""
    # TODO: filter on IMAGE_EXTENSIONS, error if folder missing or empty
    raise NotImplementedError("io_utils.list_images")


def load_image(path: Path) -> np.ndarray:
    """Photo as BGR array."""
    # TODO: cv2.imread returns None on failure -> raise
    raise NotImplementedError("io_utils.load_image")


def save_debug_image(image: np.ndarray, card_name: str, step_name: str) -> None:
    """Save to output/debug/<card_name>/<step_name>.png."""
    raise NotImplementedError("io_utils.save_debug_image")
