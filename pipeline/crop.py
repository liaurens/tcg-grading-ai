"""Step 3: find card, cut out, straighten to portrait 63:88.

OPEN: mask method (depends on background).
"""
import numpy as np


class CardNotFoundError(Exception):
    """No card found in photo."""


def find_card_mask(image: np.ndarray) -> np.ndarray:
    """Mask: card white, background black."""
    raise NotImplementedError("crop.find_card_mask")


def find_card_corners(mask: np.ndarray) -> np.ndarray:
    """4 corners (TL, TR, BR, BL) as float32 (4, 2)."""
    raise NotImplementedError("crop.find_card_corners")


def warp_card(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Straighten card to portrait 63:88."""
    raise NotImplementedError("crop.warp_card")


def crop_card(image: np.ndarray) -> np.ndarray:
    """Photo in, straight card out."""
    mask = find_card_mask(image)
    corners = find_card_corners(mask)
    return warp_card(image, corners)
