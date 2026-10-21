"""Steps 4-6: grayscale, Canny, high-pass. OPEN: parameters."""
import numpy as np


def to_gray(card: np.ndarray) -> np.ndarray:
    """Step 4: BGR -> gray."""
    raise NotImplementedError("preprocess.to_gray")


def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Step 5: Canny edges."""
    raise NotImplementedError("preprocess.detect_edges")


def high_pass(gray: np.ndarray) -> np.ndarray:
    """Step 6: fine detail only (scratches, creases)."""
    raise NotImplementedError("preprocess.high_pass")
