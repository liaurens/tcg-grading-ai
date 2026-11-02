"""Step 5c: tears (afscheuringen): a piece of the card is missing.

Idea: visible in the outline. The card shape no longer fills its rectangle,
and white background shows where the piece is missing.
OPEN: needs the outline; crop_3 returns only the straight card, not the mask.
"""
import numpy as np

from pipeline.crop_3 import ShowFn, _no_show


def load_model() -> object:
    """Trained tear model from models/."""
    raise NotImplementedError("damage_5.tear.load_model")


def preprocess(card: np.ndarray, show: ShowFn = _no_show) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> image in which missing pieces stand out."""
    raise NotImplementedError("damage_5.tear.preprocess")


def features(image: np.ndarray) -> np.ndarray:
    """Preprocessed image -> numbers for the classifier."""
    raise NotImplementedError("damage_5.tear.features")


def predict(model: object, features: np.ndarray) -> float:
    """Tear score 0-1."""
    raise NotImplementedError("damage_5.tear.predict")
