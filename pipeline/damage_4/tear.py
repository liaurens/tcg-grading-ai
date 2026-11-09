"""Step 4b: tears (afscheuringen): a piece of the card is missing.

Idea: visible in the outline. The card shape no longer fills its rectangle,
and white background shows where the piece is missing.
OPEN: needs the outline; crop_2 returns only the straight card, not the mask.
"""
import numpy as np

from pipeline.io_utils_1 import ShowFn, no_show


def load_model() -> object:
    """Trained tear model from models/."""
    raise NotImplementedError("damage_4.tear.load_model")


def save_model(model: object) -> None:
    """Save the trained model to models/ (same format load_model reads)."""
    raise NotImplementedError("damage_4.tear.save_model")


def preprocess(card: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> image in which missing pieces stand out."""
    raise NotImplementedError("damage_4.tear.preprocess")


def features(image: np.ndarray) -> np.ndarray:
    """Preprocessed image -> numbers for the classifier."""
    raise NotImplementedError("damage_4.tear.features")


def predict(model: object, features: np.ndarray) -> float:
    """Tear score 0-1."""
    raise NotImplementedError("damage_4.tear.predict")


def train(features: list[np.ndarray], labels: list[int]) -> object:
    """Fit a model on features of the training photos. Label 1 = this damage, 0 = not."""
    raise NotImplementedError("damage_4.tear.train")
