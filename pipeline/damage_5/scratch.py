"""Step 5a: scratches (krassen): short, thin lines on the surface.

Idea: high-pass filter keeps only fine detail, so thin scratches stand out
and the large areas of the card art fall away. OPEN: filter + parameters, features, classifier.
"""
import numpy as np

from pipeline.io_utils_1 import ShowFn, no_show


def load_model() -> object:
    """Trained scratch model from models/."""
    raise NotImplementedError("damage_5.scratch.load_model")


def save_model(model: object) -> None:
    """Save the trained model to models/ (same format load_model reads)."""
    raise NotImplementedError("damage_5.scratch.save_model")


def preprocess(card: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> image in which scratches stand out."""
    raise NotImplementedError("damage_5.scratch.preprocess")


def features(image: np.ndarray) -> np.ndarray:
    """Preprocessed image -> numbers for the classifier."""
    raise NotImplementedError("damage_5.scratch.features")


def predict(model: object, features: np.ndarray) -> float:
    """Scratch score 0-1."""
    raise NotImplementedError("damage_5.scratch.predict")


def train(features: list[np.ndarray], labels: list[int]) -> object:
    """Fit a model on features of the training photos. Label 1 = this damage, 0 = not."""
    raise NotImplementedError("damage_5.scratch.train")
