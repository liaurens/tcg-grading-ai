"""Step 5b: creases (vouwrandjes): a fold that was folded back, but left a ridge.

Idea: the ridge catches the oblique light, so it shows as a long, straight, brighter
line across the card (longer and straighter than a scratch). OPEN: method, features, classifier.
"""
import numpy as np

from pipeline.io_utils_1 import ShowFn, no_show


def load_model() -> object:
    """Trained crease model from models/."""
    raise NotImplementedError("damage_5.crease.load_model")


def save_model(model: object) -> None:
    """Save the trained model to models/ (same format load_model reads)."""
    raise NotImplementedError("damage_5.crease.save_model")


def preprocess(card: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> image in which creases stand out."""
    raise NotImplementedError("damage_5.crease.preprocess")


def features(image: np.ndarray) -> np.ndarray:
    """Preprocessed image -> numbers for the classifier."""
    raise NotImplementedError("damage_5.crease.features")


def predict(model: object, features: np.ndarray) -> float:
    """Crease score 0-1."""
    raise NotImplementedError("damage_5.crease.predict")


def train(features: list[np.ndarray], labels: list[int]) -> object:
    """Fit a model on features of the training photos. Label 1 = this damage, 0 = not."""
    raise NotImplementedError("damage_5.crease.train")
