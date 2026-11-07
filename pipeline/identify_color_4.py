"""Step 4: which card is it? Game (MTG/Pokémon) and side (front/back).

Two models, each with: load/save model, features, predict, train.
Both work on the color card from step 3. OPEN: features + classifier type.
"""
import cv2
import numpy as np
import joblib

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_DIR = Path("models")
COLOR_MODEL_PATH = MODEL_DIR / "color_model.joblib"


def load_model() -> object:
    """Load trained color model."""
    return joblib.load(COLOR_MODEL_PATH)


def save_model(model: object) -> None:
    """Save trained color model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, COLOR_MODEL_PATH)


def features(card: np.ndarray) -> np.ndarray:
    """
    Extract color features from a card.

    Input:
        Color card (880, 630, 3), BGR.

    Output:
        Numerical feature vector.
    """

    if card.shape[:2] != (880, 630):
        raise ValueError(
            f"Expected (880, 630), got {card.shape[:2]}"
        )

    # Lower text box containing the large mana symbol.
    roi = card[
        int(880 * 0.62):int(880 * 0.92),
        int(630 * 0.12):int(630 * 0.88)
    ]

    # Convert to HSV and LAB.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # HSV histogram.
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [18, 8],
        [0, 180, 0, 256]
    )

    cv2.normalize(
        histogram,
        histogram,
        alpha=1,
        norm_type=cv2.NORM_L1
    )

    histogram = histogram.flatten()

    # Color statistics.
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    lab_mean = lab.mean(axis=(0, 1))
    lab_std = lab.std(axis=(0, 1))

    return np.concatenate([
        histogram,
        hsv_mean,
        hsv_std,
        lab_mean,
        lab_std
    ]).astype(np.float32)


def predict(
    model: object,
    features: np.ndarray
) -> str:
    """Predict white, green, black, blue or red."""
    return str(model.predict([features])[0])


def train(
    features: list[np.ndarray],
    labels: list[str]
) -> object:
    """
    Train a three-class color classifier.

    Labels:
        white
        green
        red
        black
        blue
    """

    allowed_labels = {"white", "green", "red", "blue", "black"}

    if not set(labels).issubset(allowed_labels):
        raise ValueError("Labels must be white, green, blue, black or red")

    if len(set(labels)) != 5:
        raise ValueError("Training requires all five colors")

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True
        )
    )

    model.fit(features, labels)

    return model