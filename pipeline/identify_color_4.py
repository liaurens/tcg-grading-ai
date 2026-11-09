"""Step 4: which color card is it? Red, green, blue, black or white.

"""
import cv2
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pipeline import config

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

def predict_debug(
    model: object,
    card: np.ndarray,
    feature_vector: np.ndarray,
    card_name: str
) -> str:
    """Predict card color and save debug information."""

    # Extract ROI again for visualization
    roi = card[
        int(880 * 0.62):int(880 * 0.92),
        int(630 * 0.12):int(630 * 0.88)
    ]

    # Predict color
    prediction = str(model.predict([feature_vector])[0])

    # Get probabilities for each color
    probabilities = model.predict_proba([feature_vector])[0]
    classes = model.classes_

    # Save prediction results as JSON
    result = {
        "card": card_name,
        "prediction": prediction,
        "probabilities": {
            color: round(float(probability), 4)
            for color, probability in zip(classes, probabilities)
        }
    }

    folder = config.DEBUG_DIR / card_name

    json_path = folder / f"4_color.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    # Create debug visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Show ROI
    axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Color detection ROI")
    axes[0].axis("off")

    # Show probabilities
    axes[1].bar(classes, probabilities)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Predicted color: {prediction}")
    axes[1].set_ylabel("Probability")

    plt.tight_layout()

    plt.savefig(folder / f"4_color.png")
    plt.close(fig)

    return prediction

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