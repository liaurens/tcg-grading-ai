
import sys
from pathlib import Path

import cv2
import numpy as np
import argparse

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import crop_3, undistort_2, io_utils_1, identify_color_4

# Dataset locaties
DATASET_DIR = PROJECT_ROOT / "data" 

TRAIN_DIR = DATASET_DIR / "train_dataset"
VALIDATE_DIR = DATASET_DIR / "validate_dataset"
TEST_DIR = DATASET_DIR / "test_dataset"

CLASSES = ["red", "green", "white", "black", "blue"]

IMAGE_EXTENSIONS = {".jpeg"}

def load_dataset(folder: Path):
    """Load images and extract features."""

    features = []
    labels = []

    for label in CLASSES:

        class_folder = folder / label

        if not class_folder.exists():
            raise FileNotFoundError(class_folder)

        image_paths = [
            p for p in class_folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"\nLoading {label}: {len(image_paths)} images")

        for path in image_paths:

            image = cv2.imread(str(path))

            if image is None:
                print(f"Could not load: {path}")
                continue

            # Stap 1: perspectiefcorrectie
            image = undistort_2.undistort(image)

            # Stap 2: kaart uitsnijden
            card = crop_3.crop_card(image, io_utils_1.no_show)

            # Extract features
            feature = identify_color_4.features(card)

            features.append(feature)
            labels.append(label)

    if not features:
        raise ValueError(f"No images found in {folder}")

    return np.array(features), np.array(labels)


def evaluate_model(model, features, labels, dataset_name):
    """Evaluate classifier on a dataset."""

    predictions = model.predict(features)

    accuracy = accuracy_score(labels, predictions)

    print(f"\n{'=' * 45}")
    print(f"{dataset_name.upper()} RESULTS")
    print(f"{'=' * 45}")

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")

    print(classification_report(
        labels,
        predictions,
        labels=CLASSES,
        zero_division=0
    ))

    print("Confusion matrix:")
    print("Rows = actual, columns = predicted")

    print(confusion_matrix(
        labels,
        predictions,
        labels=CLASSES
    ))

    return accuracy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test an already trained model"
    )

    args = parser.parse_args()

    # TEST MODE
    if args.test:

        print("\nLoading saved model...")
        model = identify_color_4.load_model()

        print("\nLoading test dataset...")
        X_test, y_test = load_dataset(TEST_DIR)

        evaluate_model(
            model,
            X_test,
            y_test,
            "Test"
        )

        return

    # TRAIN MODE
    print("\nLoading training dataset...")
    X_train, y_train = load_dataset(TRAIN_DIR)

    print("\nTraining color classifier...")

    model = identify_color_4.train(
        X_train.tolist(),
        y_train.tolist()
    )

    # VALIDATION
    print("\nLoading validation dataset...")
    X_validate, y_validate = load_dataset(VALIDATE_DIR)

    evaluate_model(
        model,
        X_validate,
        y_validate,
        "Validation"
    )

    # SAVE
    identify_color_4.save_model(model)

    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()