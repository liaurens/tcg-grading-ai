"""Test one model on the test set.

python -m training.evaluate <model>     e.g. python -m training.evaluate tear

OPEN: metrics (accuracy, precision/recall, confusion matrix) and which error is worst.
"""
import argparse

from training.train import MODEL_NAMES


def evaluate(model_name: str) -> None:
    """Load the trained model, predict on the test set, print the metrics."""
    raise NotImplementedError(f"evaluate.evaluate ({model_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test one model on the test set.")
    parser.add_argument("model", choices=MODEL_NAMES, help="which model to test")
    args = parser.parse_args()
    evaluate(args.model)


if __name__ == "__main__":
    main()
