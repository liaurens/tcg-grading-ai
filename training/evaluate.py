"""Test one model on the test set.

python -m training.evaluate <model> [--data <dataset folder>]     e.g. python -m training.evaluate lines

Prints accuracy, classification report and confusion matrix in the layout of the color model (see train.print_report).
OPEN: which metric counts most / which error is worst (decision still to be taken).
"""
import argparse
from pathlib import Path

from pipeline import config, damage_4
from training import dataset
from training.train import MODEL_NAMES, REPORT_SKIP, build_features, print_report


def evaluate(model_name: str, dataset_dir: Path, per_photo: bool = False) -> None:
    """Load the trained model, predict on the test set, print the metrics."""
    if model_name not in damage_4.MODELS:
        raise NotImplementedError(f"evaluate.evaluate ({model_name})")  # only damage models here
    model = damage_4.MODELS[model_name]
    trained = model.load_model()
    _train, _val, test_samples = dataset.load_split(dataset_dir)

    features, labels, used = build_features(model, model_name, test_samples)
    print(f"Model '{model_name}', dataset {dataset_dir}")
    print_report(model, trained, features, labels, used, "Test", REPORT_SKIP.get(model_name, frozenset()), per_photo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test one model on the test set.")
    parser.add_argument("model", choices=MODEL_NAMES, help="which model to test")
    parser.add_argument("--data", type=Path, default=config.DATASET_DIR,
                        help="folder with train_dataset/, validate_dataset/, test_dataset/")
    parser.add_argument("--per-photo", action="store_true", help="also print the score of every photo")
    args = parser.parse_args()
    evaluate(args.model, args.data, args.per_photo)


if __name__ == "__main__":
    main()
