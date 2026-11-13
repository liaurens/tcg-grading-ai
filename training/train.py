"""Train one model and save it to models/.

python -m training.train <model> [--data <dataset folder>]     e.g. python -m training.train lines

Flow (same for every model): photos -> crop -> label for this model ->
model preprocess + features -> model.train -> model.save_model.
The model-specific part (classifier, library) lives in the model file itself.
Trains on train_dataset, then prints how it does on validate_dataset.
"""
import argparse
from pathlib import Path
from types import ModuleType

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pipeline import config, crop_2, damage_4, io_utils_1
from training import dataset

# every damage model is trained on its own; the color model has its own script (training.train_color)
MODEL_NAMES = [*damage_4.MODELS]

# Only for the printouts below: score >= 0.5 counts as "damaged". The real goed/beschadigd
# threshold is set in damage_4.combine (OPEN).
REPORT_THRESHOLD = 0.5
CLASS_NAMES = ["goed", "beschadigd"]  # label 0 / label 1
# photos that are left out of the report per model: the lines model does not look for tears,
# so a `scheur` photo is neither a hit nor a miss (tears are the tear model's job).
# The tear model needs no entry: kras/vouw photos have no tear label (dataset.LABELS), so they are never loaded for it.
REPORT_SKIP = {"lines": frozenset({"scheur"})}


def build_features(model: ModuleType, model_name: str, samples: list[dataset.Sample]
                   ) -> tuple[list[np.ndarray], list[int], list[dataset.Sample]]:
    """(features, labels, used samples) for one damage model. Skips photos without a label."""
    features, labels, used = [], [], []
    for sample in samples:
        sample_label = dataset.label(sample, model_name)
        if sample_label is None:
            continue
        # same steps 1-3 as main.py, so the model trains on exactly what it will see
        image = io_utils_1.load_image(sample.path)
        try:
            card = crop_2.crop_card(image)
        except crop_2.CardNotFoundError as error:
            print(f"  overgeslagen: {sample.path.name} ({error})")
            continue
        features.append(model.features(model.preprocess(card)))
        labels.append(sample_label)
        used.append(sample)
    return features, labels, used


def print_report(model: ModuleType, trained: object, features: list[np.ndarray], labels: list[int],
                 samples: list[dataset.Sample], title: str, skip_damage: frozenset = frozenset(),
                 per_photo: bool = False) -> None:
    """Same layout as the color model (training.train_color): accuracy, classification report, confusion matrix.

    skip_damage: damage words (from the file name) left out of the numbers, e.g. {"scheur"}: a tear is
    not what this model looks for, so it should not count as a hit or a miss. per_photo: also one line per photo.
    """
    keep = [i for i, s in enumerate(samples) if s.schade not in skip_damage]
    features = [features[i] for i in keep]
    labels = [labels[i] for i in keep]
    samples = [samples[i] for i in keep]
    scores = [model.predict(trained, f) for f in features]
    predicted = [int(score >= REPORT_THRESHOLD) for score in scores]

    print(f"\n{'=' * 45}")
    print(f"{title.upper()} RESULTS")
    print(f"{'=' * 45}")
    print(f"Photos: {len(labels)}" + (f"  ({', '.join(sorted(skip_damage))} photos not counted)" if skip_damage else ""))
    accuracy = accuracy_score(labels, predicted)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 0 = goed, 1 = beschadigd; confusion matrix rows = actual, columns = predicted (same order as the report)
    print("\nClassification report:")
    print(classification_report(labels, predicted, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(f"Rows = actual, columns = predicted  (order: {', '.join(CLASS_NAMES)})")
    print(confusion_matrix(labels, predicted, labels=[0, 1]))

    if per_photo:
        print("\nPer photo:")
        for sample, label, score, guess in zip(samples, labels, scores, predicted):
            mark = "" if guess == label else "   <-- wrong"
            print(f"  {sample.path.name:22s} label {label}  score {score:.2f}{mark}")


def train(model_name: str, dataset_dir: Path, per_photo: bool = False) -> None:
    """Run the flow above for one model."""
    if model_name not in damage_4.MODELS:
        raise NotImplementedError(f"train.train ({model_name})")  # only damage models here
    model = damage_4.MODELS[model_name]
    train_samples, val_samples, _test = dataset.load_split(dataset_dir)

    print(f"Model '{model_name}', dataset {dataset_dir}")
    train_features, train_labels, train_used = build_features(model, model_name, train_samples)
    print(f"Train: {len(train_labels)} foto's, {sum(train_labels)} met label 1")

    trained = model.train(train_features, train_labels)
    model.save_model(trained)
    print(f"Opgeslagen in {config.MODELS_DIR}")

    # on train it should do well; if val is much worse, the model learned the training photos by heart
    skip = REPORT_SKIP.get(model_name, frozenset())
    print_report(model, trained, train_features, train_labels, train_used, "Train", skip, per_photo)
    val_features, val_labels, val_used = build_features(model, model_name, val_samples)
    print_report(model, trained, val_features, val_labels, val_used, "Validation", skip, per_photo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model.")
    parser.add_argument("model", choices=MODEL_NAMES, help="which model to train")
    parser.add_argument("--data", type=Path, default=config.DATASET_DIR,
                        help="folder with train_dataset/, validate_dataset/, test_dataset/")
    parser.add_argument("--per-photo", action="store_true", help="also print the score of every photo")
    args = parser.parse_args()
    train(args.model, args.data, args.per_photo)


if __name__ == "__main__":
    main()
