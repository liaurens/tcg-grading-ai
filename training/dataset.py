"""Labels from file name (<kleur>_<schade>_<nr>.jpeg) + train/val/test split.

Every model gets its own label per photo (decisions 2026-11-02 and 2026-11-05):
    kras, vouw -> lines = 1    scheur -> tear = 1
    geen       -> 0 for every damage model
    scheur     -> lines = 0
    mix        -> None: which damage is on it is not known yet, so not used for now
Only MTG fronts: game and side are out of scope (decision 2026-11-07).

Split: the existing folders train_dataset / validate_dataset / test_dataset in one
dataset folder (config.DATASET_DIR, or --data). Decision 2026-11-05.
final_dataset has one subfolder per color inside each split; photos are found there too.
OPEN: label format (file name vs CSV).
"""
from dataclasses import dataclass
from pathlib import Path

from pipeline import config


@dataclass
class Sample:
    """Photo + what the file name says."""
    path: Path
    kleur: str   # blauw / groen / rood / wit / zwart
    schade: str  # geen / kras / scheur / vouw / mix


# damage word in the file name -> label per model. Missing = None = photo not used for that model.
LABELS = {
    "lines": {"kras": 1, "vouw": 1, "geen": 0, "scheur": 0},  # decision 2026-11-05
    "tear": {"scheur": 1, "geen": 0},  # OPEN: kras/vouw for tear (owner of tear.py decides)
}

# folder name -> part of the split (decision 2026-11-05: use the existing folders)
SPLIT_FOLDERS = ("train_dataset", "validate_dataset", "test_dataset")


def parse_filename(path: Path) -> Sample:
    """wit_kras_03.jpeg -> Sample."""
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"{path.name}: expected <kleur>_<schade>_<nr>, e.g. wit_kras_03.jpeg")
    kleur, schade, _number = parts
    return Sample(path=path, kleur=kleur, schade=schade)


def load_samples(folder: Path) -> list[Sample]:
    """All Samples in folder, also in its subfolders (final_dataset: train_dataset/black/...)."""
    # rglob("*"): every file at any depth, so a flat folder works as well
    # lower(): iPhone sometimes writes .JPEG
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"Geen foto's in map of submappen: {folder}")
    return [parse_filename(path) for path in paths]


def label(sample: Sample, model_name: str) -> int | None:
    """Label of this photo for one model (see table above). None = do not use this photo."""
    if model_name not in LABELS:
        # only damage models have a label table; color uses training.train_color
        raise NotImplementedError(f"dataset.label for {model_name}")
    return LABELS[model_name].get(sample.schade)


def load_split(dataset_dir: Path) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """(train, val, test) from the 3 folders in dataset_dir."""
    train, val, test = (load_samples(dataset_dir / name) for name in SPLIT_FOLDERS)
    return train, val, test
