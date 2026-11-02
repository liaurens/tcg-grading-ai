"""Labels from file name (<kleur>_<schade>_<nr>.jpeg) + train/val/test split.

Every model gets its own label per photo (decision 2026-11-02):
    kras   -> scratch = 1     vouw -> crease = 1     scheur -> tear = 1
    geen   -> 0 for every damage model
    mix    -> None: which damage is on it is not known yet, so not used for now
Game and side are not in the file names yet (only MTG fronts so far).

OPEN: label format (file name vs CSV), split method (per physical card).
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    """Photo + what the file name says."""
    path: Path
    kleur: str   # groen / rood / wit
    schade: str  # geen / kras / scheur / vouw / mix


def parse_filename(path: Path) -> Sample:
    """wit_kras_03.jpeg -> Sample."""
    raise NotImplementedError("dataset.parse_filename")


def load_samples(folder: Path) -> list[Sample]:
    """All Samples in folder."""
    raise NotImplementedError("dataset.load_samples")


def label(sample: Sample, model_name: str) -> int | str | None:
    """Label of this photo for one model (see table above). None = do not use this photo."""
    raise NotImplementedError("dataset.label")


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """(train, val, test)."""
    raise NotImplementedError("dataset.split_samples")
