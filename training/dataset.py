"""Labels from file name (<kleur>_<schade>_<nr>.jpeg) + train/val/test split.

OPEN: split method (CSV vs folders, per physical card).
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    """Photo + labels."""
    path: Path
    kleur: str   # groen / rood / wit
    schade: str  # geen / kras / scheur / vouw / mix
    klasse: str  # "goed" if schade == "geen", else "beschadigd"


def parse_filename(path: Path) -> Sample:
    """wit_kras_03.jpeg -> Sample."""
    raise NotImplementedError("dataset.parse_filename")


def load_samples(folder: Path) -> list[Sample]:
    """All Samples in folder."""
    raise NotImplementedError("dataset.load_samples")


def split_samples(samples: list[Sample]) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """(train, val, test)."""
    raise NotImplementedError("dataset.split_samples")
