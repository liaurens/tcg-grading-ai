"""Step 8: per card output/results/<name>/ with result.json + original photo."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CardResult:
    """Result of one card."""
    photo: str
    game: str | None = None
    side: str | None = None
    damage: str | None = None
    damage_score: float | None = None
    error: str | None = None  # e.g. "card not found"


def write_result(result: CardResult, original_path: Path) -> Path:
    """Write result.json + copy of photo, return the card folder."""
    # TODO: json.dump(asdict(result)), shutil.copy2(original_path)
    raise NotImplementedError("report.write_result")
