"""Step 6: per card output/results/<name>/ with result.json + original photo."""
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline import config


@dataclass
class CardResult:
    """Result of one card. None = not known (model not built yet, or error)."""
    photo: str
    game: str | None = None    # "mtg" / "pokemon"
    side: str | None = None    # "front" / "back"
    damage: str | None = None  # "goed" / "beschadigd" (from damage_5.combine)
    # score 0-1 per damage model, e.g. {"scratch": 0.1, "crease": None, "tear": 0.9}
    damage_scores: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None   # e.g. "card not found"
    # first unbuilt function per model, so you see what is still missing
    not_built: list[str] = field(default_factory=list)


def write_result(result: CardResult, original_path: Path) -> Path:
    """Write result.json + copy of photo, return the card folder."""
    folder = config.RESULTS_DIR / original_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False: keep é in "Pokémon" readable; indent: readable by hand
    with open(folder / "result.json", "w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2, ensure_ascii=False)
    # copy2 also keeps the photo's date
    shutil.copy2(original_path, folder / original_path.name)
    return folder
