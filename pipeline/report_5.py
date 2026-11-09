"""Step 5: per card output/results/<name>/ with result.json + original photo,
and one overview of all cards of the run in output/results/overzicht.csv."""
import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline import config


@dataclass
class CardResult:
    """Result of one card. None = not known (model not built yet, or error)."""
    photo: str
    color: str | None = None   # "red" / "green" / "blue" / "black" / "white"
    damage: str | None = None  # "goed" / "beschadigd" (from damage_4.combine)
    # score 0-1 per damage model, e.g. {"lines": 0.1, "tear": None}
    damage_scores: dict[str, float | None] = field(default_factory=dict)
    # models the damage verdict is based on, e.g. ["lines"] while tear is not built yet
    damage_checked: list[str] = field(default_factory=list)
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


def write_overview(results: list[CardResult]) -> Path:
    """All cards of this run in one table: config.OVERVIEW_FILE. Returns its path."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = sorted({name for r in results for name in r.damage_scores})
    # utf-8-sig: Excel only reads é correctly with this marker; ";" + decimal comma: Dutch Excel
    # opens the file straight into columns (with "," and "." it puts everything in one column)
    with open(config.OVERVIEW_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["foto", "kleur", "schade", "gecontroleerd op",
                         *[f"score {name}" for name in model_names], "fout", "nog niet gebouwd"])
        for r in results:
            scores = [_decimal_comma(r.damage_scores.get(name)) for name in model_names]
            writer.writerow([r.photo, r.color or "", r.damage or "", ", ".join(r.damage_checked),
                             *scores, r.error or "", ", ".join(r.not_built)])
    return config.OVERVIEW_FILE


def summary(results: list[CardResult]) -> str:
    """Short text for the console: totals per verdict and per color."""
    lines = [f"{len(results)} kaarten: "
             + ", ".join(f"{sum(r.damage == v for r in results)} {v}" for v in ("goed", "beschadigd"))
             + f", {sum(r.error is not None for r in results)} met fout"]
    for color in sorted({r.color for r in results if r.color}):
        same = [r for r in results if r.color == color]
        lines.append(f"  {color:6s} {len(same):3d} kaarten, {sum(r.damage == 'beschadigd' for r in same)} beschadigd")
    return "\n".join(lines)


def _decimal_comma(score: float | None) -> str:
    """0.4321 -> "0,43"; None (model not built) -> empty cell."""
    return "" if score is None else f"{score:.2f}".replace(".", ",")
