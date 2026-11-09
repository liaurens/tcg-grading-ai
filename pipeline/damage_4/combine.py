"""Step 4c: scores of the damage models -> "goed" or "beschadigd" (decision 2026-11-07).

Rule: "beschadigd" as soon as ONE model scores at or above its own threshold (config.DAMAGE_THRESHOLDS).
Each model looks at a different kind of damage, so one clear signal is enough.
Models that are not built yet (score None) are skipped; main.py writes which models were
checked in result.json, so "goed" never silently means "also checked for tears".
"""
from pipeline import config


def combine(scores: dict[str, float | None]) -> str:
    """{"lines": 0.8, "tear": None} -> "goed" or "beschadigd"."""
    checked = {name: score for name, score in scores.items() if score is not None}
    if not checked:
        # no model gave a score: we cannot say anything, and "goed" would be a guess
        raise ValueError("geen enkel schademodel gaf een score")
    damaged = any(score >= config.DAMAGE_THRESHOLDS[name] for name, score in checked.items())
    return "beschadigd" if damaged else "goed"
