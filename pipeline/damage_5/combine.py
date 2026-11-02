"""Step 5d: scores of all damage models -> "goed" or "beschadigd".

OPEN: how scores are combined (e.g. damaged as soon as one model is above its own
threshold, or one combined score) and the threshold values (in config.py).
"""


def combine(scores: dict[str, float]) -> str:
    """{"scratch": 0.1, "crease": 0.8, "tear": 0.0} -> "goed" or "beschadigd"."""
    raise NotImplementedError("damage_5.combine.combine")
