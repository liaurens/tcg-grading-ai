"""Step 7: game -> side -> damage. OPEN: classifier type."""
import numpy as np


def load_models() -> dict:
    """Load trained models from models/."""
    raise NotImplementedError("classify.load_models")


def classify_game(models: dict, features: np.ndarray) -> str:
    """"mtg" or "pokemon"."""
    raise NotImplementedError("classify.classify_game")


def classify_side(models: dict, features: np.ndarray) -> str:
    """"front" or "back"."""
    raise NotImplementedError("classify.classify_side")


def classify_damage(models: dict, features: np.ndarray) -> tuple[str, float]:
    """("goed" or "beschadigd", damage score)."""
    raise NotImplementedError("classify.classify_damage")
