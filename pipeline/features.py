"""Images -> numbers for the classifiers. OPEN: which features."""
import numpy as np


def game_features(card: np.ndarray) -> np.ndarray:
    """Features for MTG vs Pokémon."""
    raise NotImplementedError("features.game_features")


def side_features(card: np.ndarray) -> np.ndarray:
    """Features for front vs back."""
    raise NotImplementedError("features.side_features")


def damage_features(gray: np.ndarray, edges: np.ndarray, high_pass: np.ndarray) -> np.ndarray:
    """Features for scratches, creases, tears."""
    raise NotImplementedError("features.damage_features")
