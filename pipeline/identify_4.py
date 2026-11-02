"""Step 4: which card is it? Game (MTG/Pokémon) and side (front/back).

Two models, each with: load model -> features -> prediction.
Both work on the color card from step 3. OPEN: features + classifier type.
"""
import numpy as np


def load_game_model() -> object:
    """Trained game model from models/. OPEN: file format depends on the library."""
    raise NotImplementedError("identify_4.load_game_model")


def game_features(card: np.ndarray) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> numbers for MTG vs Pokémon."""
    raise NotImplementedError("identify_4.game_features")


def predict_game(model: object, features: np.ndarray) -> str:
    """"mtg" or "pokemon"."""
    raise NotImplementedError("identify_4.predict_game")


def load_side_model() -> object:
    """Trained side model from models/."""
    raise NotImplementedError("identify_4.load_side_model")


def side_features(card: np.ndarray) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> numbers for front vs back."""
    raise NotImplementedError("identify_4.side_features")


def predict_side(model: object, features: np.ndarray) -> str:
    """"front" or "back"."""
    raise NotImplementedError("identify_4.predict_side")
