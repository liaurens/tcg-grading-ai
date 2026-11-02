"""Step 4: which card is it? Game (MTG/Pokémon) and side (front/back).

Two models, each with: load/save model, features, predict, train.
Both work on the color card from step 3. OPEN: features + classifier type.
"""
import numpy as np


def load_game_model() -> object:
    """Trained game model from models/. OPEN: file format depends on the library."""
    raise NotImplementedError("identify_4.load_game_model")


def save_game_model(model: object) -> None:
    """Save the trained game model to models/."""
    raise NotImplementedError("identify_4.save_game_model")


def game_features(card: np.ndarray) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> numbers for MTG vs Pokémon."""
    raise NotImplementedError("identify_4.game_features")


def predict_game(model: object, features: np.ndarray) -> str:
    """"mtg" or "pokemon"."""
    raise NotImplementedError("identify_4.predict_game")


def train_game(features: list[np.ndarray], labels: list[str]) -> object:
    """Fit the game model. Labels: "mtg" / "pokemon"."""
    raise NotImplementedError("identify_4.train_game")


def load_side_model() -> object:
    """Trained side model from models/."""
    raise NotImplementedError("identify_4.load_side_model")


def save_side_model(model: object) -> None:
    """Save the trained side model to models/."""
    raise NotImplementedError("identify_4.save_side_model")


def side_features(card: np.ndarray) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> numbers for front vs back."""
    raise NotImplementedError("identify_4.side_features")


def predict_side(model: object, features: np.ndarray) -> str:
    """"front" or "back"."""
    raise NotImplementedError("identify_4.predict_side")


def train_side(features: list[np.ndarray], labels: list[str]) -> object:
    """Fit the side model. Labels: "front" / "back"."""
    raise NotImplementedError("identify_4.train_side")
