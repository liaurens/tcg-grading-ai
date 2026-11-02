"""Train one model and save it to models/.

python -m training.train <model>        e.g. python -m training.train tear

Flow (same for every model): photos -> crop -> label for this model ->
model preprocess + features -> model.train -> model.save_model.
The model-specific part (classifier, library) lives in the model file itself.
"""
import argparse

from pipeline import damage_5

# every model is trained on its own
MODEL_NAMES = ["game", "side", *damage_5.MODELS]


def train(model_name: str) -> None:
    """Run the flow above for one model."""
    raise NotImplementedError(f"train.train ({model_name})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model.")
    parser.add_argument("model", choices=MODEL_NAMES, help="which model to train")
    args = parser.parse_args()
    train(args.model)


if __name__ == "__main__":
    main()
