"""Grade all card photos in a folder.

python main.py data/<folder> [--debug]
"""
import argparse
from pathlib import Path
from typing import Callable

from pipeline import crop_3, damage_5, identify_color_4, io_utils_1, report_6, undistort_2
from pipeline.damage_5 import combine


def run_if_built(step: Callable[[], object], result: report_6.CardResult) -> object | None:
    """Run one model. Not built yet: note it in the result and return None.

    So one unbuilt model does not stop the others (decision 2026-11-02).
    """
    try:
        return step()
    except NotImplementedError as missing:
        result.not_built.append(str(missing))
        return None


# Features before load_model in the 3 functions below: a model's preprocessing and debug
# images can be built and checked before any model is trained.

def predict_color(card, debug=False, card_name="card") -> str:
    features = identify_color_4.features(card)
    model = identify_color_4.load_model()

    if debug:
        return identify_color_4.predict_debug(
            model,
            card,
            features,
            card_name
        )

    return identify_color_4.predict(model, features)


def damage_score(model, card, show) -> float:
    features = model.features(model.preprocess(card, show))
    return model.predict(model.load_model(), features)


def grade_card(path: Path, debug: bool) -> report_6.CardResult:
    """Full pipeline for one photo."""
    card_name = path.stem

    def show(image, step_name):
        # save step image in debug mode
        if debug:
            io_utils_1.save_debug_image(image, card_name, step_name)

    image = io_utils_1.load_image(path)
    image = undistort_2.undistort(image)
    show(image, "2_undistort")

    card = crop_3.crop_card(image, show)
    show(card, "3_crop")

    result = report_6.CardResult(photo=path.name)

    # step 4: which card color. Models are loaded only here, so cropping works without any model.
    # TODO: load once for all photos as soon as the models exist
    result.color = run_if_built(
        lambda: predict_color(card, debug, card_name), 
        result
    )

    # step 5: every damage model on its own (the lambda runs right away, inside this loop turn)
    for name, model in damage_5.MODELS.items():
        result.damage_scores[name] = run_if_built(lambda: damage_score(model, card, show), result)

    # combine only when every model gave a score: a missing score must not count as "goed"
    if None not in result.damage_scores.values():
        result.damage = run_if_built(lambda: combine.combine(result.damage_scores), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade TCG card photos in a folder.")
    parser.add_argument("folder", type=Path, help="folder with card photos")
    parser.add_argument("--debug", action="store_true", help="save the image of every step")
    args = parser.parse_args()

    not_built = set()
    for path in io_utils_1.list_images(args.folder):
        # OPEN: catch crop_3.CardNotFoundError per photo
        result = grade_card(path, args.debug)
        report_6.write_result(result, path)
        not_built.update(result.not_built)
        print(f"{result.photo}: color={result.color} "
              f"damage={result.damage} scores={result.damage_scores}")

    if not_built:
        print("\nNog niet gebouwd (eerste ontbrekende functie per model):")
        for name in sorted(not_built):
            print(f"  {name}")


if __name__ == "__main__":
    main()
