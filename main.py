"""Grade all card photos in a folder.

python main.py data/<folder> [--debug]
"""
import argparse
from pathlib import Path
from typing import Callable

from pipeline import config, crop_2, damage_4, identify_color_3, io_utils_1, report_5
from pipeline.damage_4 import combine
from training import score_run


def run_if_built(step: Callable[[], object], result: report_5.CardResult) -> object | None:
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
    features = identify_color_3.features(card)
    model = identify_color_3.load_model()

    if debug:
        return identify_color_3.predict_debug(
            model,
            card,
            features,
            card_name
        )

    return identify_color_3.predict(model, features)


def damage_score(model, card, show) -> float:
    features = model.features(model.preprocess(card, show))
    return model.predict(model.load_model(), features)


def grade_card(path: Path, debug: bool) -> report_5.CardResult:
    """Full pipeline for one photo."""
    card_name = path.stem

    def show(image, step_name):
        # save step image in debug mode
        if debug:
            io_utils_1.save_debug_image(image, card_name, step_name)

    result = report_5.CardResult(photo=path.name)
    image = io_utils_1.load_image(path)

    try:
        card = crop_2.crop_card(image, show)
    except crop_2.CardNotFoundError as error:
        # no card shape found: record why and go on with the next photo instead of stopping the run
        result.error = f"kaart niet gevonden: {error}"
        return result
    show(card, "2_crop")

    # step 3: which card color. Models are loaded only here, so cropping works without any model.
    # TODO: load once for all photos as soon as the models exist
    result.color = run_if_built(
        lambda: predict_color(card, debug, card_name), 
        result
    )

    # step 4: every damage model on its own (the lambda runs right away, inside this loop turn)
    for name, model in damage_4.MODELS.items():
        result.damage_scores[name] = run_if_built(lambda: damage_score(model, card, show), result)

    # combine over the models that gave a score (decision 2026-11-07); damage_checked shows which ones,
    # so a "goed" while tear is not built yet is visibly "goed for line damage only"
    result.damage_checked = [name for name, score in result.damage_scores.items() if score is not None]
    if result.damage_checked:
        result.damage = run_if_built(lambda: combine.combine(result.damage_scores), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade TCG card photos in a folder.")
    parser.add_argument("folder", type=Path, help="folder with card photos")
    parser.add_argument("--debug", action="store_true", help="save the image of every step")
    args = parser.parse_args()

    not_built = set()
    results = []
    for path in io_utils_1.list_images(args.folder):
        result = grade_card(path, args.debug)
        report_5.write_result(result, path)
        results.append(result)
        not_built.update(result.not_built)
        # round only for the screen; result.json keeps the full score
        scores = ", ".join(f"{name} {'-' if s is None else f'{s:.2f}'}" for name, s in result.damage_scores.items())
        print(f"{result.photo:24s} kleur {str(result.color):6s} schade {str(result.damage):10s} ({scores})"
              + (f"  FOUT: {result.error}" if result.error else ""))

    overview = report_5.write_overview(results)
    print(f"\n{report_5.summary(results)}")

    # our dataset: compare with the labels in the file names, per split (own photos: skipped)
    evaluation = score_run.evaluate_run(results)
    if evaluation:
        print(f"\n{evaluation}")
        config.EVALUATION_FILE.write_text(evaluation, encoding="utf-8")

    print(f"\nOverzicht van alle kaarten: {overview}")
    if evaluation:
        print(f"Evaluatie per split:        {config.EVALUATION_FILE}")
    print(f"Per kaart (result.json + foto): {config.RESULTS_DIR}")

    if not_built:
        print("\nNog niet gebouwd (eerste ontbrekende functie per model):")
        for name in sorted(not_built):
            print(f"  {name}")


if __name__ == "__main__":
    main()
