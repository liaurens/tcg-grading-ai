"""Grade all card photos in a folder.

python main.py data/<folder> [--debug]
"""
import argparse
import sys
from pathlib import Path

from pipeline import classify, crop, features, io_utils, preprocess, report, undistort


def grade_card(path: Path, models: dict, debug: bool) -> report.CardResult:
    """Full pipeline for one photo."""
    card_name = path.stem

    def show(image, step_name):
        # save step image in debug mode
        if debug:
            io_utils.save_debug_image(image, card_name, step_name)

    image = io_utils.load_image(path)
    image = undistort.undistort(image)
    show(image, "2_undistort")

    card = crop.crop_card(image)
    show(card, "3_crop")

    gray = preprocess.to_gray(card)
    show(gray, "4_gray")
    edges = preprocess.detect_edges(gray)
    show(edges, "5_canny")
    detail = preprocess.high_pass(gray)
    show(detail, "6_high_pass")

    game = classify.classify_game(models, features.game_features(card))
    side = classify.classify_side(models, features.side_features(card))
    damage, score = classify.classify_damage(models, features.damage_features(gray, edges, detail))

    return report.CardResult(photo=path.name, game=game, side=side,
                             damage=damage, damage_score=score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade TCG card photos in a folder.")
    parser.add_argument("folder", type=Path, help="folder with card photos")
    parser.add_argument("--debug", action="store_true", help="save the image of every step")
    args = parser.parse_args()

    try:
        models = classify.load_models()
        for path in io_utils.list_images(args.folder):
            # OPEN: catch crop.CardNotFoundError per photo
            result = grade_card(path, models, args.debug)
            report.write_result(result, path)
    except NotImplementedError as missing:
        # skeleton: report first unbuilt function
        sys.exit(f"Pipeline stopt: '{missing}' is nog niet gebouwd.")


if __name__ == "__main__":
    main()
