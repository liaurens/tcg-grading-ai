"""Score a main.py run against the labels in the file names, per split (train / validate / test).

Only when every photo is named <kleur>_<schade>_<nr> (our dataset); otherwise main.py skips this.
The split of a photo is looked up by its file name in config.DATASET_DIR/<split>/ (every name is
in exactly one split). Expected verdict: geen -> goed; kras, vouw, scheur, mix -> beschadigd.
Tear cards (scheur) count in the damage table and the missed list since the tear model exists (2026-11-13).
"""
from pathlib import Path

from pipeline import config
from pipeline.report_5 import CardResult
from training import dataset

COLOR_NAMES = {"blauw": "blue", "groen": "green", "rood": "red", "wit": "white", "zwart": "black"}
SPLIT_NAMES = {"train_dataset": "train", "validate_dataset": "validate", "test_dataset": "test"}


def evaluate_run(results: list[CardResult]) -> str | None:
    """Text with tables per split + list of wrong cards. None when the file names carry no labels."""
    try:
        samples = [dataset.parse_filename(Path(r.photo)) for r in results]
    except ValueError:
        return None  # own photos without <kleur>_<schade>_<nr>: nothing to compare with
    split_of = _split_of_photos()

    groups: dict[str, list[tuple[CardResult, dataset.Sample]]] = {}
    for result, sample in zip(results, samples):
        groups.setdefault(split_of.get(result.photo, "onbekend"), []).append((result, sample))
    order = [s for s in ("train", "validate", "test", "onbekend") if s in groups]
    groups["totaal"] = [pair for name in order for pair in groups[name]]
    order.append("totaal")

    out = ["EVALUATIE: oordeel van de pipeline tegen het label in de bestandsnaam",
           "(train: het schademodel is op deze foto's getraind, dus die cijfers zijn te optimistisch)", ""]
    out += _color_table(groups, order)
    out += _damage_table(groups, order)
    out += _wrong_cards(groups, order[:-1])
    return "\n".join(out)


def _split_of_photos() -> dict[str, str]:
    """Photo file name -> "train" / "validate" / "test", from the split folders of the dataset."""
    split_of = {}
    for folder, name in SPLIT_NAMES.items():
        if (config.DATASET_DIR / folder).is_dir():
            for sample in dataset.load_samples(config.DATASET_DIR / folder):
                split_of[sample.path.name] = name
    return split_of


def _usable(pairs):
    """Only cards without an error (a card that could not be cropped has no verdict)."""
    return [(r, s) for r, s in pairs if r.error is None]


def _pct(part: int, whole: int) -> str:
    return "   -" if whole == 0 else f"{part / whole:4.0%}"


def _binary(pairs: list[tuple[bool, bool]]) -> tuple[str, str, str, str]:
    """(predicted damaged, really damaged) pairs -> accuracy, precision, recall, F1 as text."""
    tp = sum(p and a for p, a in pairs)
    fp = sum(p and not a for p, a in pairs)
    fn = sum(a and not p for p, a in pairs)
    correct = sum(p == a for p, a in pairs)
    # F1 = 2*P*R / (P+R) = 2*tp / (2*tp + fp + fn): harmonic mean of precision and recall
    f1 = _pct(2 * tp, 2 * tp + fp + fn)
    return _pct(correct, len(pairs)), _pct(tp, tp + fp), _pct(tp, tp + fn), f1


def _color_table(groups, order) -> list[str]:
    out = ["KLEUR", f"  {'split':9s} {'n':>4s}  accuracy  fout"]
    for name in order:
        pairs = _usable(groups[name])
        right = sum(r.color == COLOR_NAMES.get(s.kleur) for r, s in pairs)
        out.append(f"  {name:9s} {len(pairs):4d}  {_pct(right, len(pairs)):>8s}  {len(pairs) - right:4d}")
    return out + [""]


def _damage_table(groups, order) -> list[str]:
    """Final verdict goed/beschadigd over all cards (tears count since the tear model exists, 2026-11-13)."""
    out = ["SCHADE, eindoordeel goed/beschadigd",
           f"  {'split':9s} {'n':>4s}  accuracy  precision  recall  F1  gave afgekeurd  kras gevonden  vouw gevonden  scheur gevonden"]
    for name in order:
        pairs = _usable(groups[name])
        acc, prec, rec, f1 = _binary([(r.damage == "beschadigd", s.schade != "geen") for r, s in pairs])

        def found(kind):
            same = [r for r, s in pairs if s.schade == kind]
            return f"{sum(r.damage == 'beschadigd' for r in same)}/{len(same)}"
        out.append(f"  {name:9s} {len(pairs):4d}  {acc:>8s}  {prec:>9s}  {rec:>6s}  {f1:>4s}  {found('geen'):>14s}  "
                   f"{found('kras'):>13s}  {found('vouw'):>13s}  {found('scheur'):>15s}")
    return out + [""]


def _wrong_cards(groups, order) -> list[str]:
    """Names of the wrong cards per split, with the score of every damage model, so you can look them up with --debug."""
    def score(r):
        return " ".join(f"{model} {'-' if s is None else f'{s:.2f}'}" for model, s in r.damage_scores.items())

    out = ["FOUTE KAARTEN (score per schademodel)"]
    for name in order:
        pairs = groups[name]
        errors = [r.photo for r, s in pairs if r.error is not None]
        color = [f"{r.photo} ({r.color})" for r, s in _usable(pairs) if r.color != COLOR_NAMES.get(s.kleur)]
        rejected = [f"{r.photo} ({score(r)})" for r, s in _usable(pairs) if s.schade == "geen" and r.damage == "beschadigd"]
        missed = [f"{r.photo} ({score(r)})" for r, s in _usable(pairs)
                  if s.schade != "geen" and r.damage == "goed"]
        out.append(f"  {name}:")
        out.append(f"    verkeerde kleur ({len(color)}): {', '.join(color) or '-'}")
        out.append(f"    gave kaart afgekeurd ({len(rejected)}): {', '.join(rejected) or '-'}")
        out.append(f"    schade gemist ({len(missed)}): {', '.join(missed) or '-'}")
        if errors:
            out.append(f"    niet verwerkt ({len(errors)}): {', '.join(errors)}")
    return out
