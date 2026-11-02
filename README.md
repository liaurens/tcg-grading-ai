# TCG grading door AI (EASMVI70, groep 13)

Een vereenvoudigde, PSA-achtige kwaliteitscontrole voor ruilkaarten. Per foto van één kaart bepaalt de pipeline:
Pokémon of MTG, voor- of achterkant, en **Goed** of **Beschadigd** (krassen, vouwen, scheuren).

## Pipeline

Multistage: elk onderdeel heeft een eigen model (of, bij het uitsnijden, eigen regels), dat los te bouwen, trainen en testen is.

| Stap | Wat | Bestand | Status |
| --- | --- | --- | --- |
| 1 | Foto laden | `pipeline/io_utils_1.py` | klaar |
| 2 | Lensvervorming corrigeren | `pipeline/undistort_2.py` | uitgesteld (geeft foto ongewijzigd terug) |
| 3 | Kaart vinden, uitsnijden, rechttrekken naar 630×880 (regels, geen model) | `pipeline/crop_3.py` | klaar |
| 4 | Welke kaart: game (MTG/Pokémon) en kant (voor/achter), 2 modellen | `pipeline/identify_4.py` | te bouwen |
| 5a | Krassen-model | `pipeline/damage_5/scratch.py` | te bouwen |
| 5b | Vouwrandjes-model | `pipeline/damage_5/crease.py` | te bouwen |
| 5c | Afscheuringen-model | `pipeline/damage_5/tear.py` | te bouwen |
| 5d | Scores samenvoegen tot Goed/Beschadigd | `pipeline/damage_5/combine.py` | te bouwen |
| 6 | Resultaat + originele foto opslaan | `pipeline/report_6.py` | klaar |

Uitgebreide uitleg per stap (wat, waarom, hoe): [`docs/pipeline.md`](docs/pipeline.md).

Alle parameters staan met naam en reden in `pipeline/config.py`. Trainingscode staat in `training/`, getrainde modellen in `models/`.

## Installeren

Python 3.12:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

De foto's staan niet in git. Zet de dataset (via OneDrive/Teams) in `data/`, bijvoorbeeld `data/schade_class_v2/complete_dataset/`.

## Gebruik

### Hele pipeline (eindresultaat)

```
python main.py data/schade_class_v2/complete_dataset
```

Per kaart komt er een map `output/results/<fotonaam>/` met `result.json` (game, kant, schade, score per schademodel, fout, nog niet gebouwd) en een kopie van de originele foto.

> Een model dat nog niet gebouwd is, wordt overgeslagen (waarde `null`); de andere modellen draaien gewoon door.
> Aan het eind meldt het programma per model de eerste functie die nog gebouwd moet worden, bijv. `damage_5.tear.preprocess`.

### Tussenstappen bekijken (`--debug`)

```
python main.py data/schade_class_v2/complete_dataset --debug
```

Slaat per kaart een PNG per (sub)stap op in `output/debug/<fotonaam>/`. Zonder `--debug` wordt er niets opgeslagen.

| Bestand | Wat je ziet |
| --- | --- |
| `2_undistort.png` | Foto na stap 2 (nu nog gelijk aan het origineel) |
| `3a_blur.png` | Grijze, vervaagde kopie op ¼ grootte |
| `3b_canny.png` | Gevonden randen |
| `3c_closed.png` | Randen met de gaatjes dichtgemaakt |
| `3d_mask.png` | Masker: kaart wit, achtergrond zwart |
| `3e_corners.png` | Masker met de gevonden rechthoek en hoeknummers 0–3 |
| `3_crop.png` | Rechtgetrokken kaart, 630×880 |
| `5a_…`, `5b_…`, `5c_…` | Tussenplaatjes van het krassen-, vouwrandjes- en afscheuringenmodel |

### Eén stap los gebruiken (vanuit Python)

Elke stap is een gewone functie. Bijvoorbeeld alleen het uitsnijden van één foto:

```python
from pathlib import Path
import cv2
from pipeline import crop_3, io_utils_1

image = io_utils_1.load_image(Path("data/schade_class_v2/complete_dataset/wit_kras_02.jpeg"))
card = crop_3.crop_card(image)               # array (880, 630, 3): hoogte, breedte, BGR
cv2.imwrite("output/kaart.png", card)
```

Wil je daarbij ook de tussenplaatjes, geef dan een `show`-functie mee:

```python
show = lambda img, step: io_utils_1.save_debug_image(img, "wit_kras_02", step)
card = crop_3.crop_card(image, show)
```

Lukt het uitsnijden niet (geen kaartvorm gevonden), dan volgt `crop_3.CardNotFoundError` met de reden.

## Een model trainen en testen

Elk model wordt los getraind en getest:

```
python -m training.train tear        # of: game, side, scratch, crease
python -m training.evaluate tear
```

## Een eigen schademodel bouwen of toevoegen

Elk schademodel is één bestand in `pipeline/damage_5/` met dezelfde 6 functies, zodat `main.py` en `training/` ze allemaal op dezelfde manier aanroepen:

| Functie | Doet |
| --- | --- |
| `preprocess(card, show)` | kaart → plaatje waarin deze schade opvalt; tussenplaatjes via `show(img, "5c_…")` |
| `features(image)` | plaatje → getallen voor de classifier |
| `train(features, labels)` | getraind model (label 1 = deze schade, 0 = niet) |
| `save_model(model)` / `load_model()` | model opslaan in / laden uit `models/` |
| `predict(model, features)` | schadescore 0–1 |

- **Bouwvolgorde:** begin met `preprocess` en `features` en bekijk de plaatjes met `--debug`. Daarvoor is nog geen getraind model nodig.
- **Nieuw model** (bijv. snedes): kopieer `scratch.py` naar `cut.py`, bouw de 6 functies en zet één regel in `MODELS` in `pipeline/damage_5/__init__.py`.
- **Parameters** van je model komen in `pipeline/config.py`, onder een eigen kopje.
- **Samenwerken:** werk in je eigen modelbestand; dan zit je niet in de code van een ander.
