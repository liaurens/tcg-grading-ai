# TCG grading door AI (EASMVI70, groep 13)

Een vereenvoudigde, PSA-achtige kwaliteitscontrole voor ruilkaarten. Per foto van één kaart bepaalt de pipeline:
Pokémon of MTG, voor- of achterkant, en **Goed** of **Beschadigd** (krassen, vouwen, scheuren).

## Pipeline

| Stap | Wat | Bestand | Status |
| --- | --- | --- | --- |
| 1 | Foto laden | `pipeline/io_utils.py` | klaar |
| 2 | Lensvervorming corrigeren | `pipeline/undistort.py` | uitgesteld (geeft foto ongewijzigd terug) |
| 3 | Kaart vinden, uitsnijden, rechttrekken naar 630×880 | `pipeline/crop.py` | klaar |
| 4–6 | Grijswaarden, Canny, high-pass | `pipeline/preprocess.py` | te bouwen |
| 7 | Classificatie: game, kant, schade | `pipeline/features.py`, `pipeline/classify.py` | te bouwen |
| 8 | Resultaat + originele foto opslaan | `pipeline/report.py` | te bouwen |

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

Per kaart komt er een map `output/results/<fotonaam>/` met `result.json` (game, kant, schade, schadescore, fout) en een kopie van de originele foto.

> Nog niet alle stappen zijn gebouwd. Het programma stopt dan met een melding als
> `Pipeline stopt: 'preprocess.to_gray' is nog niet gebouwd.` Dat is de eerstvolgende functie die gebouwd moet worden.

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

### Eén stap los gebruiken (vanuit Python)

Elke stap is een gewone functie. Bijvoorbeeld alleen het uitsnijden van één foto:

```python
from pathlib import Path
import cv2
from pipeline import crop, io_utils

image = io_utils.load_image(Path("data/schade_class_v2/complete_dataset/wit_kras_02.jpeg"))
card = crop.crop_card(image)                 # array (880, 630, 3): hoogte, breedte, BGR
cv2.imwrite("output/kaart.png", card)
```

Wil je daarbij ook de tussenplaatjes, geef dan een `show`-functie mee:

```python
show = lambda img, step: io_utils.save_debug_image(img, "wit_kras_02", step)
card = crop.crop_card(image, show)
```

Lukt het uitsnijden niet (geen kaartvorm gevonden), dan volgt `crop.CardNotFoundError` met de reden.
