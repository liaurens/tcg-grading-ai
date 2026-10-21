# Beslissingen (kort)

Snelle samenvatting van `beslissingen.md`. Daar staan de alternatieven en de volledige redenering.

## Genomen

| Datum | Onderwerp | Keuze | Waarom |
|---|---|---|---|
| 10-17 | Python | 3.12 + venv + `requirements.txt` (vaste versies) | Iedereen kent het, werkt in PyCharm |
| 10-17 | Schadeklassen | 2: Goed / Beschadigd, grens via threshold | Simpeler labelen, grens later bij te stellen |
| 10-17 | Git | Eigen repo in `Pipline/`, branch `main` | Gescheiden van de rest van de gebruikersmap |
| 10-17 | Mappen | `pipeline/` (1 bestand per stap), `data/`, `output/`, `docs/`, `main.py` | Elke stap los te testen |
| 10-17 | Library | OpenCV `5.0.0.93` + NumPy `2.5.4` | Alles voor de pipeline, sluit aan op cursus |
| 10-17 | Undistort | Uitgesteld | Camera staat nog niet vast (waarschijnlijk iPhone) |
| 10-18 | Scope v1 | Alleen MTG basic lands (groen/rood/wit), voorkant | Klein beginnen met de data die er is |
| 10-18 | Dataset | Niet in git, delen via OneDrive/Teams | ~400 MB, verandert niet |
| 10-21 | Input | Een map met foto's | Hele dataset in één keer |
| 10-21 | Output | `output/results/<foto>/result.json` + origineel | Leesbaar, past bij "resultaat + origineel" |
| 10-21 | Parameters | Centraal in `pipeline/config.py` | Alles op één plek te tunen |
| 10-21 | Training | Aparte map `training/`, modellen in `models/` | Beoordelen en model maken gescheiden |

## Kaart vinden in de foto (besloten 10-21: witte achtergrond + Canny)

**Nieuw 10-21 op witte achtergrond (`schade_class_v2`, 40 foto's):**

| Methode | Vorm ok |
|---|---|
| A. oud (helder binnenvlak) | 0/40 |
| B. donkere rand + Otsu | 35/40 |
| **C. Canny-contour (gekozen)** | **40/40** |

Waarom C: beste score en werkt ook voor Pokémon (gele rand). Check is vorm + visueel, alleen Plains getest.

**Oud (10-18, grijze stof):**

Eerste plan (10-17): vaste bandkleur wegfilteren in HSV met `cv2.inRange`. Dat werkt niet op de huidige donkergrijze stof, want Plains zijn bijna onverzadigd. De groep bespreekt nog of de achtergrond verandert.

Experiment 10-18 op de huidige achtergrond (V ≥ 100, opening, grootste vlak, `minAreaRect`, `warpPerspective` naar 630×880):

| Set | Foto's | Goed | Percentage | Fouten |
|---|---|---|---|---|
| `schade_class` (instellingen hierop afgestemd) | 39 | 37 | 95% | `wit_geen_02`, `wit_geen_09` |
| `kleur_class` (nieuwe foto's) | 20 | 16 | 80% | `groen_geen_05`, `groen_geen_10`, `rood_geen_05`, `rood_geen_06` |
| Totaal | 59 | 53 | 90% | |

- Meestal gaat het fout doordat alleen de lichte helft van de kaart wordt gevonden (donkere art valt weg).
- Een andere achtergrond helpt weinig bij krassen en vouwen, maar wel bij scheuren en randschade.
- Advies Claude: matte, egale (groene) ondergrond. Eerst een proef met 5 foto's.

## Nog te beslissen

1. ~~Achtergrond en uitsnijmethode~~ (besloten, zie boven); nog te testen op groene/rode kaarten
2. Canny- en high-pass-parameters
3. Dataset: aantal foto's per klasse, split train/val/test (per fysieke kaart), labelformaat
4. Features + classifier, of CNN
5. Oriëntatie: normaliseren of augmentatie (pas bij voor/achter en Pokémon)
6. Metrics en welke fout het ergst is
7. Threshold goed/beschadigd
8. Camera, belichting, kalibratie
9. Hoe het resultaat verstuurd wordt
