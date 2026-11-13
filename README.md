# TCG grading door AI (EASMVI70, groep 13)

Vereenvoudigde, PSA-achtige kwaliteitscontrole voor Magic: The Gathering-kaarten (voorkant).
Per foto: de **kleur** van de kaart en **goed** of **beschadigd** (krassen, vouwen, scheuren).

## Installeren

Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

De foto's staan niet in git; zet ze in `data/final_dataset/`.

## Draaien

```powershell
python main.py data\final_dataset\complete_dataset [--debug]
```

Uitvoer in `output/results/`: per kaart `result.json` + originele foto, `overzicht.csv` en `evaluatie.txt`.
Met `--debug` komen de tussenplaatjes in `output/debug/`.

## Trainen en testen

```powershell
python -m training.train lines        # lijnschade (krassen + vouwen)
python -m training.train tear         # scheuren
python -m training.evaluate <model>   # uitslag op de test-set
python -m training.train_color        # kleur
```

Getrainde modellen staan in `models/`, alle parameters in `pipeline/config.py`.
