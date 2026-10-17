# Beslissingen

Elke ontwerpbeslissing: datum, beslissing, gekozen optie, alternatieven, reden.

---

## 2026-10-17 — Python-omgeving en dependency management

- **Beslissing:** hoe we Python en packages beheren, zodat de hele groep dezelfde versies gebruikt.
- **Gekozen:** Python 3.12 + `venv` + `requirements.txt` met vastgepinde versies (`package==x.y.z`).
- **Alternatieven:**
  - uv + `pyproject.toml` + lockfile: reproduceerbaar en snel, maar extra tool om te leren voor een project met weinig dependencies.
  - conda: zwaar, botst met bestaande venv, niet nodig voor OpenCV/scikit-learn.
- **Reden:** venv stond al klaar, iedereen kent het, werkt direct in PyCharm, en met vastgepinde versies is het genoeg om bij alle groepsleden hetzelfde resultaat te krijgen.

## 2026-10-17 — Aantal schadeklassen

- **Beslissing:** in hoeveel klassen we schade indelen.
- **Gekozen:** 2 klassen: **Goed** en **Beschadigd**. Waar de grens ligt, bepalen we later met een instelbare drempelwaarde (threshold) op een of meer gemeten waarden.
- **Alternatieven:** 3 klassen (Goed / Licht beschadigd / Ernstig beschadigd), zoals in de oorspronkelijke opzet.
- **Reden:** groepsbesluit. Eenvoudiger om te labelen en te classificeren. Met een threshold kunnen we de grens later bijstellen zonder het model of de labels opnieuw op te zetten.

## 2026-10-17 — Versiebeheer

- **Beslissing:** waar de git-repository van het project staat.
- **Gekozen:** een eigen repo in de projectmap `Pipline/`, branch `main`.
- **Alternatieven:** de bestaande repo in `C:\Users\laure` gebruiken (bevat de hele gebruikersmap en een ander project; risico op lekken van persoonlijke bestanden).
- **Reden:** het project scheiden van al het andere en het later te kunnen delen met de groep.
- **Genegeerd in git:** `.venv/` (iedereen maakt zijn eigen venv vanuit `requirements.txt`), `__pycache__/`, `.idea/` (PyCharm-instellingen zijn per persoon).

## 2026-10-17 — Mappenstructuur

- **Beslissing:** hoe we de projectmap indelen.
- **Gekozen:** één module per pipelinestap:
  - `pipeline/` — één `.py`-bestand per stap (bijv. `undistort.py`, `crop.py`, `edges.py`)
  - `data/` — foto's van kaarten
  - `output/` — tussenresultaten (plaatjes per stap) en eindresultaten
  - `docs/` — beslissingen en geleerde concepten
  - `main.py` — startpunt dat de hele pipeline draait
- **Alternatieven:**
  - Plat (losse scripts in de hoofdmap): snel, maar wordt rommelig bij ~8 stappen + classifier.
  - Jupyter-notebooks per stap: plaatjes direct zichtbaar, maar slecht te versiebeheren (merge-conflicten) en lastig als één doorlopende pipeline.
- **Reden:** elke stap is los te testen met een zichtbaar tussenresultaat (werkafspraak), en het blijft overzichtelijk en uitlegbaar.
- **Aanvulling:** de inhoud van `output/` wordt genegeerd in git (wordt steeds opnieuw gegenereerd); alleen `output/.gitkeep` wordt bijgehouden zodat de map bestaat.

## 2026-10-17 — Camerakalibratie (undistort) uitgesteld

- **Beslissing:** of we lensvervorming corrigeren met camerakalibratie (pipelinestap 2).
- **Gekozen:** voorlopig **overslaan**. We richten ons eerst op het uitsnijden en rechttrekken van de kaart (stap 3).
- **Alternatieven:** nu al kalibreren met een schaakbordpatroon (`cv2.calibrateCamera` + `cv2.undistort`).
- **Reden:** het is nog niet zeker welke camera we gebruiken (waarschijnlijk een iPhone). Kalibratie geldt per camera, dus nu kalibreren kan weggegooid werk zijn. De iPhone corrigeert lensvervorming al grotendeels in software.
- **TODO:** opnieuw bekijken zodra de camera vaststaat. Dan met een testfoto van een recht raster controleren of er nog zichtbare vervorming is (rechte lijnen die krom lopen, vooral aan de randen).

## 2026-10-17 — Kaart vinden in de foto (stap 3)

- **Beslissing:** hoe we de kaart in de foto vinden voordat we hem rechttrekken.
- **Gekozen:** **achtergrondkleur wegfilteren**: vaste, egale bandkleur met `cv2.inRange` in HSV wegfilteren → wat overblijft is de kaart → contour → 4 hoeken → `warpPerspective` naar verhouding 63:88 (standaard kaartformaat Pokémon én MTG).
- **Alternatieven:**
  - Contouren op randen (Canny + `findContours`): werkt op wisselende achtergronden, maar gevoelig voor weinig contrast en schaduwen van het schuine licht.
  - Neuraal netwerk (YOLO-segmentatie/SAM): robuust, maar zwaar, data nodig en slecht uitlegbaar.
- **Reden:** de achtergrond (lopende band) blijft altijd hetzelfde, dus een vaste kleur wegfilteren is simpel, betrouwbaar en weinig gevoelig voor schaduwen.
- **Gevolg:** het datasetplan in CLAUDE.md is aangepast van "verschillende achtergronden" naar "één vaste, egale achtergrond".
- **Let op:** kaarten met veel van de bandkleur kunnen gaten in het masker geven; bandkleur kiezen die weinig in kaarten voorkomt.

## 2026-10-17 — Beeldbewerkingslibrary

- **Beslissing:** welke library we gebruiken voor beeldbewerking.
- **Gekozen:** OpenCV (`opencv-python==5.0.0.93`) + NumPy (`numpy==2.5.4`), vastgezet in `requirements.txt`.
- **Alternatieven:** scikit-image (zwakker in perspectief/kalibratie, minder aansluiting op cursus), Pillow (mist contouren, Canny, perspectieffuncties).
- **Reden:** bevat alles voor de hele pipeline (masker, contouren, warp, Canny, filters, kalibratie) en sluit aan op de cursus. Niet de `-headless`-variant, zodat `cv2.imshow` werkt voor tussenresultaten.
- **Valkuil:** OpenCV laadt plaatjes als **BGR**, niet RGB.

## 2026-10-18 — Scope voor de eerste versie

- **Beslissing:** met welke kaarten we beginnen.
- **Gekozen:** alleen **MTG basic lands in 3 kleuren** (Forest/groen, Mountain/rood, Plains/wit), voorkant, zoals in de huidige dataset (`data/Dataset_MTG`).
- **Alternatieven:** meteen de volledige scope (Pokémon én MTG, voor- én achterkant).
- **Reden:** klein en overzichtelijk beginnen met de data die er al is; Pokémon en achterkanten later toevoegen.

## 2026-10-18 — OPEN: kaart vinden in de foto (herziening)

- **Status:** open, de groep bespreekt of de achtergrond verandert.
- **Aanleiding:** de huidige dataset is gefotografeerd op een donkergrijze stof met textuur. MTG-kaarten hebben een zwarte rand en Plains zijn bijna onverzadigd, dus "achtergrondkleur wegfilteren in HSV" (beslissing van 2026-10-17) werkt niet op deze foto's.
- **Test (4 foto's, S- en V-kanaal bekeken):** S scheidt Plains niet van de achtergrond (bij een blauwige foto is de achtergrond zelfs meer verzadigd dan de kaart). V werkt wel: het binnenvlak van de kaart is bijna wit, de achtergrond middengrijs, en de zwarte rand vormt er een donkere ring omheen. Valkuil: het schuine licht geeft een helderheidsverloop over de achtergrond.
- **Opties als de achtergrond blijft:** binnenvlak vinden op helderheid (V + Otsu) → grootste rechthoek → uitbreiden met de breedte van de zwarte rand, via (A) een vaste marge of (B) zoeken naar de overgang van zwart naar grijs.
- **Optie als de achtergrond verandert:** nieuwe foto's op een egale, groene ondergrond en dan de oorspronkelijke HSV-aanpak.

## 2026-10-18 — Dataset niet in git

- **Beslissing:** of de foto's in git komen.
- **Gekozen:** nee. `data/` staat in `.gitignore` (alleen `data/.gitkeep` wordt bijgehouden); foto's worden gedeeld via OneDrive/Teams.
- **Alternatieven:** foto's gewoon committen (~400 MB, maakt de repo traag); Git LFS (extra tool en opslaglimieten).
- **Reden:** foto's veranderen niet en zijn groot; git is voor code en documentatie.
