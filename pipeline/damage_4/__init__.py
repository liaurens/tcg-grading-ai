"""Step 4: damage. One model per damage type (decision 2026-11-02).

Every model file has the same 6 functions, so main and training call them the same way:
    load_model()               -> trained model from models/
    save_model(model)          -> write trained model to models/
    preprocess(card, show)     -> image in which this damage type stands out
    features(image)            -> numbers for the classifier
    predict(model, features)   -> damage score 0-1 (higher = more likely damaged)
    train(features, labels)    -> trained model (labels: 1 = this damage, 0 = not)

Debug images of a model: show(image, "4a_..."), with the letter of that model below.

Add a model: copy lines.py, build the 6 functions, add one line to MODELS.
combine.py turns the scores of all models into "goed" or "beschadigd".
"""
from pipeline.damage_4 import lines, tear

# name -> module; main and training/ loop over this, in this order
MODELS = {
    "lines": lines,  # 4a: scratches + creases (decision 2026-11-05)
    "tear": tear,    # 4b: tears, white paper showing (decision 2026-11-13)
}
