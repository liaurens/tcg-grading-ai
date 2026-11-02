"""Step 5: damage. One model per damage type (decision 2026-11-02).

Every model file has the same 4 functions, so main and training call them the same way:
    load_model()               -> trained model from models/
    preprocess(card, show)     -> image in which this damage type stands out
    features(image)            -> numbers for the classifier
    predict(model, features)   -> damage score 0-1 (higher = more likely damaged)

combine.py turns the scores of all models into "goed" or "beschadigd".
"""
