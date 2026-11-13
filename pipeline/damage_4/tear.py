"""Step 4b: tears (scheuren): the card is torn in from the edge (decision 2026-11-13).

Same recipe as the lines model (4a): standard OpenCV/NumPy features -> StandardScaler + SVC (scikit-learn).
Idea: where a card is torn, the print is gone and the white paper core shows. Bare paper is light AND
uncolored, the print almost never is both.
Preprocessing: "whiteness" image = per pixel the smallest of B, G, R.
Features (18 numbers):
  - histogram of the whiteness values (10): how much of the card is how white
  - white blobs (8): per size class (0.3-1, 1-3, 3-10, >10 mm2) how many blobs and their total area
The classifier learns from the labelled photos which of these numbers point at a tear.
Known weak spot: a crease also cracks the print and shows white paper, so creased cards score high too.
For the final verdict that is no error (both are "beschadigd"), but this model cannot tell the two apart.
"""
import cv2
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline import config
from pipeline.io_utils_1 import ShowFn, no_show

# only for the debug image 4b_blobs: one BGR color per size class (small -> large), gray = too small to count
_DEBUG_CLASS_COLORS = [(255, 120, 0), (0, 200, 0), (0, 255, 255), (0, 0, 255)]
_DEBUG_TOO_SMALL_COLOR = (90, 90, 90)


def load_model() -> object:
    """Trained tear model from models/."""
    if not config.TEAR_MODEL_FILE.exists():
        raise FileNotFoundError(f"{config.TEAR_MODEL_FILE} not found: run python -m training.train tear")
    return joblib.load(config.TEAR_MODEL_FILE)


def save_model(model: object) -> None:
    """Save the trained model to models/ (same format load_model reads)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.TEAR_MODEL_FILE)


def preprocess(card: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> whiteness image (880, 630) uint8.

    Each pixel: the smallest of its B, G, R values (255 = pure white, low = dark or colored).
    """
    # axis=2 is the color axis: a light blue text box has a high B but a lower R, so its minimum stays low;
    # bare paper has all three high
    white = card.min(axis=2)
    show(white, "4b_white")

    if show is not no_show:  # only for --debug: show the blobs features() will count
        m = config.TEAR_MARGIN_PX
        labels, areas = find_blobs(white[m:-m, m:-m])
        colors = np.full((len(areas), 3), _DEBUG_TOO_SMALL_COLOR, dtype=np.uint8)
        colors[0] = 0  # number 0 = not white: black
        for (low_mm2, high_mm2), color in zip(config.TEAR_AREA_CLASSES_MM2, _DEBUG_CLASS_COLORS):
            colors[_in_class(areas, low_mm2, high_mm2)] = color
        # colors[labels]: look up the color of every pixel's blob number -> (height, width, 3) image
        show(colors[labels], "4b_blobs")
    return white


def features(image: np.ndarray) -> np.ndarray:
    """Whiteness image -> 18 numbers: [10 histogram bins, 4 x (number of blobs, log total area)]."""
    m = config.TEAR_MARGIN_PX
    inner = image[m:-m, m:-m]  # without the card edge, where white background can show

    # histogram: fraction of pixels per whiteness range; log1p because the fractions differ by orders of magnitude
    counts, _ = np.histogram(inner, bins=config.TEAR_HIST_EDGES)
    histogram = np.log1p(counts / inner.size * config.TEAR_HIST_SCALE)

    # blobs: a tear is a few big connected pieces of white, glare on the art is many small specks
    _labels, areas = find_blobs(inner)
    per_class = []
    for low_mm2, high_mm2 in config.TEAR_AREA_CLASSES_MM2:
        in_class = areas[_in_class(areas, low_mm2, high_mm2)]
        # log1p: the total area is 0 on most cards and thousands of px on a torn one
        per_class += [len(in_class), np.log1p(in_class.sum())]
    return np.concatenate([histogram, per_class]).astype(np.float32)


def find_blobs(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whiteness image -> (labels, areas).

    labels: same size as image, every pixel holds the number of its blob (0 = not white).
    areas: areas[i] = size of blob i in px.
    """
    mask = (image > config.TEAR_WHITE).astype(np.uint8)
    # connectivity 8: pixels that touch diagonally belong to the same blob (a ragged tear runs diagonally)
    _count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0  # number 0 is everything that is not white: not a blob, so it must not fall in a size class
    return labels, areas


def _in_class(areas: np.ndarray, low_mm2: float, high_mm2: float) -> np.ndarray:
    """True per blob whose area lies in [low, high) mm2."""
    px_per_mm2 = config.PX_PER_MM ** 2  # 10 px/mm -> 100 px per mm2
    return (areas >= low_mm2 * px_per_mm2) & (areas < high_mm2 * px_per_mm2)


def predict(model: object, features: np.ndarray) -> float:
    """Tear score 0-1: the chance of label 1 according to the model."""
    # predict_proba wants a 2D table (photos x features) and gives [chance of 0, chance of 1] per photo
    return float(model.predict_proba(features.reshape(1, -1))[0, 1])


def train(features: list[np.ndarray], labels: list[int]) -> object:
    """Fit a model on features of the training photos. Label 1 = tear, 0 = not."""
    # same classifier as the lines model: StandardScaler so the blob counts do not outweigh the histogram,
    # SVC with its default settings (RBF kernel, C=1): no tuning; probability=True gives a score 0-1
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0))
    model.fit(np.array(features), np.array(labels))
    return model
