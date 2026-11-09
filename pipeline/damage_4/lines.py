"""Step 4a: line damage: scratches (krassen) and creases (vouwrandjes) in one model
(decision 2026-11-05; rebuilt simpler on 2026-11-10).

Same recipe as the color model (step 3): standard OpenCV features -> StandardScaler + SVC (scikit-learn).
Idea: under the oblique light a scratch or crease catches the light and is LIGHTER than its surroundings.
Preprocessing: gray + high-pass filter (card minus its own blur, only the lighter pixels).
Features (19 numbers, all standard functions, no hand-made reference):
  - histogram of the high-pass values (11): how much of the card is how much lighter than its surroundings
  - Hough lines (8): per length class (6-15, 15-30, 30-60, >60 mm) how many diagonal lines and their total length
The classifier learns from the labelled photos which of these numbers point at damage.
Known weak spot: the printed card design (frames, text, symbol) also gives bright lines.
"""
import cv2
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline import config
from pipeline.io_utils_1 import ShowFn, no_show


def load_model() -> object:
    """Trained line damage model from models/."""
    if not config.LINES_MODEL_FILE.exists():
        raise FileNotFoundError(f"{config.LINES_MODEL_FILE} not found: run python -m training.train lines")
    return joblib.load(config.LINES_MODEL_FILE)


def save_model(model: object) -> None:
    """Save the trained model to models/ (same format load_model reads)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.LINES_MODEL_FILE)


def preprocess(card: np.ndarray, show: ShowFn = no_show) -> np.ndarray:
    """Color card (880, 630, 3) BGR -> high-pass image (880, 630) uint8.

    Each pixel: how much lighter it is than its surroundings (0 = not lighter).
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    show(gray, "4a_gray")

    # ksize (0, 0): OpenCV picks the kernel size from sigma
    blurred = cv2.GaussianBlur(gray, (0, 0), config.LINES_HIGHPASS_SIGMA)
    # int16: the difference can be negative; in uint8 -5 would wrap around to 251
    difference = gray.astype(np.int16) - blurred.astype(np.int16)
    # keep only "lighter than surroundings": scratches and creases catch the oblique light
    highpass = np.clip(difference, 0, 255).astype(np.uint8)

    visible = np.clip(highpass.astype(np.int16) * config.LINES_DEBUG_GAIN, 0, 255).astype(np.uint8)
    show(visible, "4a_highpass")

    if show is not no_show:  # only for --debug: show what features() will feed to Hough, and what Hough finds
        m = config.LINES_MARGIN_PX
        inner = highpass[m:-m, m:-m]
        bright = (inner > config.LINES_BRIGHT).astype(np.uint8) * 255
        show(bright, "4a_bright")
        found = cv2.cvtColor(bright, cv2.COLOR_GRAY2BGR)
        for segment in find_lines(inner):
            # yellow = diagonal (counts as feature), blue = along the frame (ignored)
            diagonal = _angle_from_axis(segment) >= config.LINES_MIN_ANGLE_DEG
            x1, y1, x2, y2 = segment
            cv2.line(found, (x1, y1), (x2, y2), (0, 255, 255) if diagonal else (255, 120, 0), 3)
        show(found, "4a_lines")
    return highpass


def features(image: np.ndarray) -> np.ndarray:
    """High-pass image -> 19 numbers: [11 histogram bins, 4 x (number of lines, log total length)]."""
    m = config.LINES_MARGIN_PX
    inner = image[m:-m, m:-m]  # without the card edge

    # histogram: fraction of pixels per high-pass value range; log1p because the fractions differ by orders of magnitude
    counts, _ = np.histogram(inner, bins=config.LINES_HIST_EDGES)
    histogram = np.log1p(counts / inner.size * config.LINES_HIST_SCALE)

    # Hough: straight line pieces in the pixels that are clearly lighter than their surroundings
    lengths = np.array([_line_length(s) for s in find_lines(inner)
                        if _angle_from_axis(s) >= config.LINES_MIN_ANGLE_DEG])
    per_class = []
    for low_mm, high_mm in config.LINES_LENGTH_CLASSES_MM:
        in_class = lengths[(lengths >= low_mm * config.PX_PER_MM) & (lengths < high_mm * config.PX_PER_MM)]
        # log1p: the total length is 0 on most cards and thousands of px on a scratched one
        per_class += [len(in_class), np.log1p(in_class.sum())]
    return np.concatenate([histogram, per_class]).astype(np.float32)


def find_lines(image: np.ndarray) -> np.ndarray:
    """High-pass image -> line segments, shape (n, 4): x1, y1, x2, y2."""
    bright = (image > config.LINES_BRIGHT).astype(np.uint8) * 255
    # theta 1 degree, rho 1 px: the resolution at which HoughLinesP tries line directions/positions
    segments = cv2.HoughLinesP(bright, rho=1, theta=np.pi / 180, threshold=config.LINES_HOUGH_VOTES,
                               minLineLength=config.LINES_MIN_LENGTH_PX, maxLineGap=config.LINES_MAX_GAP_PX)
    # no lines -> None; OpenCV 5 returns (n, 4), older versions (n, 1, 4): reshape handles both
    return np.empty((0, 4)) if segments is None else segments.reshape(-1, 4)


def _line_length(segment: np.ndarray) -> float:
    x1, y1, x2, y2 = segment
    return float(np.hypot(x2 - x1, y2 - y1))


def _angle_from_axis(segment: np.ndarray) -> float:
    """0 = horizontal or vertical, 45 = exactly diagonal."""
    x1, y1, x2, y2 = segment
    angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))  # 0..90
    return min(angle, 90 - angle)


def predict(model: object, features: np.ndarray) -> float:
    """Line damage score 0-1: the chance of label 1 according to the model."""
    # predict_proba wants a 2D table (photos x features) and gives [chance of 0, chance of 1] per photo
    return float(model.predict_proba(features.reshape(1, -1))[0, 1])


def train(features: list[np.ndarray], labels: list[int]) -> object:
    """Fit a model on features of the training photos. Label 1 = scratch or crease, 0 = not."""
    # StandardScaler: every feature to mean 0, spread 1, so a feature with big numbers
    # (total length) does not outweigh one with small numbers (histogram), same as the color model
    # SVC with its default settings (RBF kernel, C=1): no tuning; probability=True gives a score 0-1
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0))
    model.fit(np.array(features), np.array(labels))
    return model
