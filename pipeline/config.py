"""All paths and parameters. Each value: name + reason. OPEN = not decided yet."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"  # result.json + original per card
DEBUG_DIR = OUTPUT_DIR / "debug"      # step images (--debug)
MODELS_DIR = PROJECT_ROOT / "models"  # trained classifiers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # iPhone exports .jpeg

# Card size in mm (same for MTG and Pokémon)
CARD_WIDTH_MM = 63
CARD_HEIGHT_MM = 88
PX_PER_MM = 10  # provisional: 630x880 px, enough to see scratches

# Crop (step 2): Canny contour on white background (decision 2026-10-21).
# Blur and kernel size were tuned at DETECT_SCALE; they only hold at that scale.
DETECT_SCALE = 0.25        # find card on a 1/4 copy (1008x756): fast, and enough to locate the outline
CROP_BLUR_SIGMA = 2        # smooths paper texture and noise so Canny only fires on the card outline
CROP_CANNY_LOW = 40        # weak edge: kept only if connected to a strong edge (hysteresis)
CROP_CANNY_HIGH = 120      # strong edge: surely a border (dark card edge vs white background)
CROP_CLOSE_KERNEL_PX = 15  # ellipse that closes gaps in the outline up to ~15 px (~60 px full size)
CROP_MIN_FILL = 0.90       # mask area / rectangle area; our 40 photos: 0.955-0.991, below 0.90 it is no card shape
# Identify (step 3) and damage models (step 4): each model adds its own parameters
# here, under its own heading (e.g. "# Line damage model (step 4a)").
# How damage_4.combine merges the scores: see "Combine (step 4c)" below.

# Line damage model (step 4a): scratches + creases (decisions 2026-11-05)
LINES_HIGHPASS_SIGMA = 5  # px (0.5 mm at 10 px/mm): blur wipes out lines up to ~1 mm wide, so card - blur keeps them; big areas fall away (experiment 2026-11-05)
LINES_DEBUG_GAIN = 4      # only for the debug image: high-pass values are small, x4 makes them visible (same for every card)
# Features: standard OpenCV/NumPy functions on the high-pass image (decision 2026-11-10, simple standard features + SVM)
LINES_MARGIN_PX = 20        # skip the outer 2 mm: card edge and any background the crop left (up to ~15 px)
LINES_HIST_EDGES = [1, 5, 10, 15, 20, 25, 30, 40, 50, 70, 100, 256]  # high-pass value bins (0 = not lighter, left out): 11 histogram features
LINES_HIST_SCALE = 10000    # histogram fractions span 1e-4 .. 0.3; log1p(fraction * 10000) makes the small ones visible next to the big ones
LINES_BRIGHT = 35           # high-pass value above which a pixel counts as part of a line (35: fewer good cards rejected than 25)
LINES_HOUGH_VOTES = 40      # HoughLinesP: at least 40 bright pixels must lie on a line before it counts
LINES_MIN_LENGTH_PX = 60    # 6 mm: shortest line Hough returns; the length classes below split short from long
LINES_MAX_GAP_PX = 10       # 1 mm: a scratch with small interruptions still counts as one line
LINES_MIN_ANGLE_DEG = 3     # only diagonal lines count: the printed frame is within ~1 deg of horizontal/vertical after cropping
LINES_LENGTH_CLASSES_MM = [(6, 15), (15, 30), (30, 60), (60, 1000)]  # per class: how many lines and their total length (8 features)
LINES_MODEL_FILE = MODELS_DIR / "lines.joblib"  # joblib: the save format scikit-learn recommends

# Combine (step 4c): "beschadigd" as soon as one built damage model scores >= its own threshold (decision 2026-11-07).
# Models that are not built yet (score None) are skipped; result.json lists which models were checked.
DAMAGE_THRESHOLDS = {
    "lines": 0.5,  # CV: 0.5 -> good cards rejected 4%, recall 70%; 0.4/0.3 reject more good cards, precision drops (2026-11-07)
    "tear": 0.5,   # placeholder until tear.py is built: choose with that model's own validation
}

# Report (step 5)
OVERVIEW_FILE = RESULTS_DIR / "overzicht.csv"  # one row per card of the last run; opens in Excel
EVALUATION_FILE = RESULTS_DIR / "evaluatie.txt"  # scores per split vs file-name labels (only for our dataset)

# Training
DATASET_DIR = DATA_DIR / "final_dataset"  # decision 2026-11-07; contains train_dataset/, validate_dataset/, test_dataset/; override with --data
