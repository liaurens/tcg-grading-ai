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

# Crop (step 3): Canny contour on white background (decision 2026-10-21).
# Blur and kernel size were tuned at DETECT_SCALE; they only hold at that scale.
DETECT_SCALE = 0.25        # find card on a 1/4 copy (1008x756): fast, and enough to locate the outline
CROP_BLUR_SIGMA = 2        # smooths paper texture and noise so Canny only fires on the card outline
CROP_CANNY_LOW = 40        # weak edge: kept only if connected to a strong edge (hysteresis)
CROP_CANNY_HIGH = 120      # strong edge: surely a border (dark card edge vs white background)
CROP_CLOSE_KERNEL_PX = 15  # ellipse that closes gaps in the outline up to ~15 px (~60 px full size)
CROP_MIN_FILL = 0.90       # mask area / rectangle area; our 40 photos: 0.955-0.991, below 0.90 it is no card shape
# Identify (step 4) and damage models (step 5): each model adds its own parameters
# here, under its own heading (e.g. "# Scratch model (step 5a)").
# OPEN: how damage_5.combine merges the scores + the threshold(s) goed/beschadigd
