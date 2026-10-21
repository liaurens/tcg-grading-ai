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

# OPEN: crop parameters (depends on background)
# OPEN: Canny + high-pass parameters
# OPEN: damage threshold goed/beschadigd
