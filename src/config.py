from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

CLASSES = ["Negative", "Positive"]  # a posição na lista é o rótulo: 0 = sem fenda, 1 = com fenda
SEED = 42

SPLITS_CSV = ROOT / "data" / "splits.csv"

MODELS_DIR = ROOT / "models"
