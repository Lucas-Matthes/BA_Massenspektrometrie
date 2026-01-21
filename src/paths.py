from pathlib import Path

# Projekt-Root = BA/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = SRC_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"

MAIN_DATA = DATA_DIR / "main"
NOVEL_DATA = DATA_DIR / "novel"
MAIN_CSV = OUTPUT_DIR / "main.csv"
NOVEL_CSV = OUTPUT_DIR / "novel.csv"
VECTORS_CSV = MODELS_DIR / "vectors.csv"

RF_MODEL = MODELS_DIR / "rf_model.joblib"
DBSCAN_MODEL = MODELS_DIR / "dbscan_classifier.pkl"
COMPOUND_MODEL = MODELS_DIR / "compound_classifier.pkl"