from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
TABLES_DIR = OUTPUTS_DIR / "tables"

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "es")
DEFAULT_MAX_COMMENTS = int(os.getenv("DEFAULT_MAX_COMMENTS", "500"))

for folder in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR, ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
