from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
MODELS = ROOT / "models"


def ensure_directories() -> None:
    for path in (RAW, INTERIM, PROCESSED, FIGURES, METRICS, MODELS):
        path.mkdir(parents=True, exist_ok=True)

