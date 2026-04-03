from __future__ import annotations

import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.config import ARTIFACTS_DIR, METRICS_DIR, PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_baseline(model_file: str = "baseline_logreg.joblib", input_file: str = "labeled_comments.csv") -> None:
    model = joblib.load(ARTIFACTS_DIR / model_file)
    df = pd.read_csv(PROCESSED_DIR / input_file)
    preds = model.predict(df["text_clean"])

    macro_f1 = f1_score(df["label"], preds, average="macro")
    report = classification_report(df["label"], preds, output_dict=True)
    matrix = confusion_matrix(df["label"], preds).tolist()

    with open(METRICS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"macro_f1": macro_f1, "report": report, "confusion_matrix": matrix}, f, ensure_ascii=False, indent=2)

    logger.info("Evaluación completada. Macro-F1: %.4f", macro_f1)


if __name__ == "__main__":
    evaluate_baseline()
