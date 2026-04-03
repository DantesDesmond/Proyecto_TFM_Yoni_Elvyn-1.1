from __future__ import annotations

import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, ARTIFACTS_DIR, METRICS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_baseline(input_file: str = "labeled_comments.csv") -> None:
    data = pd.read_csv(PROCESSED_DIR / input_file)
    X_train, X_test, y_train, y_test = train_test_split(
        data["text_clean"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    macro_f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, output_dict=True)
    matrix = confusion_matrix(y_test, preds).tolist()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACTS_DIR / "baseline_logreg.joblib")
    with open(METRICS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"macro_f1": macro_f1, "report": report, "confusion_matrix": matrix}, f, ensure_ascii=False, indent=2)

    logger.info("Baseline entrenado. Macro-F1: %.4f", macro_f1)


if __name__ == "__main__":
    train_baseline()
