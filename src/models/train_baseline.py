from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, PROCESSED_DIR

MODEL_NAME = "baseline_tfidf_logreg"
RANDOM_STATE = 42


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    required = {"text_clean", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path.name}: {missing}")
    return df


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"
    ext_path = PROCESSED_DIR / "external_validation_detoxis.csv"

    train_df = load_split(train_path)
    val_df = load_split(val_path)
    test_df = load_split(test_path)
    ext_df = load_split(ext_path)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )

    model.fit(train_df["text_clean"], train_df["label"])

    evaluations = {
        "val": val_df,
        "test": test_df,
        "external_validation_detoxis": ext_df,
    }

    all_metrics = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now().isoformat(),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "external_rows": int(len(ext_df)),
    }

    for split_name, df in evaluations.items():
        y_true = df["label"]
        y_pred = model.predict(df["text_clean"])
        metrics = compute_metrics(y_true, y_pred)
        all_metrics[split_name] = metrics

        cm_path = FIGURES_DIR / f"{MODEL_NAME}_{split_name}_confusion_matrix.png"
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title=f"{MODEL_NAME} - {split_name}",
            out_path=cm_path,
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / f"{MODEL_NAME}.joblib"
    metrics_path = METRICS_DIR / f"{MODEL_NAME}_metrics.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("BASELINE ENTRENADO")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print("=" * 70)
    print("Macro-F1")
    print(f"VAL  : {all_metrics['val']['f1_macro']:.4f}")
    print(f"TEST : {all_metrics['test']['f1_macro']:.4f}")
    print(f"EXT  : {all_metrics['external_validation_detoxis']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
