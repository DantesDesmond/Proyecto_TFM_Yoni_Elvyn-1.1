from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "artifacts" / "beto_toxicity_binary"
LABELED_PATH = BASE_DIR / "data" / "processed" / "youtube_sample_labeled.csv"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labeled_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_texts(texts: List[str], tokenizer, model, batch_size: int = 16, max_length: int = 256) -> List[Dict]:
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        for text, pred, prob in zip(batch, preds.cpu().tolist(), probs.cpu().tolist()):
            results.append({
                "text": text,
                "pred_label": pred,
                "pred_class": "tóxico" if pred == 1 else "no_tóxico",
                "score_no_toxico": float(prob[0]),
                "score_toxico": float(prob[1]),
                "score_confianza": float(max(prob[0], prob[1])),
            })

    return results


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("YouTube validation confusion matrix")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_labeled_data(LABELED_PATH)
    tokenizer, model = load_model()

    preds = predict_texts(df["text"].tolist(), tokenizer=tokenizer, model=model)
    preds_df = pd.DataFrame(preds)

    merged_df = pd.concat([df.reset_index(drop=True), preds_df.drop(columns=["text"])], axis=1)
    merged_df["error_type"] = merged_df.apply(
        lambda r: "correcto" if r["label"] == r["pred_label"] else ("FN" if r["label"] == 1 else "FP"),
        axis=1,
    )

    y_true = merged_df["label"]
    y_pred = merged_df["pred_label"]

    metrics = {
        "created_at": datetime.now().isoformat(),
        "rows": int(len(merged_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }

    metrics_path = METRICS_DIR / "youtube_validation_metrics.json"
    preds_path = TABLES_DIR / "youtube_predictions_sample.csv"
    cm_path = FIGURES_DIR / "youtube_validation_confusion_matrix.png"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    merged_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
    save_confusion_matrix(y_true, y_pred, cm_path)

    print("=" * 70)
    print("VALIDACIÓN YOUTUBE COMPLETADA")
    print(f"Métricas -> {metrics_path}")
    print(f"Predicciones -> {preds_path}")
    print(f"Confusion matrix -> {cm_path}")
    print("=" * 70)
    print(f"Macro-F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
