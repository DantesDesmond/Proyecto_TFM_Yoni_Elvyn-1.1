from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, PROCESSED_DIR

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
RUN_NAME = "beto_toxicity_binary"
RANDOM_STATE = 42
MAX_LENGTH = 256
NUM_EPOCHS = 3
LR = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    required = {"text_clean", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path.name}: {missing}")
    df = df.dropna(subset=["text_clean", "label"]).copy()
    df["text_clean"] = df["text_clean"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def compute_metrics_dict(y_true, y_pred) -> dict:
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


def df_to_hf(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text_clean", "label"]].reset_index(drop=True))


def build_training_args(output_dir: Path, use_cuda: bool) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": LR,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_f1_macro",
        "greater_is_better": True,
        "seed": RANDOM_STATE,
        "save_total_limit": 2,
    }

    if "overwrite_output_dir" in params:
        kwargs["overwrite_output_dir"] = True

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"

    if "logging_strategy" in params:
        kwargs["logging_strategy"] = "steps"

    if "logging_steps" in params:
        kwargs["logging_steps"] = 50

    if "report_to" in params:
        kwargs["report_to"] = "none"

    if "fp16" in params and use_cuda:
        kwargs["fp16"] = True

    return TrainingArguments(**kwargs)


def build_trainer(model, training_args, train_ds, val_ds, data_collator, compute_metrics, tokenizer):
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_params = trainer_sig.parameters

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    # Compatibilidad entre versiones de transformers
    if "processing_class" in trainer_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    train_df = load_split(PROCESSED_DIR / "train.csv")
    val_df = load_split(PROCESSED_DIR / "val.csv")
    test_df = load_split(PROCESSED_DIR / "test.csv")
    ext_df = load_split(PROCESSED_DIR / "external_validation_detoxis.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text_clean"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = df_to_hf(train_df).map(tokenize_batch, batched=True)
    val_ds = df_to_hf(val_df).map(tokenize_batch, batched=True)
    test_ds = df_to_hf(test_df).map(tokenize_batch, batched=True)
    ext_ds = df_to_hf(ext_df).map(tokenize_batch, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "no_toxico", 1: "toxico"},
        label2id={"no_toxico": 0, "toxico": 1},
    )

    output_dir = ARTIFACTS_DIR / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    training_args = build_training_args(output_dir, use_cuda)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "eval_accuracy": accuracy_score(labels, preds),
            "eval_precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "eval_recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "eval_f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_ds=train_ds,
        val_ds=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    all_metrics = {
        "model_name": MODEL_NAME,
        "run_name": RUN_NAME,
        "created_at": datetime.now().isoformat(),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "external_rows": int(len(ext_df)),
        "device": "cuda" if use_cuda else "cpu",
    }

    evaluations = {
        "val": (val_df, val_ds),
        "test": (test_df, test_ds),
        "external_validation_detoxis": (ext_df, ext_ds),
    }

    for split_name, (df, ds) in evaluations.items():
        pred_output = trainer.predict(ds)
        y_true = df["label"].tolist()
        y_pred = np.argmax(pred_output.predictions, axis=-1)

        split_metrics = compute_metrics_dict(y_true, y_pred)
        all_metrics[split_name] = split_metrics

        cm_path = FIGURES_DIR / f"{RUN_NAME}_{split_name}_confusion_matrix.png"
        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title=f"{RUN_NAME} - {split_name}",
            out_path=cm_path,
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{RUN_NAME}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("TRANSFORMER ENTRENADO")
    print(f"Modelo guardado en: {output_dir}")
    print(f"Métricas guardadas en: {metrics_path}")
    print("=" * 70)
    print("Macro-F1")
    print(f"VAL  : {all_metrics['val']['f1_macro']:.4f}")
    print(f"TEST : {all_metrics['test']['f1_macro']:.4f}")
    print(f"EXT  : {all_metrics['external_validation_detoxis']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
