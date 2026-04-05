from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "artifacts" / "beto_toxicity_binary"

_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.to(_device)
        _model.eval()

    return _tokenizer, _model


def predict_texts(texts: List[str], max_length: int = 256) -> List[Dict]:
    tokenizer, model = load_model()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    encodings = {k: v.to(_device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    results = []
    for text, pred, prob in zip(texts, preds.cpu().tolist(), probs.cpu().tolist()):
        toxic_prob = float(prob[1])
        no_toxic_prob = float(prob[0])

        results.append({
            "text": text,
            "pred_label": pred,
            "pred_class": "tóxico" if pred == 1 else "no_tóxico",
            "score_no_toxico": no_toxic_prob,
            "score_toxico": toxic_prob,
            "score_confianza": max(no_toxic_prob, toxic_prob),
        })

    return results