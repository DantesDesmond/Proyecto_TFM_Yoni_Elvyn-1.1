from __future__ import annotations

import re
import pandas as pd
from src.config import INTERIM_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_comments(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["text_clean"] = data["text"].fillna("").apply(clean_text)
    data = data[data["text_clean"].str.len() > 2].copy()
    logger.info("Comentarios después de limpieza: %s", len(data))
    return data


def save_interim(df: pd.DataFrame, filename: str) -> None:
    path = INTERIM_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Datos intermedios guardados en %s", path)
