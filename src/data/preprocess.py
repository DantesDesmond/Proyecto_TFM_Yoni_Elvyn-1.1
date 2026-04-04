from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import EXTERNAL_DIR, INTERIM_DIR, PROCESSED_DIR

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

CLANDESTINO_TOXIC_THRESHOLD = 3.0
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CLANDESTINO_HARM_QUESTION = "Question 1: In your opinion, will this text be harmful to anyone?"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_project_path(raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) >= 2 and parts[0] == "data" and parts[1] == "external":
        return EXTERNAL_DIR / Path(*parts[2:])
    if len(parts) >= 1 and parts[0] == "data":
        return EXTERNAL_DIR.parent / Path(*parts[1:])
    return Path(raw_path)


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return pd.json_normalize(data)
            return pd.DataFrame(data)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def normalize_binary_label(series: pd.Series) -> pd.Series:
    mapping = {
        "0": 0, "1": 1,
        "false": 0, "true": 1,
        "no": 0, "yes": 1,
        "non-toxic": 0, "non toxic": 0, "not toxic": 0,
        "toxic": 1,
        "no_toxico": 0, "no tóxico": 0, "notoxico": 0,
        "toxico": 1, "tóxico": 1,
    }

    def _map(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None

        text = str(value).strip().lower()
        if text in mapping:
            return mapping[text]

        try:
            num = float(text)
            if num in (0.0, 1.0):
                return int(num)
        except Exception:
            return None

        return None

    return series.apply(_map)


def normalize_toxicity_level(series: pd.Series) -> pd.Series:
    def _map(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None

        text = str(value).strip().lower()
        if text in {"low", "none", "non-toxic", "non toxic", "no_toxico", "no tóxico"}:
            return 0
        if text in {"medium", "high", "toxic", "toxicity", "toxico", "tóxico"}:
            return 1
        return None

    return series.apply(_map)


def extract_clandestino_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    # Esperado: lista de dicts por anotador
    if isinstance(value, list):
        scores = []
        for item in value:
            if not isinstance(item, dict):
                continue
            raw = item.get(CLANDESTINO_HARM_QUESTION)
            if raw is None:
                continue
            try:
                score = float(raw)
            except Exception:
                continue
            if 0 <= score <= 5:
                scores.append(score)

        if not scores:
            return None
        return sum(scores) / len(scores)

    # A veces puede venir serializado como string JSON
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                parsed = json.loads(text)
                return extract_clandestino_score(parsed)
            except Exception:
                return None

    return None


def normalize_clandestino_score(series: pd.Series, threshold: float = CLANDESTINO_TOXIC_THRESHOLD) -> pd.Series:
    def _map(value) -> Optional[int]:
        score = extract_clandestino_score(value)
        if score is None:
            return None
        return int(score >= threshold)

    return series.apply(_map)


def prepare_dataset(row: pd.Series) -> Optional[pd.DataFrame]:
    dataset_name = row["dataset"]
    use_for = row["use_for"]
    file_path = resolve_project_path(row["file_path"])

    logger.info("Procesando dataset=%s | use_for=%s | path=%s", dataset_name, use_for, file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo/ruta: {file_path}")

    df = read_dataset(file_path)

    data = pd.DataFrame()
    text_column = row["text_column"]
    label_column = row["label_column"]

    if dataset_name == "clandestino":
        if "Sentence" not in df.columns or "Annotators" not in df.columns:
            raise KeyError(f"CLANDESTINO no trae las columnas esperadas. Columnas reales: {df.columns.tolist()}")
        data["text"] = df["Sentence"].astype(str)
        data["dataset_source"] = dataset_name
        data["use_for"] = use_for
        data["label"] = normalize_clandestino_score(df["Annotators"])
    else:
        if text_column not in df.columns:
            raise KeyError(
                f"La columna de texto '{text_column}' no existe en {dataset_name}. "
                f"Columnas reales: {df.columns.tolist()}"
            )

        data["text"] = df[text_column].astype(str)
        data["dataset_source"] = dataset_name
        data["use_for"] = use_for

        if use_for in {"train", "train_optional", "validation", "final_validation"}:
            if dataset_name == "multilingual_toxicity_explained_es":
                data["label"] = normalize_toxicity_level(df[label_column])
            else:
                if label_column not in df.columns:
                    raise KeyError(
                        f"La columna de etiqueta '{label_column}' no existe en {dataset_name}. "
                        f"Columnas reales: {df.columns.tolist()}"
                    )
                data["label"] = normalize_binary_label(df[label_column])
        else:
            data["label"] = None

    data["text_clean"] = data["text"].apply(clean_text)
    data = data[data["text_clean"].str.len() > 2].copy()
    data = data.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    if use_for in {"train", "train_optional", "validation", "final_validation"}:
        before = len(data)
        data = data.dropna(subset=["label"]).copy()

        if data.empty:
            if use_for == "train_optional":
                logger.warning("Se omite %s porque no se pudieron extraer etiquetas válidas.", dataset_name)
                return None
            raise ValueError(f"{dataset_name} quedó sin etiquetas válidas tras la normalización.")

        data["label"] = data["label"].astype(int)
        logger.info("Etiquetas válidas en %s: %s/%s", dataset_name, len(data), before)

    logger.info("Filas finales en %s: %s", dataset_name, len(data))
    return data


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Guardado: %s", path)


def main() -> None:
    mapping_path = EXTERNAL_DIR / "datasets_mapeo_final.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de mapeo: {mapping_path}")

    mapping = pd.read_csv(mapping_path)
    prepared: dict[str, pd.DataFrame] = {}

    for _, row in mapping.iterrows():
        if row["dataset"] == "youtube_own_sample":
            logger.info("Se omite por ahora youtube_own_sample porque aún no existe.")
            continue

        dataset_df = prepare_dataset(row)
        if dataset_df is not None:
            prepared[row["dataset"]] = dataset_df

    for name, df in prepared.items():
        save_dataframe(df, INTERIM_DIR / f"{name}_normalized.csv")

    train_parts = []
    for _, df in prepared.items():
        use_for = df["use_for"].iloc[0]
        if use_for in {"train", "train_optional"}:
            train_parts.append(df[["text", "text_clean", "label", "dataset_source"]].copy())

    if not train_parts:
        raise ValueError("No se encontraron datasets de entrenamiento válidos para construir el dataset maestro.")

    master_train = pd.concat(train_parts, ignore_index=True)
    master_train = master_train.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    external_validation_parts = []
    for _, df in prepared.items():
        use_for = df["use_for"].iloc[0]
        if use_for == "validation":
            external_validation_parts.append(df[["text", "text_clean", "label", "dataset_source"]].copy())

    if external_validation_parts:
        external_validation = pd.concat(external_validation_parts, ignore_index=True)
        external_validation = external_validation.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    else:
        external_validation = pd.DataFrame(columns=["text", "text_clean", "label", "dataset_source"])

    if master_train["label"].nunique() < 2:
        raise ValueError("El dataset maestro de entrenamiento no contiene al menos dos clases.")

    train_df, temp_df = train_test_split(
        master_train,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=master_train["label"],
    )

    relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"],
    )

    save_dataframe(master_train, PROCESSED_DIR / "master_train_full.csv")
    save_dataframe(train_df.reset_index(drop=True), PROCESSED_DIR / "train.csv")
    save_dataframe(val_df.reset_index(drop=True), PROCESSED_DIR / "val.csv")
    save_dataframe(test_df.reset_index(drop=True), PROCESSED_DIR / "test.csv")
    save_dataframe(external_validation, PROCESSED_DIR / "external_validation_detoxis.csv")

    summary = pd.DataFrame([
        {"file": "master_train_full.csv", "rows": len(master_train), "label_0": int((master_train["label"] == 0).sum()), "label_1": int((master_train["label"] == 1).sum())},
        {"file": "train.csv", "rows": len(train_df), "label_0": int((train_df["label"] == 0).sum()), "label_1": int((train_df["label"] == 1).sum())},
        {"file": "val.csv", "rows": len(val_df), "label_0": int((val_df["label"] == 0).sum()), "label_1": int((val_df["label"] == 1).sum())},
        {"file": "test.csv", "rows": len(test_df), "label_0": int((test_df["label"] == 0).sum()), "label_1": int((test_df["label"] == 1).sum())},
        {"file": "external_validation_detoxis.csv", "rows": len(external_validation), "label_0": int((external_validation["label"] == 0).sum()) if not external_validation.empty else 0, "label_1": int((external_validation["label"] == 1).sum()) if not external_validation.empty else 0},
    ])
    save_dataframe(summary, PROCESSED_DIR / "dataset_summary.csv")

    logger.info("Preprocesamiento completado correctamente.")


if __name__ == "__main__":
    main()
