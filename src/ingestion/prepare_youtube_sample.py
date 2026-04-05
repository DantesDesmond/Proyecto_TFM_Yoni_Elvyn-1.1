from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "youtube_comments" / "youtube_comments_consolidado.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "youtube_sample_labeled.csv"

SAMPLE_SIZE = 350
MIN_TEXT_LEN = 12

def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo consolidado: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # limpieza mínima para muestreo útil
    df = df.dropna(subset=["text_original"]).copy()
    df["text_original"] = df["text_original"].astype(str).str.strip()
    df = df[df["text_original"].str.len() >= MIN_TEXT_LEN].copy()
    df = df.drop_duplicates(subset=["text_original"]).reset_index(drop=True)

    # muestra aleatoria reproducible
    sample_n = min(SAMPLE_SIZE, len(df))
    sample_df = df.sample(n=sample_n, random_state=42).copy()

    # estructura final para anotación
    labeled_df = pd.DataFrame({
        "video_id": sample_df["video_id"],
        "comment_id": sample_df["comment_id"],
        "text": sample_df["text_original"],
        "label": "",
        "notes": "",
    })

    labeled_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("MUESTRA DE YOUTUBE CREADA")
    print(f"Archivo -> {OUT_PATH}")
    print(f"Filas -> {len(labeled_df)}")
    print("=" * 70)

if __name__ == "__main__":
    main()