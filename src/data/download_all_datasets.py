from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from datasets import load_dataset


BASE_DIR = Path(r"C:\Users\ELVYN\OneDrive\Desktop\TFM\TFM_Model")
DATA_DIR = BASE_DIR / "data" / "external"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"download_datasets_{RUN_TS}.log"


def log(message: str) -> None:
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_hf_split(dataset_name: str, split_name: str, output_dir: Path, file_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[HF] Descargando {dataset_name} | split={split_name}")

    ds = load_dataset(dataset_name, split=split_name)
    df = ds.to_pandas()

    csv_path = output_dir / f"{file_stem}.csv"
    parquet_path = output_dir / f"{file_stem}.parquet"
    meta_path = output_dir / f"{file_stem}_meta.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)

    meta = {
        "dataset_name": dataset_name,
        "split": split_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "downloaded_at": datetime.now().isoformat(),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"[OK] Guardado CSV: {csv_path}")
    log(f"[OK] Guardado Parquet: {parquet_path}")
    log(f"[OK] Filas: {len(df)} | Columnas: {len(df.columns)}")


def git_clone(repo_url: str, target_dir: Path) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        log(f"[SKIP] Ya existe contenido en: {target_dir}")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"[GIT] Clonando {repo_url} -> {target_dir}")

    subprocess.run(
        ["git", "clone", repo_url, str(target_dir)],
        check=True,
        shell=False,
    )

    log("[OK] Repo clonado correctamente")


def main() -> None:
    log("=" * 80)
    log("INICIO DE DESCARGA DE DATASETS")
    log(f"Proyecto: {BASE_DIR}")
    log(f"Timestamp: {RUN_TS}")
    log("=" * 80)

    save_hf_split(
        dataset_name="textdetox/multilingual_toxicity_dataset",
        split_name="es",
        output_dir=DATA_DIR / "hf_multilingual_toxicity_dataset_es",
        file_stem="multilingual_toxicity_dataset_es",
    )

    save_hf_split(
        dataset_name="textdetox/multilingual_toxicity_explained",
        split_name="es",
        output_dir=DATA_DIR / "hf_multilingual_toxicity_explained_es",
        file_stem="multilingual_toxicity_explained_es",
    )

    save_hf_split(
        dataset_name="textdetox/multilingual_paradetox",
        split_name="es",
        output_dir=DATA_DIR / "hf_multilingual_paradetox_es",
        file_stem="multilingual_paradetox_es",
    )

    git_clone(
        repo_url="https://github.com/microsoft/Clandestino.git",
        target_dir=DATA_DIR / "clandestino" / "repo",
    )

    git_clone(
        repo_url="https://github.com/alvaro-mazcu-herreros/DETOXIS_2021.git",
        target_dir=DATA_DIR / "detoxis" / "repo",
    )

    log("=" * 80)
    log("DESCARGA COMPLETADA")
    log("=" * 80)


if __name__ == "__main__":
    main()