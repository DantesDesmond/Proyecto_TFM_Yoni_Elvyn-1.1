from pathlib import Path
import pandas as pd

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data" / "external"
OUT_FILE = DATA_DIR / "datasets_resumen.csv"

rows = []

def safe_read_table(path: Path):
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=5)
            full_rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
            return df, max(full_rows, 0)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            return df.head(5), len(df)
    except Exception as e:
        return str(e), None
    return None, None

for path in DATA_DIR.rglob("*"):
    if path.is_file() and path.suffix.lower() in [".csv", ".parquet"]:
        obj, total_rows = safe_read_table(path)

        if isinstance(obj, pd.DataFrame):
            rows.append({
                "dataset_folder": path.parent.name,
                "file_name": path.name,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "rows_estimated": total_rows,
                "n_columns": len(obj.columns),
                "columns": " | ".join(obj.columns.astype(str).tolist())
            })
        else:
            rows.append({
                "dataset_folder": path.parent.name,
                "file_name": path.name,
                "file_path": str(path),
                "extension": path.suffix.lower(),
                "rows_estimated": "",
                "n_columns": "",
                "columns": f"ERROR: {obj}"
            })

summary = pd.DataFrame(rows)
summary = summary.sort_values(["dataset_folder", "file_name"]).reset_index(drop=True)
summary.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

print(f"Resumen guardado en: {OUT_FILE}")
print(summary)