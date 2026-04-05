from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "youtube_comments"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config_youtube_local.json"

API_URL_THREADS = "https://www.googleapis.com/youtube/v3/commentThreads"
DEFAULT_VIDEOS_FILE = RAW_DIR / "videos_objetivo.csv"


def get_api_key() -> str:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        return api_key.strip()

    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        api_key = str(data.get("youtube_api_key", "")).strip()
        if api_key:
            return api_key

    raise RuntimeError(
        "No se encontró la API key. Define YOUTUBE_API_KEY o crea "
        f"{CONFIG_FILE} con el campo 'youtube_api_key'."
    )


def load_videos(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")
    df = pd.read_csv(csv_path)
    if "video_id" not in df.columns:
        raise ValueError("El archivo de videos debe contener la columna 'video_id'.")
    df = df.dropna(subset=["video_id"]).copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df = df[df["video_id"] != ""].copy()
    return df


def parse_top_level_item(item: Dict, video_meta: Dict) -> Dict:
    snippet = item["snippet"]["topLevelComment"]["snippet"]
    comment_id = item["snippet"]["topLevelComment"]["id"]
    return {
        "video_id": video_meta.get("video_id", ""),
        "canal": video_meta.get("canal", ""),
        "tema": video_meta.get("tema", ""),
        "idioma_estimado": video_meta.get("idioma_estimado", ""),
        "prioridad": video_meta.get("prioridad", ""),
        "comment_id": comment_id,
        "parent_id": "",
        "is_reply": 0,
        "author_display_name": snippet.get("authorDisplayName", ""),
        "text_original": snippet.get("textDisplay", ""),
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
        "like_count": snippet.get("likeCount", 0),
        "reply_count": item["snippet"].get("totalReplyCount", 0),
        "source_platform": "youtube",
        "downloaded_at": datetime.now().isoformat(),
    }


def parse_reply_item(reply: Dict, video_meta: Dict, parent_id: str) -> Dict:
    snippet = reply["snippet"]
    return {
        "video_id": video_meta.get("video_id", ""),
        "canal": video_meta.get("canal", ""),
        "tema": video_meta.get("tema", ""),
        "idioma_estimado": video_meta.get("idioma_estimado", ""),
        "prioridad": video_meta.get("prioridad", ""),
        "comment_id": reply["id"],
        "parent_id": parent_id,
        "is_reply": 1,
        "author_display_name": snippet.get("authorDisplayName", ""),
        "text_original": snippet.get("textDisplay", ""),
        "published_at": snippet.get("publishedAt", ""),
        "updated_at": snippet.get("updatedAt", ""),
        "like_count": snippet.get("likeCount", 0),
        "reply_count": 0,
        "source_platform": "youtube",
        "downloaded_at": datetime.now().isoformat(),
    }


def fetch_comments_for_video(api_key: str, video_meta: Dict, max_threads: int = 500) -> List[Dict]:
    rows: List[Dict] = []
    next_page_token: Optional[str] = None
    fetched_threads = 0

    while True:
        params = {
            "part": "snippet,replies",
            "videoId": video_meta["video_id"],
            "maxResults": 100,
            "order": "time",
            "textFormat": "plainText",
            "key": api_key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(API_URL_THREADS, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()

        items = payload.get("items", [])
        if not items:
            break

        for item in items:
            top_row = parse_top_level_item(item, video_meta)
            rows.append(top_row)
            fetched_threads += 1

            replies = item.get("replies", {}).get("comments", [])
            for reply in replies:
                rows.append(parse_reply_item(reply, video_meta, parent_id=top_row["comment_id"]))

            if fetched_threads >= max_threads:
                break

        if fetched_threads >= max_threads:
            break

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    return rows


def save_batch(rows: List[Dict], batch_name: str) -> Path:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No se recuperaron comentarios para guardar.")
    df = df.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)

    out_path = RAW_DIR / batch_name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def save_consolidated(raw_paths: Iterable[Path]) -> Path:
    frames = []
    for path in raw_paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("No hay archivos raw para consolidar.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)

    out_path = RAW_DIR / "youtube_comments_consolidado.csv"
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    api_key = get_api_key()
    videos_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEOS_FILE
    videos_df = load_videos(videos_file)

    if videos_df.empty:
        raise ValueError("El archivo de videos no contiene video_id válidos.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths: List[Path] = []
    total_rows = 0

    for _, row in videos_df.iterrows():
        video_meta = row.to_dict()
        print(f"Descargando video_id={video_meta['video_id']} ...")
        rows = fetch_comments_for_video(api_key=api_key, video_meta=video_meta, max_threads=500)
        if not rows:
            print(f"Sin comentarios recuperados para {video_meta['video_id']}")
            continue

        file_name = f"youtube_comments_{video_meta['video_id']}_{timestamp}.csv"
        out_path = save_batch(rows, file_name)
        saved_paths.append(out_path)
        total_rows += len(rows)
        print(f"OK -> {out_path} | filas recuperadas: {len(rows)}")

    if saved_paths:
        consolidated = save_consolidated(saved_paths)
        print(f"Consolidado -> {consolidated}")

    print("=" * 70)
    print(f"Videos procesados: {len(saved_paths)}")
    print(f"Filas recuperadas (antes de consolidado global): {total_rows}")
    print("=" * 70)


if __name__ == "__main__":
    main()
