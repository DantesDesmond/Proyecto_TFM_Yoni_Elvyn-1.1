from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory

from src.config import RAW_DIR, YOUTUBE_API_KEY, DEFAULT_MAX_COMMENTS
from src.utils.logger import get_logger

DetectorFactory.seed = 0
logger = get_logger(__name__)


def safe_detect_language(text: str) -> str:
    try:
        return detect(text) if isinstance(text, str) and text.strip() else "unknown"
    except Exception:
        return "unknown"


def fetch_video_comments(video_id: str, max_comments: int = DEFAULT_MAX_COMMENTS) -> pd.DataFrame:
    if not YOUTUBE_API_KEY:
        raise ValueError("No se encontró YOUTUBE_API_KEY en el archivo .env")

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    rows: list[dict[str, Any]] = []
    next_page_token = None

    while len(rows) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(rows)),
            textFormat="plainText",
            pageToken=next_page_token,
            order="relevance",
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "")
            rows.append(
                {
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "video_id": video_id,
                    "author": snippet.get("authorDisplayName", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "like_count": snippet.get("likeCount", 0),
                    "text": text,
                    "language_detected": safe_detect_language(text),
                    "source": "youtube",
                }
            )

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["comment_id"])
    logger.info("Comentarios descargados: %s", len(df))
    return df


def save_raw_comments(df: pd.DataFrame, filename: str) -> Path:
    output_path = RAW_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Archivo guardado en %s", output_path)
    return output_path


if __name__ == "__main__":
    sample_video_id = "dQw4w9WgXcQ"
    data = fetch_video_comments(sample_video_id, max_comments=100)
    save_raw_comments(data, f"youtube_comments_{sample_video_id}.csv")
