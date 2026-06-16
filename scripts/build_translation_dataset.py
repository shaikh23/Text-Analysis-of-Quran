from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from quran_analysis.config import settings


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


@dataclass(frozen=True)
class ChapterMeta:
    chapter_id: int
    revelation_place: str
    revelation_order: int
    chapter_name_simple: str
    chapter_name_complex: str
    chapter_name_arabic: str
    chapter_name_translated: str
    chapter_verses_count: int


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.api_key:
        headers["x-api-key"] = settings.api_key
    return headers


def fetch_chapters() -> list[dict]:
    url = f"{settings.api_base_url}/chapters"
    response = requests.get(url, headers=_headers(), timeout=30)
    response.raise_for_status()
    return response.json().get("chapters", [])


def build_chapter_lookup(chapters: list[dict]) -> dict[int, ChapterMeta]:
    lookup: dict[int, ChapterMeta] = {}
    for chapter in chapters:
        lookup[chapter["id"]] = ChapterMeta(
            chapter_id=chapter["id"],
            revelation_place=chapter["revelation_place"],
            revelation_order=chapter["revelation_order"],
            chapter_name_simple=chapter["name_simple"],
            chapter_name_complex=chapter["name_complex"],
            chapter_name_arabic=chapter["name_arabic"],
            chapter_name_translated=chapter.get("translated_name", {}).get("name", ""),
            chapter_verses_count=chapter["verses_count"],
        )
    return lookup


def fetch_chapter_verses(chapter_id: int, translation_id: int) -> list[dict]:
    verses: list[dict] = []
    page = 1

    while True:
        params = {
            "language": "en",
            "words": "false",
            "translations": translation_id,
            "per_page": 50,
            "page": page,
        }
        url = f"{settings.api_base_url}/verses/by_chapter/{chapter_id}"
        response = requests.get(url, params=params, headers=_headers(), timeout=30)
        response.raise_for_status()

        payload = response.json()
        verses.extend(payload.get("verses", []))

        next_page = payload.get("pagination", {}).get("next_page")
        if not next_page:
            break
        page = int(next_page)

    return verses


def normalize_verse_row(verse: dict, chapter_meta: ChapterMeta, translation_id: int) -> dict:
    translations = verse.get("translations", [])
    translation = translations[0] if translations else {}

    return {
        "verse_id": verse.get("id"),
        "verse_key": verse.get("verse_key"),
        "chapter_id": chapter_meta.chapter_id,
        "verse_number": verse.get("verse_number"),
        "juz_number": verse.get("juz_number"),
        "hizb_number": verse.get("hizb_number"),
        "rub_el_hizb_number": verse.get("rub_el_hizb_number"),
        "ruku_number": verse.get("ruku_number"),
        "manzil_number": verse.get("manzil_number"),
        "page_number": verse.get("page_number"),
        "sajdah_number": verse.get("sajdah_number"),
        "revelation_place": chapter_meta.revelation_place,
        "revelation_order": chapter_meta.revelation_order,
        "chapter_name_simple": chapter_meta.chapter_name_simple,
        "chapter_name_complex": chapter_meta.chapter_name_complex,
        "chapter_name_arabic": chapter_meta.chapter_name_arabic,
        "chapter_name_translated": chapter_meta.chapter_name_translated,
        "chapter_verses_count": chapter_meta.chapter_verses_count,
        "translation_id": translation_id,
        "translation_resource_id": translation.get("resource_id"),
        "translation_text": translation.get("text", ""),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    translation_id = settings.translation_id
    chapters = fetch_chapters()
    chapter_lookup = build_chapter_lookup(chapters)

    rows: list[dict] = []
    for chapter in chapters:
        chapter_id = chapter["id"]
        chapter_verses = fetch_chapter_verses(chapter_id, translation_id)
        chapter_meta = chapter_lookup[chapter_id]
        for verse in chapter_verses:
            rows.append(normalize_verse_row(verse, chapter_meta, translation_id))

    df = pd.DataFrame(rows).sort_values(by=["chapter_id", "verse_number"]).reset_index(drop=True)

    raw_json_path = RAW_DIR / f"verses_translation_{translation_id}.json"
    csv_path = PROCESSED_DIR / f"verses_translation_{translation_id}.csv"
    parquet_path = PROCESSED_DIR / f"verses_translation_{translation_id}.parquet"

    df.to_json(raw_json_path, orient="records", indent=2, force_ascii=False)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"Saved {len(df)} verses for translation {translation_id}")
    print(f"Raw JSON: {raw_json_path}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")


if __name__ == "__main__":
    main()
