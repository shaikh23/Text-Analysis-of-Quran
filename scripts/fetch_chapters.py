from __future__ import annotations

import json
from pathlib import Path

import requests

from quran_analysis.config import settings


RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "chapters.json"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{settings.api_base_url}/chapters"
    headers = {}
    if settings.api_key:
        headers["x-api-key"] = settings.api_key

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    payload = response.json()
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    chapter_count = len(payload.get("chapters", []))
    print(f"Saved {chapter_count} chapters to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
