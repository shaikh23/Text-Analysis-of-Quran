from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_base_url: str = os.getenv("QURAN_API_BASE_URL", "https://api.quran.com/api/v4")
    api_key: str = os.getenv("QURAN_API_KEY", "")


settings = Settings()
