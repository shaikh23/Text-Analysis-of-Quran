# Text Analysis of Quran

Python project for thematic and exploratory analysis of English Quran translations.

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

3. Copy environment template:

```bash
cp .env.example .env
```

4. Fetch chapter metadata from Quran Foundation API:

```bash
python scripts/fetch_chapters.py
```

5. Build normalized verse dataset for one English translation:

```bash
python scripts/build_translation_dataset.py
```

This writes:

- `data/raw/verses_translation_<id>.json`
- `data/processed/verses_translation_<id>.csv`
- `data/processed/verses_translation_<id>.parquet`

Set translation ID in `.env` via `QURAN_TRANSLATION_ID`.

## Initial Project Layout

- `src/quran_analysis/`: package code
- `scripts/`: runnable scripts for data ingestion and utilities
- `data/raw/`: unprocessed downloaded data
- `data/processed/`: cleaned/tabular outputs
- `notebooks/`: analysis notebooks

## Data Source Notes

Primary source is the Quran Foundation API docs:
https://api-docs.quran.foundation/

When publishing analysis, explicitly label findings as analysis of English translations.
