"""Reusable EDA helpers for the verse-translation dataset.

These functions are plot-free and return plain DataFrames/Series so they can be
unit-tested and reused from notebooks or scripts. Plotting lives in the notebook.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from quran_analysis import text as qtext

DEFAULT_DATASET = Path("data/processed/verses_translation_20.parquet")


def load_verses(path: str | Path | None = None) -> pd.DataFrame:
    """Load the verse dataset and attach derived text columns.

    Two cleaned variants are provided:

    - ``clean_text`` keeps the translator's ``[bracketed]`` clarifications.
    - ``scripture_text`` removes them, leaving only directly-rendered words.

    Word counts are derived for each (``word_count`` / ``scripture_word_count``)
    so length and lexical analyses can choose either basis. Also adds
    ``char_count``, ``footnote_count``, and a categorical ``revelation_place``.
    """
    df = pd.read_parquet(path or DEFAULT_DATASET)
    df["clean_text"] = df["translation_text"].map(qtext.clean_text)
    df["scripture_text"] = df["translation_text"].map(qtext.scripture_only)
    df["word_count"] = df["clean_text"].map(qtext.word_count)
    df["scripture_word_count"] = df["scripture_text"].map(qtext.word_count)
    df["char_count"] = df["clean_text"].str.len()
    df["footnote_count"] = df["translation_text"].map(qtext.count_footnotes)
    df["revelation_place"] = df["revelation_place"].astype("category")
    return df


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def data_quality_report(df: pd.DataFrame) -> pd.Series:
    """High-level integrity checks as a labeled Series."""
    raw = df["translation_text"]
    counts = {
        "total_verses": len(df),
        "unique_chapters": df["chapter_id"].nunique(),
        "null_translation": int(raw.isna().sum()),
        "empty_translation": int((df["clean_text"].str.strip() == "").sum()),
        "duplicate_translation_text": int(df["clean_text"].duplicated().sum()),
        "verses_with_footnotes": int((df["footnote_count"] > 0).sum()),
        "total_footnotes": int(df["footnote_count"].sum()),
        "verses_with_brackets": int(raw.str.contains(r"\[", regex=True).sum()),
    }
    return pd.Series(counts, name="data_quality")


def verse_count_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Compare observed verses per chapter against ``chapter_verses_count``.

    Any row with ``matches=False`` signals missing or extra verses.
    """
    observed = df.groupby("chapter_id").size().rename("observed_verses")
    expected = df.groupby("chapter_id")["chapter_verses_count"].first().rename("expected_verses")
    names = df.groupby("chapter_id")["chapter_name_simple"].first()
    out = pd.concat([names, expected, observed], axis=1)
    out["matches"] = out["observed_verses"] == out["expected_verses"]
    return out.reset_index()


def duplicate_examples(df: pd.DataFrame, min_repeats: int = 2) -> pd.DataFrame:
    """Repeated verse texts (e.g. refrains) with where they occur."""
    grp = (
        df.groupby("clean_text")
        .agg(repeats=("verse_key", "size"), verse_keys=("verse_key", list))
        .query("repeats >= @min_repeats")
        .sort_values("repeats", ascending=False)
    )
    return grp.reset_index()


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def revelation_place_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Chapter and verse counts split by revelation place (Makkah/Madinah)."""
    chapters = df.drop_duplicates("chapter_id")
    by_place = pd.DataFrame(
        {
            "chapters": chapters.groupby("revelation_place", observed=True).size(),
            "verses": df.groupby("revelation_place", observed=True).size(),
            "mean_words_per_verse": df.groupby("revelation_place", observed=True)["word_count"].mean(),
        }
    )
    by_place["pct_verses"] = (by_place["verses"] / by_place["verses"].sum() * 100).round(1)
    return by_place.reset_index()


def chapter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chapter rollup ordered by chapter id."""
    g = df.groupby("chapter_id")
    out = pd.DataFrame(
        {
            "name": g["chapter_name_simple"].first(),
            "revelation_place": g["revelation_place"].first(),
            "revelation_order": g["revelation_order"].first(),
            "verses": g.size(),
            "total_words": g["word_count"].sum(),
            "mean_words_per_verse": g["word_count"].mean().round(1),
        }
    )
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Length
# --------------------------------------------------------------------------- #
def length_extremes(df: pd.DataFrame, n: int = 10) -> dict[str, pd.DataFrame]:
    """Longest and shortest verses by word count."""
    cols = ["verse_key", "chapter_name_simple", "word_count", "clean_text"]
    ordered = df.sort_values("word_count")
    return {
        "shortest": ordered.head(n)[cols].reset_index(drop=True),
        "longest": ordered.tail(n)[cols].iloc[::-1].reset_index(drop=True),
    }


# --------------------------------------------------------------------------- #
# Lexical
# --------------------------------------------------------------------------- #
def top_words(df: pd.DataFrame, n: int = 30, *, drop_stopwords: bool = True) -> pd.DataFrame:
    """Most frequent tokens across all verses."""
    counter: Counter[str] = Counter()
    for txt in df["translation_text"]:
        counter.update(qtext.tokenize(txt, drop_stopwords=drop_stopwords))
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def distinctive_terms(
    df: pd.DataFrame,
    group_col: str = "revelation_place",
    top_n: int = 15,
) -> dict[str, pd.DataFrame]:
    """TF-IDF distinctive terms per group.

    Each group's verses are concatenated into one document; TF-IDF then surfaces
    terms that are characteristic of that group relative to the others.
    """
    groups = sorted(df[group_col].dropna().unique())
    docs = [
        " ".join(
            tok
            for t in df.loc[df[group_col] == g, "translation_text"]
            for tok in qtext.tokenize(t, drop_stopwords=True)
        )
        for g in groups
    ]
    vectorizer = TfidfVectorizer(token_pattern=r"[a-z']+", min_df=1)
    matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    result: dict[str, pd.DataFrame] = {}
    for i, group in enumerate(groups):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        result[str(group)] = pd.DataFrame(
            {"term": vocab[top_idx], "tfidf": row[top_idx].round(4)}
        )
    return result
