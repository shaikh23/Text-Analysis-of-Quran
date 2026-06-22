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


def place_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-revelation-place profile of the key structural metrics.

    Surfaces the central finding that the Makkah/Madinah contrast is driven by
    *verse length*: Madinan verses are far longer, which in turn lowers verses
    per page and per ruku while words-per-page stays roughly constant.
    """
    per_ch = df.groupby(["revelation_place", "chapter_id"], observed=True).agg(
        verses=("verse_id", "size"),
        words=("word_count", "sum"),
        pages=("page_number", "nunique"),
        rukus=("ruku_number", "nunique"),
        mean_words_per_verse=("word_count", "mean"),
    )
    per_ch["verses_per_page"] = per_ch["verses"] / per_ch["pages"]
    per_ch["words_per_page"] = per_ch["words"] / per_ch["pages"]
    per_ch["verses_per_ruku"] = per_ch["verses"] / per_ch["rukus"]

    place = per_ch.groupby(level="revelation_place", observed=True)
    out = pd.DataFrame(
        {
            "chapters": place.size(),
            "median_verses_per_chapter": place["verses"].median(),
            "median_words_per_verse": place["mean_words_per_verse"].median().round(1),
            "median_verses_per_page": place["verses_per_page"].median().round(1),
            "median_words_per_page": place["words_per_page"].median().round(0),
            "median_verses_per_ruku": place["verses_per_ruku"].median().round(1),
        }
    )
    return out.reset_index()


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
# Ordering (canonical vs revelation)
# --------------------------------------------------------------------------- #
def ordering_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per chapter with canonical id, revelation order, and displacement.

    ``displacement = revelation_order - chapter_id``: large positive means a
    chapter was revealed late but placed early in the canonical order.
    """
    out = chapter_summary(df).sort_values("chapter_id").reset_index(drop=True)
    out["displacement"] = out["revelation_order"] - out["chapter_id"]
    return out


def place_runs(df: pd.DataFrame) -> dict[str, float]:
    """Run-clustering of revelation place along the canonical order.

    Fewer observed runs than the chance expectation means same-origin chapters
    sit adjacent (clustered) rather than interleaved.
    """
    places = ordering_table(df)["revelation_place"].to_numpy()
    n = len(places)
    counts = pd.Series(places).value_counts()
    a, b = int(counts.iloc[0]), int(counts.iloc[1])
    observed = 1 + int((places[1:] != places[:-1]).sum())
    expected = 1 + 2 * a * b / n
    return {"observed_runs": observed, "expected_runs_if_random": round(expected, 1)}


def ring_test(profiles, n_perm: int = 2000, seed: int = 0) -> dict[str, float]:
    """Test a sequence of section profiles for concentric (ring) symmetry.

    ``profiles`` is an (n_sections x n_features) matrix. We measure the mean
    cosine similarity between mirror pairs (i, n-1-i) and compare it against a
    permutation null that shuffles section order. A positive z with low p means
    sections are more similar to their mirror partner than chance allows.
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    profiles = np.asarray(profiles, dtype=float)
    profiles = profiles[profiles.sum(axis=1) > 0]
    n = len(profiles)
    sim = cosine_similarity(profiles)

    def mean_mirror(matrix: "np.ndarray") -> float:
        return float(np.mean([matrix[i, n - 1 - i] for i in range(n // 2)]))

    observed = mean_mirror(sim)
    rng = np.random.default_rng(seed)
    null = np.array(
        [mean_mirror(sim[np.ix_(p, p)]) for p in (rng.permutation(n) for _ in range(n_perm))]
    )
    return {
        "n_sections": n,
        "mirror_similarity": round(observed, 4),
        "null_mean": round(float(null.mean()), 4),
        "z": round(float((observed - null.mean()) / null.std()), 3),
        "p_value": round(float((null >= observed).mean()), 4),
    }


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
def word_frequencies(df: pd.DataFrame, *, drop_stopwords: bool = True) -> pd.Series:
    """Full token-frequency series across all verses, descending by count."""
    counter: Counter[str] = Counter()
    for txt in df["translation_text"]:
        counter.update(qtext.tokenize(txt, drop_stopwords=drop_stopwords))
    return pd.Series(counter, name="count").sort_values(ascending=False)


def top_words(df: pd.DataFrame, n: int = 30, *, drop_stopwords: bool = True) -> pd.DataFrame:
    """Most frequent tokens across all verses."""
    freqs = word_frequencies(df, drop_stopwords=drop_stopwords).head(n)
    return freqs.rename_axis("word").reset_index()


def chapter_signature_terms(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Each chapter's most distinctive terms via TF-IDF across all 114 chapters.

    With 114 documents, TF-IDF meaningfully down-weights words common to the whole
    book and surfaces each chapter's signature vocabulary.
    """
    chapters = sorted(df["chapter_id"].unique())
    names = df.groupby("chapter_id")["chapter_name_simple"].first()
    docs = [
        " ".join(
            tok
            for t in df.loc[df["chapter_id"] == c, "translation_text"]
            for tok in qtext.tokenize(t, drop_stopwords=True)
        )
        for c in chapters
    ]
    vectorizer = TfidfVectorizer(token_pattern=r"[a-z']+", min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    rows = []
    for i, c in enumerate(chapters):
        row = matrix[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        rows.append(
            {"chapter_id": c, "name": names[c], "signature_terms": ", ".join(vocab[top_idx])}
        )
    return pd.DataFrame(rows)


def vocabulary_richness(
    df: pd.DataFrame,
    group_col: str = "revelation_place",
    *,
    n_boot: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Size-matched lexical diversity per group.

    Raw type-token ratio (TTR) is biased by sample size, so each group is
    repeatedly subsampled to the smallest group's content-token count; TTR and
    the hapax rate are averaged over ``n_boot`` draws to compare fairly.
    """
    import numpy as np

    tokens = {
        str(g): [
            tok
            for t in df.loc[df[group_col] == g, "translation_text"]
            for tok in qtext.tokenize(t, drop_stopwords=True)
        ]
        for g in df[group_col].dropna().unique()
    }
    sample_size = min(len(t) for t in tokens.values())
    rng = np.random.default_rng(seed)

    rows = []
    for group, toks in tokens.items():
        arr = np.array(toks, dtype=object)
        ttrs, hapax_rates = [], []
        for _ in range(n_boot):
            draw = rng.choice(arr, size=sample_size, replace=False)
            counts = Counter(draw)
            ttrs.append(len(counts) / sample_size)
            hapax_rates.append(sum(1 for v in counts.values() if v == 1) / len(counts))
        rows.append(
            {
                "group": group,
                "total_content_tokens": len(toks),
                "matched_sample": sample_size,
                "ttr_matched": round(float(np.mean(ttrs)), 4),
                "hapax_rate_matched": round(float(np.mean(hapax_rates)), 4),
            }
        )
    return pd.DataFrame(rows)


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
