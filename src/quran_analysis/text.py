"""Text cleaning and tokenization for English Quran translations.

The translation text from the Quran Foundation API embeds footnote markers as
HTML, e.g. ``<sup foot_note=195932>1</sup>``. It also uses square brackets for
translator interpolations (``[All] praise``). These helpers strip that markup so
the prose can be measured and tokenized consistently.
"""

from __future__ import annotations

import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ``<sup foot_note=NNN>N</sup>`` footnote markers.
_FOOTNOTE_RE = re.compile(r"<sup\b[^>]*>.*?</sup>", flags=re.DOTALL | re.IGNORECASE)
# Any residual HTML tag (defensive; the dataset only contains <sup>).
_TAG_RE = re.compile(r"<[^>]+>")
# Editorial abbreviations inside translator glosses, e.g. "[i.e., ...]".
_ABBREV_RE = re.compile(r"\b(?:i\.e\.|e\.g\.)", flags=re.IGNORECASE)
# Translator interpolations in square brackets, keeping the inner words.
_BRACKET_RE = re.compile(r"[\[\]]")
# A full bracketed interpolation including its contents, e.g. "[i.e., Prophet]".
_INTERPOLATION_RE = re.compile(r"\[[^\]]*\]")
# Word tokens: letters (incl. accented) and apostrophes within words.
_WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+(?:'[A-Za-z]+)?")

# Domain stopwords on top of the generic English list. These are translation
# artifacts and high-frequency function words that dominate raw counts without
# carrying thematic signal.
_EXTRA_STOPWORDS = frozenset({"indeed", "allah", "allāh", "upon", "us", "thee", "thy"})

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | _EXTRA_STOPWORDS


def strip_footnotes(text: str) -> str:
    """Remove ``<sup>`` footnote markers and their contents."""
    return _FOOTNOTE_RE.sub("", text)


def count_footnotes(text: str) -> int:
    """Number of footnote markers in a verse."""
    return len(_FOOTNOTE_RE.findall(text))


def clean_text(text: str) -> str:
    """Return display-ready prose: no markup, no brackets, normalized whitespace."""
    if not text:
        return ""
    text = strip_footnotes(text)
    text = _TAG_RE.sub("", text)
    text = _ABBREV_RE.sub("", text)
    text = _BRACKET_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def scripture_only(text: str) -> str:
    """Cleaned prose with translator interpolations removed entirely.

    Drops the contents of ``[...]`` brackets (translator clarifications), leaving
    only the directly-rendered words. Compare against :func:`clean_text`, which
    keeps the bracketed words.
    """
    if not text:
        return ""
    text = strip_footnotes(text)
    text = _TAG_RE.sub("", text)
    text = _INTERPOLATION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _fold_accents(text: str) -> str:
    """Map accented Latin chars to ASCII so e.g. ``Allāh`` -> ``allah``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Lowercased, accent-folded word tokens from cleaned text."""
    tokens = [t.lower() for t in _WORD_RE.findall(_fold_accents(clean_text(text)))]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def word_count(text: str) -> int:
    """Number of word tokens after cleaning (markup excluded)."""
    return len(_WORD_RE.findall(clean_text(text)))


def char_count(text: str) -> int:
    """Number of characters in cleaned text (whitespace included, markup excluded)."""
    return len(clean_text(text))
