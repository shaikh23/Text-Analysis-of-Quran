"""Curated thematic motif lexicons and scoring.

Themes are defined as keyword sets ("lexicons") over the English translation.
This is interpretable and reproducible: we impose the categories explicitly
rather than inferring them. Matching is done on accent-folded word *stems* via
prefix matching, so "believ" catches believe/believed/believers/belief.

These lexicons back two analyses:
- thematic prevalence and Makkah/Madinah splits (Thread C), and
- motif-correspondence ring tests, which ask whether a chapter's sections are
  concentrically symmetric in *what they talk about* (a fairer test than raw
  bag-of-words token overlap).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from quran_analysis import text as qtext

_WORD = re.compile(r"[a-z']+")

# Each theme maps to keyword stems. Stems match by prefix after accent-folding
# and lowercasing, so list the shortest unambiguous root.
LEXICONS: dict[str, tuple[str, ...]] = {
    "belief_guidance": ("believ", "faith", "guid", "righteous", "piou", "pieti"),
    "disbelief_hypocrisy": ("disbeliev", "hypocri", "deni", "deny", "reject", "transgress"),
    "israel_moses": ("israel", "moses", "pharaoh", "calf", "torah", "jew"),
    "abraham_kaba": ("abraham", "ishmael", "sacred house", "ka'ba", "kaba", "pilgrim", "hajj"),
    "prayer_qibla": ("prayer", "pray", "qibla", "prostrat", "mosque", "masjid", "face toward"),
    "law_legislation": (
        "lawful", "unlawful", "prescrib", "ordain", "divorc", "marri", "orphan",
        "inherit", "usury", "interest", "contract", "debt", "oath", "witness",
    ),
    "fighting_striving": ("fight", "kill", "slain", "expel", "enemy", "cause of allah", "strive"),
    "food_dietary": ("eat", "food", "swine", "blood", "forbidden", "wine", "intoxic"),
    "charity_wealth": ("spend", "charit", "poor", "wealth", "alms", "needy"),
    "afterlife_judgment": ("hereafter", "resurrect", "paradise", "hellfire", "hell", "judgment", "reckoning", "punishment"),
    "mercy_forgiveness": ("merci", "merc", "forgiv", "pardon", "compassion"),
    "covenant_signs": ("covenant", "sign", "miracle", "messenger", "revelation", "scripture", "book"),
}


def _match_count(words: list[str], folded_text: str, stem: str) -> int:
    """Hits for one stem: word-prefix matches, or phrase substring for multi-word stems.

    Word-prefix matching (rather than raw substring) avoids false positives like
    the stem ``eat`` matching ``created``/``great``/``death``.
    """
    if " " in stem:  # multi-word phrase, e.g. "cause of allah"
        return folded_text.count(stem)
    return sum(1 for w in words if w.startswith(stem))


def score_text(text: str) -> dict[str, int]:
    """Count motif keyword hits per theme in a single text."""
    # Fold accents so e.g. "Mūsā"-style renderings still match plain stems.
    folded = qtext._fold_accents(qtext.clean_text(text).lower())
    words = _WORD.findall(folded)
    return {
        theme: sum(_match_count(words, folded, stem) for stem in stems)
        for theme, stems in LEXICONS.items()
    }


def score_frame(texts: pd.Series) -> pd.DataFrame:
    """Theme-score a series of texts; returns one row per text, one column per theme."""
    return pd.DataFrame([score_text(t) for t in texts], index=texts.index).fillna(0).astype(int)


def motif_profiles(units: list[str], *, normalize: bool = True) -> np.ndarray:
    """Motif-profile matrix: one row per text unit, one column per theme.

    With ``normalize`` each row is L1-normalized into a theme distribution so that
    long and short units are comparable.
    """
    mat = np.array([list(score_text(u).values()) for u in units], dtype=float)
    if normalize:
        totals = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, totals, out=np.zeros_like(mat), where=totals > 0)
    return mat
