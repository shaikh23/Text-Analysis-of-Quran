"""Generator for notebooks/01_eda.ipynb.

Keeps the EDA notebook reproducible from source control. Re-run after editing
to regenerate the notebook. Sections are appended area-by-area as the analysis
progresses.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- #
md(
    """
# Quran Translation — Exploratory Data Analysis

Analysis of the **English Saheeh International** translation (`translation_id=20`)
of the Quran, 6,236 verses across 114 chapters.

> **Scope note:** all findings describe an *English translation*, not the Arabic
> source text. Word counts, frequencies, and lengths are properties of the
> translation's prose.

Analysis logic lives in the reusable `quran_analysis` package; this notebook is a
thin presentation layer over `quran_analysis.eda` and `quran_analysis.text`.
"""
)

code(
    """
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve repo root by walking up to the directory containing data/, so the
# notebook works whether launched from repo root or notebooks/.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "src"))

from quran_analysis import eda

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

df = eda.load_verses(ROOT / "data/processed/verses_translation_20.parquet")
df.shape
"""
)

code(
    """
# Derived text columns:
#   clean_text       - footnotes/markup stripped, KEEPS [translator interpolations]
#   scripture_text   - interpolations removed (directly-rendered words only)
#   word_count / scripture_word_count - token counts for each variant
df[["verse_key", "clean_text", "scripture_text", "word_count", "scripture_word_count"]].head()
"""
)

# --------------------------------------------------------------------------- #
md(
    """
## Area 1 — Data Quality

Establish that the dataset is complete and understand its text-cleaning needs
before measuring anything.
"""
)

code(
    """
# Headline integrity checks.
eda.data_quality_report(df)
"""
)

code(
    """
# Verse counts per chapter vs the chapter metadata's expected count.
# An empty result below means every chapter has exactly the expected verses.
integrity = eda.verse_count_integrity(df)
mismatches = integrity.loc[~integrity["matches"]]
print(f"{len(mismatches)} / {len(integrity)} chapters mismatch expected verse counts")
mismatches
"""
)

code(
    """
# The 126 "duplicate" verse texts are legitimate refrains, not data errors.
# Ar-Rahman's recurring rhetorical question dominates.
eda.duplicate_examples(df).head(8)[["repeats", "verse_keys", "clean_text"]]
"""
)

code(
    """
# Footnote markup coverage (now stripped from clean_text).
ax = (
    df["footnote_count"].clip(upper=5)
    .value_counts().sort_index()
    .plot(kind="bar", color="steelblue")
)
ax.set(xlabel="footnote markers per verse (5 = 5+)", ylabel="verses",
       title="Footnote markup distribution")
plt.tight_layout()
"""
)

code(
    """
# Translator interpolations: 55% of verses contain [bracketed] clarifications,
# adding ~6.5% to the total word count. We keep both variants so each downstream
# analysis can choose its basis.
total = df["word_count"].sum()
scripture = df["scripture_word_count"].sum()
print(f"With interpolations : {total:,} words")
print(f"Scripture-only      : {scripture:,} words")
print(f"Interpolations add  : {total - scripture:,} words ({(total - scripture) / total:.1%})")
"""
)

md(
    """
**Area 1 takeaways**

- Dataset is **complete and clean**: 6,236 verses, 114 chapters, zero verse-count
  mismatches, no null/empty translations.
- Two cleaning concerns handled: `<sup>` footnote markup (1,612 verses) and
  `[bracketed]` translator interpolations (55% of verses, ~6.5% of words).
- "Duplicate" texts are real refrains (e.g. Ar-Rahman ×24), not errors.
"""
)

# --------------------------------------------------------------------------- #
md(
    """
## Area 2 — Structural Overview

### Thread A — Ordering: chronological vs canonical

The Quran's 114 chapters are **not** arranged in the order they were revealed.
Here we quantify how they *are* arranged, how far that departs from chronology,
and whether any alternate (e.g. ring/concentric) structure is detectable.
"""
)

code(
    """
order = eda.ordering_table(df)   # chapter_id, revelation_order, verses, displacement, ...

# Q1: the book is arranged roughly longest-chapter-first.
rho = order["chapter_id"].corr(order["verses"], method="spearman")
print(f"Spearman(canonical position, chapter length) = {rho:.3f}  (strong 'longest-first')")
for lo, hi, lbl in [(1, 38, "first third"), (39, 76, "middle third"), (77, 114, "last third")]:
    seg = order[order.chapter_id.between(lo, hi)]
    print(f"  {lbl:13s} (ch {lo:>3}-{hi:<3}): median {seg.verses.median():5.1f} verses")
"""
)

code(
    """
# Q1 visual: chapter length by canonical position, with the longest-first trend.
fig, ax = plt.subplots(figsize=(12, 4))
colors = order["revelation_place"].map({"makkah": "steelblue", "madinah": "darkorange"})
ax.bar(order["chapter_id"], order["verses"], color=colors, width=0.85)
ax.set(xlabel="canonical chapter number", ylabel="verses",
       title="Chapter length by position — the 'longest-first' arrangement")
ax.annotate("Al-Fatihah (opening,\\nshort but placed first)", xy=(1, 7), xytext=(8, 120),
            arrowprops=dict(arrowstyle="->"), fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="steelblue", label="Makkah"), Patch(color="darkorange", label="Madinah")])
plt.tight_layout()
"""
)

code(
    """
# Q2: displacement between revelation order and canonical position.
# Largest movers are Madinan chapters revealed late but placed near the front.
movers = order.reindex(order["displacement"].abs().sort_values(ascending=False).index).head(6)
print(movers[["chapter_id", "name", "revelation_place", "revelation_order", "displacement", "verses"]].to_string(index=False))
print()
print("Mean signed displacement by revelation place:")
print(order.groupby("revelation_place", observed=True)["displacement"].mean().round(1).to_string())
"""
)

code(
    """
# Q2 visual: revelation order vs canonical position. Distance from the diagonal
# is the displacement; Madinan chapters sit far above it (revealed late, placed early).
fig, ax = plt.subplots(figsize=(7, 7))
for place, c in [("makkah", "steelblue"), ("madinah", "darkorange")]:
    s = order[order.revelation_place == place]
    ax.scatter(s["chapter_id"], s["revelation_order"], c=c, label=place, s=28)
ax.plot([1, 114], [1, 114], "k--", lw=1, label="chronological = canonical")
ax.set(xlabel="canonical chapter number", ylabel="revelation order",
       title="Revelation order vs canonical position")
ax.legend()
plt.tight_layout()
"""
)

code(
    """
# Q3 (data caveat): in this dataset revelation_order does NOT interleave Makkah/Madinah.
mak = order[order.revelation_place == "makkah"]["revelation_order"]
mad = order[order.revelation_place == "madinah"]["revelation_order"]
print(f"Makkah revelation orders: {mak.min()}-{mak.max()}")
print(f"Madinah revelation orders: {mad.min()}-{mad.max()}")
print("=> all Makkan chapters precede all Madinan ones: revelation_order is a COARSE")
print("   traditional sequence (place-grouped), not a fine verse-level chronology.")
"""
)

code(
    """
# Q4: chronology is partly preserved locally in the canonical order.
seq = order["revelation_order"].to_numpy()
import numpy as np
lag1 = np.corrcoef(seq[:-1], seq[1:])[0, 1]
print(f"lag-1 autocorrelation of revelation order along canonical sequence = {lag1:.3f}")
print("Makkah/Madinah run-clustering:", eda.place_runs(df))
print("(far fewer runs than the random expectation => same-origin chapters cluster)")
"""
)

md(
    """
### Alternate structures — ring / concentric composition

Scholars (e.g. Farrin; Cuypers' *Semitic rhetoric*) argue some chapters are
arranged as **rings** — section A mirrors section A′ around a thematic center.
We test this on **Al-Baqarah** (the classic example) using *motif profiles*: each
section is scored over curated theme lexicons, and we check whether mirror-position
sections are more alike than a shuffled null allows.

> Caveat: this operates on the **English translation** and coarse sections. Ring
> arguments rest on Arabic verbal echoes and human-identified motifs, so a null
> here is **not** a refutation — only evidence that the pattern isn't detectable
> by these surface methods.
"""
)

code(
    """
from quran_analysis import themes

baqarah = df[df.chapter_id == 2]
rukus = baqarah.groupby("ruku_number")["clean_text"].apply(" ".join).tolist()
nine = [" ".join(g) for g in np.array_split(baqarah["clean_text"].tolist(), 9)]

for label, units in [("40 ruku units", rukus), ("9 equal sections", nine)]:
    res = eda.ring_test(themes.motif_profiles(units))
    print(f"{label:18s}: mirror={res['mirror_similarity']:.3f}  "
          f"null={res['null_mean']:.3f}  z={res['z']:+.2f}  p={res['p_value']:.3f}")
print()
print("z ~ 0 and p ~ 0.5 at both granularities => no detectable concentric symmetry.")
"""
)

md(
    """
**Thread A takeaways**

- **Arrangement is by length, not chronology**: Spearman(position, length) = −0.85;
  median chapter length falls 98.5 → 37.5 → 11 verses across the three thirds.
  Al-Fatihah is the deliberate short-but-first exception (the opening).
- **The big order-shifts are systematically Madinan** — Al-Ma'idah (+107) and
  At-Tawbah (+104) were revealed last but placed near the front.
- **Data caveat**: `revelation_order` here is place-grouped (all Makkan 1–86,
  all Madinan 87–114), so treat it as a coarse sequence, not a fine timeline.
- **Local chronology survives** in the canonical order (lag-1 autocorr 0.44;
  25 origin-runs vs ~43 expected) — same-origin chapters cluster.
- **Ring structure**: not detectable in Al-Baqarah via motif profiles on the
  English translation (z ≈ 0). A fair test of the scholarly claim needs Arabic
  roots / verbal-echo tracking.
"""
)

# --------------------------------------------------------------------------- #
md(
    """
### Thread B — Makkah vs Madinah profile

A popular characterization is that "Madinan chapters are longer." The data tells
a more precise story: the contrast is driven by **verse length**, not chapter
length, and the downstream density metrics all follow from that.
"""
)

code(
    """
# One-glance profile. Note: similar words/page across places, but Madinan verses
# are ~3x longer, so Madinan pages hold fewer (longer) verses.
eda.place_profile(df)
"""
)

code(
    """
# Q5: chapter verse-counts are NOT significantly different by place.
from scipy.stats import mannwhitneyu

mak = order[order.revelation_place == "makkah"]["verses"]
mad = order[order.revelation_place == "madinah"]["verses"]
u, p = mannwhitneyu(mak, mad)
print(f"verses/chapter  -  Makkah median {mak.median():.0f}, Madinah median {mad.median():.0f}")
print(f"Mann-Whitney U p = {p:.3f}  ->  no significant difference in chapter length")
print("(Madinan chapters are bimodal: a few giants + many short ones.)")
"""
)

code(
    """
# Q6: but verse length DOES differ, systemically. Plot the word-per-verse
# distributions; Madinan mass sits well to the right.
fig, ax = plt.subplots(figsize=(11, 4))
for place, c in [("makkah", "steelblue"), ("madinah", "darkorange")]:
    sns.kdeplot(df[df.revelation_place == place]["word_count"].clip(upper=120),
                ax=ax, fill=True, alpha=0.4, color=c, label=place)
ax.set(xlabel="words per verse (clipped at 120)", ylabel="density",
       title="Verse length by revelation place — the real Makkah/Madinah signal")
ax.legend()
plt.tight_layout()
"""
)

code(
    """
# Q7: where Madinan chapters sit in the book (share Madinan by canonical third).
thirds = pd.cut(order["chapter_id"], [0, 38, 76, 114], labels=["first", "middle", "last"])
share = order.assign(third=thirds).groupby("third", observed=True)["revelation_place"].apply(
    lambda s: (s == "madinah").mean()
)
ax = share.plot(kind="bar", color="darkorange")
ax.set(ylabel="fraction Madinan", title="Madinan chapters cluster in the front/middle, not the tail")
ax.bar_label(ax.containers[0], fmt="%.0f%%", labels=[f"{v:.0%}" for v in share])
plt.tight_layout()
"""
)

md(
    """
**Thread B takeaways**

- **It's verse length, not chapter length.** Chapter verse-counts don't differ
  significantly by place (Mann-Whitney p ≈ 0.71); Madinan *verses* are ~2–3×
  longer (median 36 vs 12 words/verse), systemically.
- **Density follows from verse length**: Madinan pages hold fewer verses
  (6 vs 10) but roughly equal words (~240 vs ~220) — a near-constant page word
  budget, echoing the equal-length juz finding.
- **Placement**: Madinan chapters concentrate in the front/middle thirds and are
  nearly absent (8%) from the short-Makkan tail — a consequence of the
  longest-first ordering meeting their bimodal sizes.
"""
)

# --------------------------------------------------------------------------- #
nb["cells"] = cells
out = Path("notebooks/01_eda.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
