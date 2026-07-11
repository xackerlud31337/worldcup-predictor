"""
data_loader.py  —  Stage 1: data loading
=========================================
Responsibilities:
  * Download the historical international-results CSV (no API key) and cache it.
  * Cache a *cleaned* copy so repeat runs skip both the download and the cleaning.
  * Canonicalise inconsistent team names into one spelling per country.
  * Provide helpers to filter by date range and to attach per-match weights
    (competition importance x exponential time-decay for recency).

The public entry point is `load_matches()`, which returns a tidy DataFrame with
columns: date, home_team, away_team, home_goals, away_goals, tournament, neutral.
"""

from __future__ import annotations

import os
import urllib.request
from datetime import date, datetime

import numpy as np
import pandas as pd

import config


# --------------------------------------------------------------------------- #
# Canonical team-name map
# --------------------------------------------------------------------------- #
# The dataset is fairly clean, but (a) countries have been renamed/merged over
# time and (b) other sources (openfootball, FIFA rankings) use different spellings.
# Mapping to ONE canonical name per modern national team means a team's whole
# history contributes to its strength estimate, and lets us join external data.
#
# Design choice: we treat a modern nation as the continuation of its predecessor
# where FIFA/most analysts do (e.g. West Germany -> Germany, Czechoslovakia ->
# Czechia). This is debatable for edge cases (USSR, Yugoslavia) but gives the
# successor team more history than starting from scratch. Flip these off if you
# prefer strict separation.
CANONICAL_TEAM_NAMES = {
    # Renames / successor states
    "West Germany": "Germany",
    "East Germany": "Germany",          # comment out for strict separation
    "Soviet Union": "Russia",
    "CIS": "Russia",
    "Czechoslovakia": "Czechia",
    "Czech Republic": "Czechia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Zaïre": "DR Congo",
    "Zaire": "DR Congo",
    "Congo DR": "DR Congo",
    # Common spelling variants across sources
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "USA": "United States",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Kyrgyz Republic": "Kyrgyzstan",
    "The Gambia": "Gambia",
    "Curacao": "Curaçao",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei",
    "Republic of Ireland": "Ireland",
}


def canonicalize(name: str) -> str:
    """Map a raw team name to its canonical spelling (identity if not in the map)."""
    if not isinstance(name, str):
        return name
    return CANONICAL_TEAM_NAMES.get(name.strip(), name.strip())


# --------------------------------------------------------------------------- #
# Download + cache
# --------------------------------------------------------------------------- #
def _ensure_cache_dir() -> None:
    os.makedirs(config.CACHE_DIR, exist_ok=True)


def _raw_csv_path() -> str:
    return os.path.join(config.CACHE_DIR, "results_raw.csv")


def _clean_cache_path() -> str:
    # CSV rather than parquet so we don't add a pyarrow dependency (minimal deps).
    return os.path.join(config.CACHE_DIR, "results_clean.csv")


def download_raw(force: bool = False) -> str:
    """Download the raw results CSV to the cache (skips if already present)."""
    _ensure_cache_dir()
    path = _raw_csv_path()
    if os.path.exists(path) and not force:
        return path
    print(f"[data] downloading {config.RESULTS_URL} ...")
    urllib.request.urlretrieve(config.RESULTS_URL, path)
    print(f"[data] saved raw CSV -> {path}")
    return path


def _build_clean(force_download: bool = False) -> pd.DataFrame:
    """Read raw CSV, clean it, and return the tidy DataFrame."""
    raw_path = download_raw(force=force_download)
    df = pd.read_csv(raw_path)

    # Standardise column names to our internal schema.
    df = df.rename(
        columns={
            "home_score": "home_goals",
            "away_score": "away_goals",
        }
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Canonicalise team names.
    df["home_team"] = df["home_team"].map(canonicalize)
    df["away_team"] = df["away_team"].map(canonicalize)

    # neutral flag: dataset stores strings "TRUE"/"FALSE".
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    # Drop rows with missing essentials.
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    # Keep only the columns we need, sorted chronologically (Elo needs order).
    keep = ["date", "home_team", "away_team", "home_goals", "away_goals",
            "tournament", "neutral"]
    df = df[keep].sort_values("date").reset_index(drop=True)
    return df


def _load_clean(force_download: bool = False) -> pd.DataFrame:
    """Return the cleaned DataFrame, using the on-disk cache when possible."""
    _ensure_cache_dir()
    cache = _clean_cache_path()
    raw = _raw_csv_path()
    # Rebuild the clean cache if it is missing or older than the raw file.
    stale = (
        force_download
        or not os.path.exists(cache)
        or (os.path.exists(raw) and os.path.getmtime(cache) < os.path.getmtime(raw))
    )
    if stale:
        df = _build_clean(force_download=force_download)
        df.to_csv(cache, index=False)
        return df
    df = pd.read_csv(cache, parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(bool)
    return df


# --------------------------------------------------------------------------- #
# Weighting: competition importance x time-decay
# --------------------------------------------------------------------------- #
def _competition_weight(tournament: str) -> float:
    """Importance multiplier for a tournament name (substring match)."""
    if not isinstance(tournament, str):
        return config.COMPETITION_WEIGHTS["_default"]
    t = tournament.lower()
    for key, w in config.COMPETITION_WEIGHTS.items():
        if key == "_default":
            continue
        if key.lower() in t:
            return w
    return config.COMPETITION_WEIGHTS["_default"]


def add_weights(
    df: pd.DataFrame,
    reference_date: date | str | None = None,
    half_life_days: float | None = None,
) -> pd.DataFrame:
    """
    Attach a `weight` column = competition_importance * time_decay.

    time_decay = 0.5 ** (age_days / half_life_days), so a match one half-life old
    counts half as much. `reference_date` is "now" for the decay clock; pin it to
    the eve of a tournament for a leak-free back-test.
    """
    df = df.copy()
    if half_life_days is None:
        half_life_days = config.HALF_LIFE_DAYS
    if reference_date is None:
        reference_date = config.REFERENCE_DATE or date.today()
    ref = pd.Timestamp(reference_date)

    age_days = (ref - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    time_decay = 0.5 ** (age_days / half_life_days)

    comp = df["tournament"].map(_competition_weight).to_numpy(dtype=float)
    df["weight"] = comp * time_decay
    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_matches(
    start: str | None = None,
    end: str | None = None,
    min_date: str | None = None,
    reference_date: date | str | None = None,
    half_life_days: float | None = None,
    with_weights: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load cleaned international match results, filtered and (optionally) weighted.

    Parameters
    ----------
    start, end   : ISO date strings; keep matches with start <= date <= end.
    min_date     : hard floor (defaults to config.MIN_DATE) applied before `start`.
    reference_date, half_life_days : passed to `add_weights` for recency weighting.
    with_weights : attach the `weight` column.
    force_download : bypass caches and refetch.

    Returns a DataFrame sorted by date.
    """
    df = _load_clean(force_download=force_download)

    floor = min_date if min_date is not None else config.MIN_DATE
    if floor is not None:
        df = df[df["date"] >= pd.Timestamp(floor)]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]

    df = df.sort_values("date").reset_index(drop=True)

    if with_weights:
        df = add_weights(df, reference_date=reference_date, half_life_days=half_life_days)
    return df


if __name__ == "__main__":
    # Smoke test for Stage 1.
    m = load_matches()
    print("rows:", len(m))
    print("date range:", m["date"].min().date(), "->", m["date"].max().date())
    print("unique teams:", pd.unique(m[["home_team", "away_team"]].values.ravel()).size)
    print("\nsample:")
    print(m.tail(5).to_string(index=False))
    wc22 = m[(m.tournament == "FIFA World Cup") & (m.date.dt.year == 2022)]
    print("\n2022 WC matches loaded:", len(wc22))
    print("weight stats:", m["weight"].describe()[["min", "mean", "max"]].to_dict())
