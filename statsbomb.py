"""
statsbomb.py  —  event data (x/y shot coordinates) loader
=========================================================
Downloads StatsBomb Open Data (free, no API key) for a set of international
tournaments and extracts every SHOT with its pitch coordinates. This is the raw
material for the xG model in xg.py.

IMPORTANT SCOPE NOTE: StatsBomb's free international coverage is only a handful of
big tournaments (recent World Cups, Euros, Copa America). So this data augments
the ~30 elite teams that appear in them; most national teams have NO event data
and keep using real goals in the Dixon-Coles fit. See README / model.py.

Pipeline: competitions.json -> matches/{comp}/{season}.json -> events/{match}.json
We keep only shots, and cache the extracted shot table per tournament as a small
CSV so we never re-download or re-parse (event files are big; the shot table is tiny).
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import config
from data_loader import canonicalize

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# (competition_name, season_name) -> we look up the numeric ids at runtime.
# Default set chosen for relevance: recent men's internationals with modern squads.
DEFAULT_TOURNAMENTS = [
    ("FIFA World Cup", "2018"),
    ("UEFA Euro", "2020"),
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2024"),
    ("Copa America", "2024"),
]

# StatsBomb pitch is 120 x 80; the goal is at x=120, mouth spanning y in [36, 44].
GOAL_X = 120.0
GOAL_Y = 40.0

_SB_CACHE = os.path.join(config.CACHE_DIR, "statsbomb")

# Names StatsBomb spells differently from the martj42 results dataset. Passed
# through data_loader.canonicalize afterwards, so we only list SB-specific ones.
_SB_NAME_FIXES = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "China PR": "China",
    "United States": "United States",
    "Czech Republic": "Czechia",
}


def _sb_team(name: str) -> str:
    return canonicalize(_SB_NAME_FIXES.get(name, name))


def _get_json(url: str, cache_path: str | None = None):
    """Fetch+parse JSON, caching the raw bytes if a cache path is given."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    return data


def _competition_ids(tournaments) -> dict:
    """Map each (name, season) to (competition_id, season_id)."""
    comps = _get_json(f"{SB_BASE}/competitions.json",
                      os.path.join(_SB_CACHE, "competitions.json"))
    out = {}
    for c in comps:
        key = (c["competition_name"], c["season_name"])
        if key in tournaments:
            out[key] = (c["competition_id"], c["season_id"])
    missing = set(tournaments) - set(out)
    if missing:
        print(f"[statsbomb] warning: not found in open data: {missing}")
    return out


def _extract_shots(match_meta: dict, events: list) -> list[dict]:
    """Pull shot rows (with x/y and features) from one match's event list."""
    home = _sb_team(match_meta["home_team"]["home_team_name"])
    away = _sb_team(match_meta["away_team"]["away_team_name"])
    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        loc = e.get("location")
        if not loc or len(loc) < 2:
            continue
        shot = e["shot"]
        team = _sb_team(e["team"]["name"])
        rows.append({
            "match_id": match_meta["match_id"],
            "date": match_meta["match_date"],
            "home_team": home,
            "away_team": away,
            "team": team,
            "opponent": away if team == home else home,
            "x": float(loc[0]),
            "y": float(loc[1]),
            "is_goal": int(shot["outcome"]["name"] == "Goal"),
            "body_part": shot.get("body_part", {}).get("name", "Unknown"),
            "shot_type": shot.get("type", {}).get("name", "Open Play"),
            "sb_xg": shot.get("statsbomb_xg", None),  # for sanity comparison only
        })
    return rows


def _load_tournament_shots(name: str, season: str, cid: int, sid: int,
                           force: bool = False) -> pd.DataFrame:
    """Download+extract all shots for one tournament, cached as a compact CSV."""
    os.makedirs(_SB_CACHE, exist_ok=True)
    tag = f"{name}_{season}".replace(" ", "").replace("/", "-")
    shot_csv = os.path.join(_SB_CACHE, f"shots_{tag}.csv")
    if os.path.exists(shot_csv) and not force:
        return pd.read_csv(shot_csv, parse_dates=["date"])

    matches = _get_json(f"{SB_BASE}/matches/{cid}/{sid}.json",
                        os.path.join(_SB_CACHE, f"matches_{tag}.json"))
    print(f"[statsbomb] {name} {season}: fetching events for {len(matches)} matches ...")

    def fetch(mm):
        ev = _get_json(f"{SB_BASE}/events/{mm['match_id']}.json",
                       os.path.join(_SB_CACHE, "events", f"{mm['match_id']}.json"))
        return _extract_shots(mm, ev)

    rows = []
    # Threaded I/O: event files are network-bound, so parallel download is much faster.
    with ThreadPoolExecutor(max_workers=8) as ex:
        for shot_rows in ex.map(fetch, matches):
            rows.extend(shot_rows)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(shot_csv, index=False)
    print(f"[statsbomb]   -> {len(df)} shots cached to {os.path.basename(shot_csv)}")
    return df


def load_shots(tournaments=None, force: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of every shot across the requested tournaments, with columns:
    match_id, date, home_team, away_team, team, opponent, x, y, is_goal,
    body_part, shot_type, sb_xg.
    """
    if tournaments is None:
        tournaments = DEFAULT_TOURNAMENTS
    ids = _competition_ids(set(tournaments))
    frames = []
    for (name, season) in tournaments:
        if (name, season) in ids:
            cid, sid = ids[(name, season)]
            frames.append(_load_tournament_shots(name, season, cid, sid, force=force))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    shots = load_shots()
    print(f"\ntotal shots: {len(shots)} across {shots['match_id'].nunique()} matches")
    print(f"date range: {shots['date'].min().date()} -> {shots['date'].max().date()}")
    print(f"overall conversion (goals/shots): {shots['is_goal'].mean():.3f}")
    print("\nshots by body part:\n", shots["body_part"].value_counts().to_string())
    print("\nsample rows:")
    print(shots[["date", "team", "opponent", "x", "y", "is_goal", "shot_type", "sb_xg"]]
          .head(6).to_string(index=False))
    # sanity: any StatsBomb team names that don't match our results dataset?
    import data_loader
    known = set(pd.unique(
        data_loader.load_matches()[["home_team", "away_team"]].values.ravel()))
    sb_teams = set(shots["team"]) | set(shots["opponent"])
    print("\nSB teams NOT matching results dataset (need name fix):",
          sorted(sb_teams - known))
