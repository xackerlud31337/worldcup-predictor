"""
players_build.py  —  per-player contribution ratings for the club predictor
============================================================================
Layer 2 of the club model: rate PLAYERS, so the site can answer "what if
Haaland doesn't play?" — transfers and injuries move a club's expected-lineup
strength instead of waiting for results to drift the team rating.

Data source: Understat (free, no key) — per-player non-penalty xG, xA and
minutes for the big-five leagues, fetched with one POST per league-season
(the same JSON the league pages load) and cached under data_cache/clubs/.

The rating, deliberately simple and explainable:
  * a player's attacking contribution per 90 = (npxG + xA) / minutes * 90,
    pooled over the last two seasons (last season counts double),
  * shrunk toward his POSITION's league-wide average by minutes played
    (600 shrinkage minutes), so a 200-minute wonder doesn't rate like a star,
  * his availability share = minutes / possible minutes last season.

A club's baseline output S0 = Σ rating·share over its squad — by construction
this tracks the club's real xG output, so with everyone available the player
layer changes NOTHING: predictions equal the team model's. Toggling a player
out replaces him with a bench-level stand-in (65% of his position's average)
and rescales the club's attack by the lost share; missing defenders and
keepers additionally leak goals to the opponent (a flat, honest approximation
— defensive value isn't measurable from xG/xA).

Squad assignment is by last season's minutes: a mid-season mover is assigned
to his destination club (identified via his previous season's club). Summer
transfers appear only once the new season's data exists — rebuild during the
season to pick them up.

Usage:
  python3 players_build.py               # build from cache
  python3 players_build.py --download    # force re-download
  python3 players_build.py --names       # team-name reconciliation, then exit
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime

import clubs_build
from clubs_build import CACHE, UNDERSTAT_ALIASES, canon, uscanon

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "web", "clubs", "players_data.js")

UNDERSTAT = "https://understat.com/main/getPlayersStats/"
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]

SHRINK_MINUTES = 600.0      # minutes of position-average belief mixed in
PREV_SEASON_WEIGHT = 0.5    # last season counts double vs the one before
REPL_FACTOR = 0.65          # bench stand-in = 65% of the position average
DEF_LEAK = {"GK": 0.08, "D": 0.06}   # extra goals conceded per missing starter
MAX_PLAYERS = 18            # exported per club
MIN_SHARE = 0.05            # skip cameo appearances

def latest_finished_season() -> int:
    # Understat labels a season by its starting year; data for season N
    # exists once matches are played (Aug of year N).
    now = datetime.now()
    return now.year - 1 if now.month < 8 else now.year


def fetch_league(league: str, season: int, force: bool) -> list[dict]:
    path = os.path.join(CACHE, f"understat_{league}_{season}.json")
    if force or not os.path.exists(path):
        body = urllib.parse.urlencode({"league": league, "season": str(season)}).encode()
        req = urllib.request.Request(
            UNDERSTAT, data=body, method="POST",
            headers={"User-Agent": "Mozilla/5.0",
                     "X-Requested-With": "XMLHttpRequest",
                     "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            import gzip
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
        data = json.loads(raw)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  understat {league} {season}: {len(data.get('players', []))} players")
    with open(path) as f:
        return json.load(f).get("players", [])


def primary_position(pos: str) -> str:
    """Understat positions look like 'F S', 'M C', 'GK', 'D M S'. First
    listed role wins; anything unrecognised (pure 'S' subs) counts as M."""
    for tok in pos.split():
        if tok == "GK":
            return "GK"
        if tok[0] in ("D", "M", "F"):
            return tok[0]
    return "M"


def load_players(force: bool):
    """Merge league records into one record per player id per season."""
    seasons = {}
    last = latest_finished_season()
    for season in (last - 1, last):
        per_player: dict[str, dict] = {}
        for lg in LEAGUES:
            for p in fetch_league(lg, season, force):
                pid = p["id"]
                rec = per_player.setdefault(pid, {
                    "name": p["player_name"], "minutes": 0.0, "prod": 0.0,
                    "pos": p["position"], "teams": defaultdict(float),
                })
                mins = float(p["time"])
                rec["minutes"] += mins
                rec["prod"] += float(p["npxG"]) + float(p["xA"])
                if mins > 0:
                    for t in p["team_title"].split(","):
                        rec["teams"][uscanon(t.strip())] += mins
                if len(p["position"]) > len(rec["pos"]):
                    rec["pos"] = p["position"]
        seasons[season] = per_player
    return seasons, last


def assign_club(rec: dict, prev_rec: dict | None) -> str:
    """A player's current club. Understat comma-joins mid-season movers
    (alphabetically, so order says nothing); the destination is whichever
    club is NOT the one he played for the season before."""
    teams = dict(rec["teams"])
    if len(teams) == 1:
        return next(iter(teams))
    prev_clubs = set(prev_rec["teams"]) if prev_rec else set()
    dest = [t for t in teams if t not in prev_clubs]
    pool = dest or list(teams)
    return max(pool, key=lambda t: teams[t])


def build_ratings(seasons: dict, last: int):
    cur, prev = seasons[last], seasons[last - 1]

    # Pooled per-90 production and position priors.
    rows = []
    for pid, rec in cur.items():
        p = prev.get(pid)
        minutes = rec["minutes"] + PREV_SEASON_WEIGHT * (p["minutes"] if p else 0.0)
        prod = rec["prod"] + PREV_SEASON_WEIGHT * (p["prod"] if p else 0.0)
        if rec["minutes"] <= 0 or minutes <= 0:
            continue
        rows.append({
            "pid": pid, "name": rec["name"],
            "pos": primary_position(rec["pos"]),
            "minutes": minutes,
            "per90": prod / minutes * 90.0,
            "share": min(1.0, rec["minutes"] / (38.0 * 90.0)),
            "club": assign_club(rec, p),
        })

    prior: dict[str, float] = {}
    for pos in ("GK", "D", "M", "F"):
        sel = [r for r in rows if r["pos"] == pos]
        w = sum(r["minutes"] for r in sel)
        prior[pos] = sum(r["per90"] * r["minutes"] for r in sel) / w if w else 0.0

    for r in rows:
        k = SHRINK_MINUTES
        r["rating"] = (r["minutes"] * r["per90"] + k * prior[r["pos"]]) / (r["minutes"] + k)
    return rows, prior


def export(rows: list[dict], prior: dict[str, float], model_teams: set[str], last: int):
    by_club: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_club[r["club"]].append(r)

    unmatched = sorted(set(by_club) - model_teams)
    if unmatched:
        print(f"[warn] player clubs not in the model: {', '.join(unmatched)}")

    teams_js = {}
    for club, players in sorted(by_club.items()):
        if club not in model_teams:
            continue
        players = [p for p in players if p["share"] >= MIN_SHARE]
        players.sort(key=lambda p: p["share"], reverse=True)
        players = players[:MAX_PLAYERS]
        s0 = sum(p["rating"] * p["share"] for p in players)
        if s0 <= 0:
            continue
        teams_js[club] = {
            "s0": round(s0, 4),
            "players": [{
                "n": p["name"],
                "pos": p["pos"],
                "r": round(p["rating"], 4),
                "s": round(p["share"], 3),
            } for p in players],
        }

    data = {
        "config": {
            "repl_factor": REPL_FACTOR,
            "def_leak": DEF_LEAK,
            "prior": {k: round(v, 4) for k, v in prior.items()},
            "season": f"{last}-{(last + 1) % 100:02d}",
            "built": datetime.now().strftime("%Y-%m-%d"),
        },
        "teams": teams_js,
    }
    with open(OUT, "w") as f:
        f.write("// Generated by players_build.py — player ratings for the club predictor.\n")
        f.write("const PLAYER_DATA = " + json.dumps(data, separators=(",", ":"),
                                                    ensure_ascii=False) + ";\n")
    print(f"[export] {len(teams_js)} clubs, "
          f"{sum(len(t['players']) for t in teams_js.values())} players "
          f"-> {os.path.relpath(OUT)}")


def model_team_names() -> set[str]:
    """Team names the club model actually exports (the JS picker list)."""
    path = clubs_build.OUT
    with open(path) as f:
        src = f.read()
    start = src.index("const MODEL_DATA = ") + len("const MODEL_DATA = ")
    data = json.loads(src[start:src.index(";\n", start)])
    return set(data["base"]["teams"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--names", action="store_true",
                    help="report Understat club names that don't match the model")
    args = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    seasons, last = load_players(args.download)
    rows, prior = build_ratings(seasons, last)
    model_teams = model_team_names()

    if args.names:
        clubs = sorted({r["club"] for r in rows})
        bad = [c for c in clubs if c not in model_teams]
        print(f"{len(clubs)} Understat clubs, {len(bad)} unmatched:")
        for c in bad:
            print("  " + c)
        return

    print(f"[ratings] {len(rows)} players, season {last}-{(last + 1) % 100:02d}; "
          f"position priors " +
          ", ".join(f"{k} {v:.3f}" for k, v in prior.items()))
    top = sorted(rows, key=lambda r: r["rating"], reverse=True)[:10]
    for r in top:
        print(f"  {r['name']:<28s} {r['club']:<24s} {r['pos']}  "
              f"rating {r['rating']:.3f}  share {r['share']:.2f}")
    export(rows, prior, model_teams, last)


if __name__ == "__main__":
    main()
