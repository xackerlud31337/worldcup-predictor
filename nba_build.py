"""
nba_build.py  —  fit the NBA model and dump it to web/nba/nba_model_data.js
===========================================================================
Data source: ESPN's public scoreboard API (site.api.espn.com), fetched
month-by-month and cached as one JSON file per season in data_cache/nba/.
The same API is CORS-open, so the site's upcoming-games box reads it live.

The model, mirroring the football/UFC offline-fit / in-browser-predict split:
  * per-franchise Elo fitted chronologically over every game since 2002,
    with home-court advantage, a margin-of-victory K multiplier and
    between-season regression to the mean (FiveThirtyEight-style),
  * a normal margin model (margin ~ N(slope * elo_diff, sigma)) so the win
    probability, fair spread and victory-margin buckets all agree,
  * a totals model from exponentially-weighted team scoring rates
    (points scored / allowed per game, recent games weighted),
  * per-team season records, last-10 form and scoring stats for the eye test.

Re-run whenever you want the site to pick up fresh games:
  python3 nba_build.py --download
"""

from __future__ import annotations

import argparse
import json
import math
import os
import urllib.request
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache", "nba")
OUT = os.path.join(HERE, "web", "nba", "nba_model_data.js")

ESPN = ("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
        "scoreboard?dates={start}-{end}&limit=500")
FIRST_SEASON = 2002          # season start year (2002 -> 2002-03)
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6]   # Oct..Jun

ELO_START = 1500.0
ELO_MEAN = 1505.0            # between-season regression target
SEASON_CARRY = 0.75          # new = carry*old + (1-carry)*mean
ELO_K = 20.0
FORM_HALFLIFE = 25.0         # games, for the scoring-rate EWMA

# Eastern/Western conference by ESPN abbreviation (stable, 30 teams)
EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
        "NY", "NYK", "ORL", "PHI", "TOR", "WSH", "WAS", "NJ"}


# ------------------------------------------------------------- downloading --

def fetch_json(url: str, tries: int = 3) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception:
            if attempt == tries - 1:
                raise
    return {}


def season_cache_path(start_year: int) -> str:
    return os.path.join(CACHE, f"games_{start_year}.json")


def current_season_start() -> int:
    now = datetime.now()
    return now.year if now.month >= 10 else now.year - 1


def download_season(start_year: int) -> list[dict]:
    """One trimmed record per final NBA game (regular season + playoffs)."""
    games = []
    for month in SEASON_MONTHS:
        year = start_year if month >= 10 else start_year + 1
        last_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        url = ESPN.format(start=f"{year}{month:02d}01",
                          end=f"{year}{month:02d}{last_day}")
        for ev in fetch_json(url).get("events", []):
            comp = (ev.get("competitions") or [None])[0]
            if not comp or not comp.get("competitors"):
                continue
            if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue
            stype = ev.get("season", {}).get("type")
            if stype not in (2, 3):       # regular season / playoffs only
                continue
            home = away = None
            for c in comp["competitors"]:
                side = {"id": int(c["team"]["id"]),
                        "abbr": c["team"].get("abbreviation", "?"),
                        "name": c["team"].get("displayName", "?"),
                        "pts": int(c.get("score") or 0)}
                if c["homeAway"] == "home":
                    home = side
                else:
                    away = side
            # franchise ids are 1..30; anything else is All-Star etc.
            if (not home or not away or home["pts"] == 0 or away["pts"] == 0
                    or not (1 <= home["id"] <= 30 and 1 <= away["id"] <= 30)):
                continue
            games.append({
                "date": ev["date"][:10],
                "type": stype,
                "neutral": bool(comp.get("neutralSite")),
                "home": home, "away": away,
            })
    games.sort(key=lambda g: g["date"])
    return games


def load_games(refresh: bool) -> list[dict]:
    os.makedirs(CACHE, exist_ok=True)
    current = current_season_start()
    all_games = []
    for year in range(FIRST_SEASON, current + 1):
        path = season_cache_path(year)
        # the season still being played is refetched on --download
        if not os.path.exists(path) or (refresh and year >= current - 1):
            print(f"  downloading {year}-{str(year + 1)[2:]} ...")
            games = download_season(year)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(games, fh, separators=(",", ":"))
        with open(path, encoding="utf-8") as fh:
            games = json.load(fh)
        for g in games:
            g["season"] = year
        all_games.extend(games)
    all_games.sort(key=lambda g: (g["date"], g["home"]["id"]))
    return all_games


# ----------------------------------------------------------------- fitting --

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def run_elo(games: list[dict], hfa: float, collect_from: int | None = None):
    """Chronological Elo pass. Returns (final ratings, samples) where samples
    are (elo_diff_incl_hfa, home_margin, home_won, season) from collect_from on."""
    elo: dict[int, float] = defaultdict(lambda: ELO_START)
    season_of: dict[int, int] = {}
    samples = []
    for g in games:
        h, a = g["home"]["id"], g["away"]["id"]
        for t in (h, a):
            if season_of.get(t) not in (None, g["season"]):
                elo[t] = SEASON_CARRY * elo[t] + (1 - SEASON_CARRY) * ELO_MEAN
            season_of[t] = g["season"]
        adv = 0.0 if g["neutral"] else hfa
        diff = elo[h] + adv - elo[a]
        margin = g["home"]["pts"] - g["away"]["pts"]
        if collect_from is not None and g["season"] >= collect_from:
            samples.append((diff, margin, 1.0 if margin > 0 else 0.0,
                            g["season"]))
        exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        won = 1.0 if margin > 0 else 0.0
        # FiveThirtyEight margin-of-victory multiplier: blowouts move ratings
        # more, but less so when the favourite was already expected to romp
        winner_diff = diff if margin > 0 else -diff
        mov = ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = ELO_K * mov * (won - exp_home)
        elo[h] += delta
        elo[a] -= delta
    return elo, samples


def fit(games: list[dict]):
    """Grid-search HFA, then fit the margin model and verify on a backtest."""
    seasons = sorted({g["season"] for g in games})
    test_from = seasons[-2]           # last two seasons held out for the report
    warmup = seasons[0] + 2           # let Elo converge before judging

    # HFA: maximise winner accuracy proxy via log-loss of the Elo logistic
    best_hfa, best_ll = 0.0, float("inf")
    for hfa in range(40, 141, 10):
        _, samples = run_elo(games, float(hfa), collect_from=warmup)
        ll = 0.0
        for diff, _, won, _ in samples:
            p = min(max(1.0 / (1.0 + 10.0 ** (-diff / 400.0)), 1e-12), 1 - 1e-12)
            ll -= won * math.log(p) + (1 - won) * math.log(1 - p)
        ll /= len(samples)
        if ll < best_ll:
            best_ll, best_hfa = ll, float(hfa)
    print(f"  home-court advantage {best_hfa:.0f} Elo pts "
          f"(log-loss {best_ll:.4f})")

    elo, samples = run_elo(games, best_hfa, collect_from=warmup)

    # margin ~ N(slope * elo_diff, sigma): least squares through the origin
    train = [s for s in samples if s[3] < test_from]
    test = [s for s in samples if s[3] >= test_from]
    sxx = sum(d * d for d, _, _, _ in train)
    sxy = sum(d * m for d, m, _, _ in train)
    slope = sxy / sxx
    var = sum((m - slope * d) ** 2 for d, m, _, _ in train) / len(train)
    sigma = math.sqrt(var)
    print(f"  margin model: {1/slope:.1f} Elo pts per point of spread, "
          f"sigma {sigma:.1f}")

    # backtest: normal-margin win prob vs the plain Elo logistic
    def report(name, prob):
        ll = hits = 0.0
        for d, _, won, _ in test:
            p = min(max(prob(d), 1e-12), 1 - 1e-12)
            ll -= won * math.log(p) + (1 - won) * math.log(1 - p)
            hits += 1.0 if (p > 0.5) == (won == 1.0) else 0.0
        print(f"    {name:<18} log-loss {ll/len(test):.4f}  "
              f"accuracy {hits/len(test):.1%}")
        return ll / len(test), hits / len(test)

    print(f"  backtest on {len(test)} games from {test_from}-{str(test_from+1)[2:]}:")
    report("Elo logistic", lambda d: 1.0 / (1.0 + 10.0 ** (-d / 400.0)))
    ll_n, acc_n = report("normal margin",
                         lambda d: norm_cdf(slope * d / sigma))
    hit_home = sum(1 for _, _, won, _ in test if won == 1.0) / len(test)
    print(f"    (picking the home team every time: {hit_home:.1%})")

    # refit slope/sigma on ALL rated games for the shipped model
    sxx = sum(d * d for d, _, _, _ in samples)
    sxy = sum(d * m for d, m, _, _ in samples)
    slope = sxy / sxx
    sigma = math.sqrt(sum((m - slope * d) ** 2
                          for d, m, _, _ in samples) / len(samples))
    return elo, best_hfa, slope, sigma, acc_n, len(test)


def team_profiles(games: list[dict], elo: dict[int, float]):
    """Latest-season records plus EWMA scoring rates across recent games."""
    latest = max(g["season"] for g in games)
    decay = 0.5 ** (1.0 / FORM_HALFLIFE)
    T = defaultdict(lambda: {
        "abbr": "?", "name": "?", "w": 0, "l": 0, "hw": 0, "hl": 0,
        "aw": 0, "al": 0, "pts": 0, "opp": 0, "n": 0,
        "off_w": 0.0, "def_w": 0.0, "wsum": 0.0,
        "res": [], "last": None,
    })
    lg_pts = lg_n = 0
    for g in games:
        for side, other, is_home in ((g["home"], g["away"], True),
                                     (g["away"], g["home"], False)):
            t = T[side["id"]]
            t["abbr"], t["name"] = side["abbr"], side["name"]
            t["off_w"] = t["off_w"] * decay + side["pts"]
            t["def_w"] = t["def_w"] * decay + other["pts"]
            t["wsum"] = t["wsum"] * decay + 1.0
            t["last"] = g["date"]
            won = side["pts"] > other["pts"]
            t["res"].append(1 if won else 0)
            if g["season"] == latest:
                t["n"] += 1
                t["pts"] += side["pts"]
                t["opp"] += other["pts"]
                if won:
                    t["w"] += 1
                    t["hw" if is_home else "aw"] += 1
                else:
                    t["l"] += 1
                    t["hl" if is_home else "al"] += 1
        if g["season"] == latest:
            lg_pts += g["home"]["pts"] + g["away"]["pts"]
            lg_n += 2

    league_avg = lg_pts / lg_n
    teams = {}
    for tid, t in T.items():
        if t["n"] == 0:          # franchise gone (relocated ids stay 1..30)
            continue
        streak = 0
        for r in reversed(t["res"]):
            if streak == 0:
                streak = 1 if r else -1
            elif (streak > 0) == bool(r):
                streak += 1 if r else -1
            else:
                break
        teams[t["name"]] = {
            "id": tid,
            "abbr": t["abbr"],
            "conf": "East" if t["abbr"] in EAST else "West",
            "elo": round(elo[tid], 1),
            "off": round(t["off_w"] / t["wsum"], 2),
            "def": round(t["def_w"] / t["wsum"], 2),
            "rec": [t["w"], t["l"]],
            "home_rec": [t["hw"], t["hl"]],
            "away_rec": [t["aw"], t["al"]],
            "ppg": round(t["pts"] / t["n"], 1),
            "oppg": round(t["opp"] / t["n"], 1),
            "l10": sum(t["res"][-10:]),
            "streak": streak,
            "last": t["last"],
        }
    return teams, league_avg, latest


def build(refresh: bool):
    games = load_games(refresh)
    seasons = sorted({g["season"] for g in games})
    print(f"  {len(games)} games across {len(seasons)} seasons "
          f"({seasons[0]}-{str(seasons[0]+1)[2:]} to "
          f"{seasons[-1]}-{str(seasons[-1]+1)[2:]})")

    elo, hfa, slope, sigma, acc, n_test = fit(games)
    teams, league_avg, latest = team_profiles(games, elo)

    # total-points spread around the ratings-based expectation
    resid2 = n = 0
    decayed = {}
    for g in games:
        if g["season"] >= latest - 2:
            th, ta = decayed.get(g["home"]["id"]), decayed.get(g["away"]["id"])
            if th and ta:
                exp_total = (th[0] + ta[1] + ta[0] + th[1]) / 2.0
                resid2 += (g["home"]["pts"] + g["away"]["pts"] - exp_total) ** 2
                n += 1
        for side, other in ((g["home"], g["away"]), (g["away"], g["home"])):
            o, d, w = decayed.get(side["id"], (side["pts"], other["pts"], 0.0))
            dc = 0.5 ** (1.0 / FORM_HALFLIFE)
            wn = w * dc + 1.0
            decayed[side["id"]] = (
                (o * w * dc + side["pts"]) / wn,
                (d * w * dc + other["pts"]) / wn,
                wn)
    sigma_total = math.sqrt(resid2 / n)
    print(f"  totals model: league average {league_avg:.1f} pts/team, "
          f"sigma {sigma_total:.1f} on the game total")

    bundle = {
        "config": {
            "hfa": hfa,
            "slope": round(slope, 6),
            "sigma": round(sigma, 2),
            "sigma_total": round(sigma_total, 2),
            "league_avg": round(league_avg, 2),
            "season": f"{latest}-{str(latest + 1)[2:]}",
            "built": datetime.now().strftime("%Y-%m-%d"),
            "accuracy_note": f"{acc:.1%} winner accuracy on the last "
                             f"{n_test} games held out from fitting",
        },
        "teams": teams,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    js = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("// Generated by nba_build.py — fitted NBA model for the static site.\n")
        fh.write(f"const NBA_DATA = {js};\n")
    print(f"  {len(teams)} teams -> {os.path.getsize(OUT)/1024:.0f} KB "
          f"written to {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="refresh the cached current-season games first")
    args = ap.parse_args()
    build(args.download)
