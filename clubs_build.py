"""
clubs_build.py  —  fit the club-football model and dump it to web/clubs/club_model_data.js
===========================================================================================
Premier League + Champions League predictor (Layer 1: team strengths).

Data sources, both free and key-less:
  * football-data.co.uk season CSVs for the domestic leagues — the big five
    (EPL, La Liga, Bundesliga, Serie A, Ligue 1) plus six feeder leagues
    (NED/POR/BEL/TUR/GRE/SCO) so Champions League regulars like Porto or
    Celtic have domestic data behind their rating,
  * ESPN's public scoreboard API for the Champions League itself — the same
    API nba_build.py uses, fetched month-by-month and cached per season.

The UCL matches are what tie the leagues onto one scale: they are the only
cross-league observations, so Arsenal and Bayern become comparable through
them. Everything is cached under data_cache/clubs/.

The model is the same Dixon-Coles machinery as the international predictor
(model.fit / predict.predict_match are imported, not duplicated), with two
club-specific choices:
  * half-life of ONE year instead of four — clubs play ~50 matches a season
    and squads turn over every transfer window, so old results decay fast,
  * goal_scale = 1.0 — the international fit needs a 1.2 boost because cagey
    qualifiers drag its average down; a club fit has no such bias.

An Elo rating (K=20, fitted draw model) is computed on the same matches and
serves both as the naive baseline the DC model must beat in the back-test and
as the thin-data prior for teams with few weighted matches (UCL qualifiers).

Back-test: fit only on matches before the last season, predict that season's
Premier League + Champions League, report log-loss / Brier / accuracy.

Usage:
  python3 clubs_build.py                # build from cache (downloads what's missing)
  python3 clubs_build.py --download     # force re-download of the current season
  python3 clubs_build.py --names        # report team-name reconciliation, then exit
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import unicodedata
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

import model
import predict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache", "clubs")
OUT = os.path.join(HERE, "web", "clubs", "club_model_data.js")

FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
ESPN_SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
                   "{slug}/scoreboard?dates={start}-{end}&limit=500")
ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams"

FIRST_SEASON = 2017          # season start year (2017 -> 2017-18)
HALF_LIFE_DAYS = 365.0       # clubs play weekly; ratings should live in the present
UCL_WEIGHT = 1.25            # UCL games also carry the cross-league calibration
CLUB_RIDGE = 0.2
ELO_K = 20.0                 # club Elo is less reactive than international (more games)

# The "big five" leagues are shown in the team picker; the feeder leagues are
# fitted (so their UCL clubs have domestic context) but only their UCL
# participants appear in the picker — a mid-table Greek club has too few
# cross-league links for its rating to mean much next to an EPL side.
BIG5 = {
    "E0":  ("eng.1", "Premier League"),
    "SP1": ("esp.1", "La Liga"),
    "D1":  ("ger.1", "Bundesliga"),
    "I1":  ("ita.1", "Serie A"),
    "F1":  ("fra.1", "Ligue 1"),
}
FEEDER = {
    "N1":  ("ned.1", "Eredivisie"),
    "P1":  ("por.1", "Primeira Liga"),
    "B1":  ("bel.1", "Belgian Pro League"),
    "T1":  ("tur.1", "Super Lig"),
    "G1":  ("gre.1", "Super League Greece"),
    "SC0": ("sco.1", "Scottish Premiership"),
}
LEAGUES = {**BIG5, **FEEDER}
UCL_SLUG = "uefa.champions"

# --------------------------------------------------------------- team names --
# Canonical names follow ESPN's displayName (the fixtures box and the logos
# come from ESPN, so this keeps the web layer free of name mapping). The alias
# table folds football-data.co.uk's short names — and ESPN's own variants —
# onto the canonical spelling. `--names` reports anything left unmatched.

ALIASES = {
    # --- Premier League (football-data -> ESPN) ---
    "Man United": "Manchester United", "Man City": "Manchester City",
    "Newcastle": "Newcastle United", "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur", "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers", "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton & Hove Albion", "Leicester": "Leicester City",
    "Leeds": "Leeds United", "Ipswich": "Ipswich Town", "Luton": "Luton Town",
    "Norwich": "Norwich City", "West Brom": "West Bromwich Albion",
    "Huddersfield": "Huddersfield Town", "Cardiff": "Cardiff City",
    "Stoke": "Stoke City", "Swansea": "Swansea City",
    # --- La Liga ---
    "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atlético Madrid",
    "Atletico Madrid": "Atlético Madrid",
    "Espanol": "Espanyol", "Sociedad": "Real Sociedad", "Betis": "Real Betis",
    "Vallecano": "Rayo Vallecano", "Celta": "Celta Vigo",
    "Alaves": "Alavés", "Leganes": "Leganés", "Almeria": "Almería",
    "Cadiz": "Cádiz", "Valladolid": "Real Valladolid", "Oviedo": "Real Oviedo",
    "La Coruna": "Deportivo La Coruña",
    # --- Bundesliga ---
    "Dortmund": "Borussia Dortmund", "Leverkusen": "Bayer Leverkusen",
    "Ein Frankfurt": "Eintracht Frankfurt", "M'gladbach": "Borussia Mönchengladbach",
    "FC Koln": "FC Cologne", "Mainz": "1. FSV Mainz 05", "Mainz 05": "1. FSV Mainz 05",
    "Union Berlin": "1. FC Union Berlin", "Freiburg": "SC Freiburg",
    "Stuttgart": "VfB Stuttgart", "Wolfsburg": "VfL Wolfsburg",
    "Augsburg": "FC Augsburg", "Heidenheim": "1. FC Heidenheim 1846",
    "Hoffenheim": "TSG Hoffenheim", "Bochum": "VfL Bochum",
    "Hertha": "Hertha Berlin", "Bielefeld": "Arminia Bielefeld",
    "Greuther Furth": "Greuther Fürth", "Fortuna Dusseldorf": "Fortuna Düsseldorf",
    "Hamburg": "Hamburg SV", "St Pauli": "FC St. Pauli", "Darmstadt": "SV Darmstadt 98",
    "Hannover": "Hannover 96", "Nurnberg": "1. FC Nürnberg", "Paderborn": "SC Paderborn 07",
    "Schalke": "Schalke 04", "Werder": "Werder Bremen",
    # --- Serie A ---
    "Milan": "AC Milan", "Inter": "Internazionale", "Inter Milan": "Internazionale",
    "Roma": "AS Roma", "Verona": "Hellas Verona", "Spal": "SPAL",
    # --- Ligue 1 ---
    "Paris SG": "Paris Saint-Germain", "St Etienne": "Saint-Étienne",
    "Clermont": "Clermont Foot", "Ajaccio": "AC Ajaccio",
    "AJ Auxerre": "Auxerre", "Le Havre AC": "Le Havre",
    # --- feeder leagues (football-data -> ESPN) ---
    "PSV Eindhoven": "PSV", "For Sittard": "Fortuna Sittard",
    "Nijmegen": "NEC Nijmegen", "Waalwijk": "RKC Waalwijk",
    "Zwolle": "PEC Zwolle", "Groningen": "FC Groningen", "Twente": "FC Twente",
    "Utrecht": "FC Utrecht", "Heerenveen": "SC Heerenveen", "Emmen": "FC Emmen",
    "Volendam": "FC Volendam", "Den Haag": "ADO Den Haag", "Graafschap": "De Graafschap",
    "Go Ahead Eagles": "Go Ahead Eagles", "Sittard": "Fortuna Sittard",
    "Sp Lisbon": "Sporting CP", "Sp Braga": "SC Braga", "Guimaraes": "Vitória SC",
    "Ferreira": "Paços de Ferreira", "Famalicao": "Famalicão", "Vizela": "FC Vizela",
    "Gil Vicente": "Gil Vicente", "Santa Clara": "Santa Clara",
    "Setubal": "Vitória Setúbal", "Pacos Ferreira": "Paços de Ferreira",
    "St Truiden": "Sint-Truiden", "St. Truiden": "Sint-Truiden",
    "Genk": "Racing Genk", "Rennes": "Stade Rennais",
    "Standard": "Standard Liège", "Waregem": "Zulte Waregem",
    "St Gilloise": "Union St.-Gilloise", "Union SG": "Union St.-Gilloise",
    "Buyuksehyr": "İstanbul Başakşehir", "Basaksehir": "İstanbul Başakşehir",
    "Istanbul Basaksehir": "İstanbul Başakşehir",
    "Fenerbahce": "Fenerbahçe", "Besiktas": "Beşiktaş",
    "Olympiakos": "Olympiacos", "PAOK": "PAOK Salonika",
    "Celtic": "Celtic", "Rangers": "Rangers", "Hearts": "Heart of Midlothian",
    # --- ESPN UCL names that should fold onto clubs we already have ---
    "Ajax": "Ajax Amsterdam", "AFC Ajax": "Ajax Amsterdam",
    "Feyenoord": "Feyenoord Rotterdam", "Monaco": "AS Monaco",
    "Porto": "FC Porto", "AEK": "AEK Athens",
    "Club Brugge KV": "Club Brugge",
    "Sporting Lisbon": "Sporting CP", "Braga": "SC Braga",
    "Atletico de Madrid": "Atlético Madrid",
}


# Understat spellings that differ from both football-data and ESPN.
UNDERSTAT_ALIASES = {
    "RasenBallsport Leipzig": "RB Leipzig",
    "Borussia M.Gladbach": "Borussia Mönchengladbach",
    "Paris Saint Germain": "Paris Saint-Germain",
    "Parma Calcio 1913": "Parma",
    "Atletico Madrid": "Atlético Madrid",
    "Alaves": "Alavés",
    "Leganes": "Leganés",
    "Cadiz": "Cádiz",
    "Almeria": "Almería",
    "Mainz 05": "1. FSV Mainz 05",
    "Union Berlin": "1. FC Union Berlin",
    "Freiburg": "SC Freiburg",
    "Augsburg": "FC Augsburg",
    "Wolfsburg": "VfL Wolfsburg",
    "Hoffenheim": "TSG Hoffenheim",
    "Bochum": "VfL Bochum",
    "Darmstadt": "SV Darmstadt 98",
    "Heidenheim": "1. FC Heidenheim 1846",
    "FC Heidenheim": "1. FC Heidenheim 1846",
    "Hamburger SV": "Hamburg SV",
    "St. Pauli": "FC St. Pauli",
    "Fortuna Duesseldorf": "Fortuna Düsseldorf",
    "Inter": "Internazionale",
    "Verona": "Hellas Verona",
    "Leeds": "Leeds United",
    "Luton": "Luton Town",
    "Norwich": "Norwich City",
    "Saint-Etienne": "Saint-Étienne",
}


def uscanon(name: str) -> str:
    """Canonical name for an Understat team spelling."""
    return canon(UNDERSTAT_ALIASES.get(name, name))


def _norm(name: str) -> str:
    """Accent-, case- and punctuation-insensitive key for fuzzy matching."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


_ALIAS_NORM = { _norm(k): v for k, v in ALIASES.items() }
_CANON_NORM: dict[str, str] = {}   # normalised canonical name -> canonical name


def canon(name: str) -> str:
    """Map any source spelling onto the canonical (ESPN) team name."""
    name = name.strip()
    target = ALIASES.get(name) or _ALIAS_NORM.get(_norm(name))
    if target is not None:
        # Register the canonical spelling too, so later accent/punctuation
        # variants of the *target* (e.g. ESPN's "Standard Liege" vs the
        # canonical "Standard Liège") fold onto it via the fuzzy key.
        _CANON_NORM.setdefault(_norm(target), target)
        return target
    key = _norm(name)
    if key in _CANON_NORM:
        return _CANON_NORM[key]
    _CANON_NORM[key] = name
    return name


# ------------------------------------------------------------- downloading --

def fetch(url: str, tries: int = 3) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            if attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
    return b""


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start() -> int:
    now = datetime.now()
    return now.year if now.month >= 8 else now.year - 1


def load_league_season(code: str, start_year: int, force: bool) -> pd.DataFrame:
    """One league season from football-data.co.uk, cached as the raw CSV."""
    path = os.path.join(CACHE, f"{code}_{start_year}.csv")
    if force or not os.path.exists(path):
        raw = fetch(FD_URL.format(season=season_code(start_year), code=code))
        with open(path, "wb") as f:
            f.write(raw)
    if os.path.getsize(path) < 200:      # 404 or not-yet-started season
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, format="mixed"),
        "home_team": df["HomeTeam"].map(canon),
        "away_team": df["AwayTeam"].map(canon),
        "home_goals": df["FTHG"].astype(int),
        "away_goals": df["FTAG"].astype(int),
    })
    out["neutral"] = False
    out["comp"] = code

    # Closing odds -> devigged market probabilities (the benchmark every model
    # should be measured against). Row-wise preference Pinnacle > Bet365 >
    # market average — bookmaker columns come and go mid-season in these files.
    out["mkt_h"] = out["mkt_d"] = out["mkt_a"] = np.nan
    for pre in ("PS", "B365", "Avg"):
        cols = [pre + "H", pre + "D", pre + "A"]
        if not all(c in df.columns for c in cols):
            continue
        inv = 1.0 / df[cols].apply(pd.to_numeric, errors="coerce")
        tot = inv.sum(axis=1)
        need = out["mkt_h"].isna().to_numpy()
        for i, side in enumerate(("mkt_h", "mkt_d", "mkt_a")):
            probs = (inv[cols[i]] / tot).to_numpy()
            out.loc[need & pd.notna(probs), side] = probs[need & pd.notna(probs)]
    return out


UCL_LOGOS: dict[str, str] = {}    # harvested from scoreboard events while parsing


def load_ucl_season(start_year: int, force: bool) -> pd.DataFrame:
    """One Champions League season from ESPN (incl. qualifiers), cached as JSON."""
    path = os.path.join(CACHE, f"ucl_{start_year}.json")
    if force or not os.path.exists(path):
        games = []
        for month in (7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6):
            year = start_year if month >= 7 else start_year + 1
            last = [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30,
                    31, 31, 30, 31, 30, 31][month - 1]
            url = ESPN_SCOREBOARD.format(slug=UCL_SLUG,
                                         start=f"{year}{month:02d}01",
                                         end=f"{year}{month:02d}{last}")
            raw = fetch(url)
            if not raw:
                continue
            data = json.loads(raw)
            for ev in data.get("events", []):
                compn = ev["competitions"][0]
                if not ev["status"]["type"]["completed"]:
                    continue
                sides = {c["homeAway"]: c for c in compn["competitors"]}
                if "home" not in sides or "away" not in sides:
                    continue
                games.append({
                    "date": ev["date"][:10],
                    "home": sides["home"]["team"]["displayName"],
                    "away": sides["away"]["team"]["displayName"],
                    "hg": int(sides["home"].get("score", 0)),
                    "ag": int(sides["away"].get("score", 0)),
                    "neutral": bool(compn.get("neutralSite") or False),
                    "logos": {sides[s]["team"]["displayName"]:
                              sides[s]["team"].get("logo", "") for s in sides},
                })
        with open(path, "w") as f:
            json.dump(games, f)
        print(f"  UCL {start_year}-{(start_year + 1) % 100:02d}: {len(games)} matches")

    with open(path) as f:
        games = json.load(f)
    if not games:
        return pd.DataFrame()
    for g in games:
        for name, logo in g.get("logos", {}).items():
            if logo:
                UCL_LOGOS.setdefault(canon(name), logo)
    return pd.DataFrame({
        "date": pd.to_datetime([g["date"] for g in games]),
        "home_team": [canon(g["home"]) for g in games],
        "away_team": [canon(g["away"]) for g in games],
        "home_goals": [g["hg"] for g in games],
        "away_goals": [g["ag"] for g in games],
        "neutral": [g["neutral"] for g in games],
        "comp": "UCL",
    })


# Second divisions, fetched ONLY for logos: ESPN's league team lists reflect
# the current season, so clubs relegated last season drop out of the top list.
LOGO_EXTRA_SLUGS = ["eng.2", "esp.2", "ger.2", "ita.2", "fra.2"]


def load_espn_team_logos(force: bool) -> dict[str, str]:
    """displayName -> logo URL for every club in the covered leagues."""
    logos: dict[str, str] = {}
    slugs = [slug for slug, _ in LEAGUES.values()] + LOGO_EXTRA_SLUGS
    for slug in slugs:
        path = os.path.join(CACHE, f"teams_{slug}.json")
        if force or not os.path.exists(path):
            raw = fetch(ESPN_TEAMS.format(slug=slug))
            with open(path, "wb") as f:
                f.write(raw or b"{}")
        try:
            with open(path) as f:
                data = json.load(f)
            entries = data["sports"][0]["leagues"][0]["teams"]
        except Exception:
            continue
        for e in entries:
            t = e["team"]
            logo = (t.get("logos") or [{}])[0].get("href", "")
            if logo:
                logos.setdefault(canon(t["displayName"]), logo)
    return logos


def load_scoreboard_logos(last_season: int, force: bool) -> dict[str, str]:
    """Logos harvested from each big-five league's final month of last season —
    the team-list endpoints reflect *next* season, so mid-offseason they lose
    relegated clubs and can be incomplete."""
    logos: dict[str, str] = {}
    for slug, _ in BIG5.values():
        path = os.path.join(CACHE, f"logos_{slug}_{last_season}.json")
        if force or not os.path.exists(path):
            url = ESPN_SCOREBOARD.format(slug=slug,
                                         start=f"{last_season + 1}0501",
                                         end=f"{last_season + 1}0531")
            raw = fetch(url)
            harvest: dict[str, str] = {}
            if raw:
                for ev in json.loads(raw).get("events", []):
                    for c in ev["competitions"][0]["competitors"]:
                        t = c["team"]
                        if t.get("logo"):
                            harvest.setdefault(t["displayName"], t["logo"])
            with open(path, "w") as f:
                json.dump(harvest, f)
        with open(path) as f:
            for name, logo in json.load(f).items():
                logos.setdefault(canon(name), logo)
    return logos


US_LEAGUE = {"E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga",
             "I1": "Serie_A", "F1": "Ligue_1"}
US_DATA = "https://understat.com/getLeagueData/{lg}/{season}"


def fetch_us_matches(code: str, start_year: int, force: bool) -> list[dict]:
    """Per-match team xG from Understat (big-five leagues, 2014+)."""
    lg = US_LEAGUE[code]
    path = os.path.join(CACHE, f"usmatches_{lg}_{start_year}.json")
    if force or not os.path.exists(path):
        req = urllib.request.Request(
            US_DATA.format(lg=lg, season=start_year),
            headers={"User-Agent": "Mozilla/5.0",
                     "X-Requested-With": "XMLHttpRequest"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                import gzip
                raw = gzip.decompress(raw)
            dates = json.loads(raw).get("dates", [])
        except Exception as e:
            print(f"  [warn] understat {lg} {start_year}: {e}")
            dates = []
        games = [{
            "date": d["datetime"][:10],
            "home": d["h"]["title"], "away": d["a"]["title"],
            "xg_h": float(d["xG"]["h"]), "xg_a": float(d["xG"]["a"]),
        } for d in dates if d.get("isResult") and d.get("xG", {}).get("h") is not None]
        with open(path, "w") as f:
            json.dump(games, f)
        if games:
            print(f"  understat {lg} {start_year}: {len(games)} matches with xG")
    with open(path) as f:
        return json.load(f)


def merge_xg(df: pd.DataFrame, force: bool) -> pd.DataFrame:
    """Attach Understat team xG to the big-five league matches (nearest-date
    match on the canonical home/away pair, tolerating small date offsets)."""
    lookup: dict[tuple[str, str], list] = {}
    cur = current_season_start()
    for code in US_LEAGUE:
        for start in range(FIRST_SEASON, cur + 1):
            for g in fetch_us_matches(code, start, force and start >= cur - 1):
                key = (uscanon(g["home"]), uscanon(g["away"]))
                lookup.setdefault(key, []).append(
                    (pd.Timestamp(g["date"]), g["xg_h"], g["xg_a"]))

    xg_h = np.full(len(df), np.nan)
    xg_a = np.full(len(df), np.nan)
    big5 = df["comp"].isin(US_LEAGUE).to_numpy()
    dates = df["date"].to_numpy()
    homes = df["home_team"].to_numpy()
    aways = df["away_team"].to_numpy()
    for i in np.flatnonzero(big5):
        cands = lookup.get((homes[i], aways[i]))
        if not cands:
            continue
        best = min(cands, key=lambda c: abs((c[0] - dates[i]).days))
        if abs((best[0] - dates[i]).days) <= 3:
            xg_h[i], xg_a[i] = best[1], best[2]
    out = df.assign(xg_home=xg_h, xg_away=xg_a)
    cov = out.loc[big5, "xg_home"].notna().mean() if big5.any() else 0.0
    print(f"[xg] Understat coverage of big-five matches: {cov:.1%}")
    return out


def load_all(force_current: bool) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    cur = current_season_start()
    frames = []
    for start in range(FIRST_SEASON, cur + 1):
        force = force_current and start >= cur - 1
        for code in LEAGUES:
            frames.append(load_league_season(code, start, force))
        frames.append(load_ucl_season(start, force))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return merge_xg(df, force_current)


# -------------------------------------------------------------------- elo --

def club_elo(df: pd.DataFrame) -> dict[str, float]:
    """Chronological Elo over every club match; baseline + thin-data prior."""
    ratings: dict[str, float] = {}
    for home, away, hg, ag, neutral in zip(
        df["home_team"], df["away_team"], df["home_goals"], df["away_goals"], df["neutral"]
    ):
        rh = ratings.get(home, 1500.0)
        ra = ratings.get(away, 1500.0)
        adv = 0.0 if neutral else 65.0
        exp_home = 1.0 / (1.0 + 10.0 ** ((ra - (rh + adv)) / 400.0))
        score = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        g = abs(hg - ag)
        mult = 1.0 if g <= 1 else (1.5 if g == 2 else (11.0 + g) / 8.0)
        delta = ELO_K * mult * (score - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def fit_draw_params(df: pd.DataFrame, ratings: dict[str, float]) -> tuple[float, float]:
    """Vectorised grid-search for the Elo draw model on the given matches."""
    rh = df["home_team"].map(lambda t: ratings.get(t, 1500.0)).to_numpy()
    ra = df["away_team"].map(lambda t: ratings.get(t, 1500.0)).to_numpy()
    adv = np.where(df["neutral"].to_numpy(dtype=bool), 0.0, 65.0)
    dr = (rh + adv) - ra
    e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    outcome = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]],
        [0, 1], default=2)

    best, best_ll = (0.28, 220.0), np.inf
    for base in np.linspace(0.15, 0.38, 12):
        for decay in np.linspace(120.0, 400.0, 12):
            p_draw = base * np.exp(-np.abs(dr) / decay)
            p_home = np.clip(e_home - 0.5 * p_draw, 1e-6, None)
            p_away = np.clip(1.0 - p_home - p_draw, 1e-6, None)
            P = np.stack([p_home, p_draw, p_away], axis=1)
            P /= P.sum(axis=1, keepdims=True)
            ll = -np.log(P[np.arange(len(P)), outcome]).sum()
            if ll < best_ll:
                best_ll, best = ll, (float(base), float(decay))
    return best


# -------------------------------------------------------------------- fit --

def add_weights(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    age = (asof - df["date"]).dt.days.clip(lower=0)
    decay = 0.5 ** (age / HALF_LIFE_DAYS)
    comp_w = np.where(df["comp"] == "UCL", UCL_WEIGHT, 1.0)
    return df.assign(weight=decay * comp_w)


def dc_fit(df: pd.DataFrame, asof: pd.Timestamp, xg_alpha: float | None = None):
    """Dixon-Coles fit on matches before `asof`. With xg_alpha, matches that
    have Understat xG use the blended target alpha*xG + (1-alpha)*goals —
    xG strips finishing luck, and unlike the international --xg mode it
    covers EVERY big-five league match, not a handful of tournaments."""
    past = add_weights(df[df["date"] < asof], asof)
    if xg_alpha is not None:
        has = past["xg_home"].notna() & past["xg_away"].notna()
        past = past.assign(
            home_target=np.where(has, xg_alpha * past["xg_home"]
                                 + (1 - xg_alpha) * past["home_goals"],
                                 past["home_goals"]),
            away_target=np.where(has, xg_alpha * past["xg_away"]
                                 + (1 - xg_alpha) * past["away_goals"],
                                 past["away_goals"]),
        )
    m = model.fit(past, ridge=CLUB_RIDGE)
    m.goal_scale = 1.0   # club data has no qualifier drag; see module docstring
    return m


def fit_bundle(df: pd.DataFrame, asof: pd.Timestamp):
    """Fit DC + Elo on matches strictly before `asof`; returns (model, elo, draw)."""
    past = df[df["date"] < asof]
    m = dc_fit(df, asof)
    ratings = club_elo(past)
    recent = past[past["date"] >= asof - pd.Timedelta(days=730)]
    draw_params = fit_draw_params(recent, ratings)
    return m, ratings, draw_params


# --------------------------------------------------------------- back-test --

def backtest(df: pd.DataFrame, test_season: int, verbose: bool = True,
             xg_alphas: tuple = (0.4, 0.6, 0.8)) -> dict:
    cutoff = pd.Timestamp(f"{test_season}-08-01")
    m, ratings, draw_params = fit_bundle(df, cutoff)
    have_xg = df["xg_home"].notna().any()
    xg_models = {a: dc_fit(df, cutoff, a) for a in xg_alphas} if have_xg else {}

    import elo as elo_mod
    test = df[(df["date"] >= cutoff) & df["comp"].isin(["E0", "UCL"])]
    rows = []
    for _, r in test.iterrows():
        preds = {}
        for label, mod in [("dc", m)] + [(f"xg{a}", xm) for a, xm in xg_models.items()]:
            p = predict.predict_match(mod, r["home_team"], r["away_team"],
                                      neutral=bool(r["neutral"]), elo_ratings=ratings)
            preds[label] = (p["p_home_win"], p["p_draw"], p["p_away_win"])
            if label == "dc":
                tot = p["lambda_home"] + p["lambda_away"]
        rh, ra = ratings.get(r["home_team"], 1500.0), ratings.get(r["away_team"], 1500.0)
        preds["elo"] = elo_mod.wdl_probs(rh, ra, bool(r["neutral"]), *draw_params)
        if pd.notna(r.get("mkt_h")):
            preds["market"] = (r["mkt_h"], r["mkt_d"], r["mkt_a"])
        outcome = 0 if r["home_goals"] > r["away_goals"] else (
            1 if r["home_goals"] == r["away_goals"] else 2)
        rows.append({"comp": r["comp"], "preds": preds, "outcome": outcome,
                     "pred_tot": tot, "act_tot": r["home_goals"] + r["away_goals"]})

    def metrics(sel, label):
        ll = br = acc = 0.0
        for row in sel:
            p, o = row["preds"][label], row["outcome"]
            ll -= np.log(max(p[o], 1e-12))
            br += sum((p[k] - (1.0 if k == o else 0.0)) ** 2 for k in range(3))
            acc += 1.0 if int(np.argmax(p)) == o else 0.0
        n = len(sel)
        return {"log_loss": ll / n, "brier": br / n, "accuracy": acc / n, "n": n}

    model_labels = (["dc"] + [f"xg{a}" for a in xg_models] + ["elo"])
    res = {seg: {lab: metrics(sel, lab) for lab in model_labels}
           for seg, sel in (("all", rows),
                            ("epl", [r for r in rows if r["comp"] == "E0"]),
                            ("ucl", [r for r in rows if r["comp"] == "UCL"]))}

    # Market benchmark: only matches with closing odds (EPL); score every
    # model on that same subset so the comparison is apples-to-apples.
    priced = [r for r in rows if "market" in r["preds"]]
    res["priced"] = {lab: metrics(priced, lab) for lab in model_labels + ["market"]}         if priced else {}

    best_alpha = None
    if xg_models:
        best_alpha = min(xg_models, key=lambda a: res["all"][f"xg{a}"]["log_loss"])
        if res["all"][f"xg{best_alpha}"]["log_loss"] >= res["all"]["dc"]["log_loss"]:
            best_alpha = None   # xG must EARN its place in the export
    res["best_alpha"] = best_alpha

    if verbose:
        names = {"dc": "Dixon-Coles", "elo": "Elo baseline", "market": "Bookmakers",
                 **{f"xg{a}": f"DC + xG a={a}" for a in xg_models}}
        print(f"\nBack-test — {test_season}-{(test_season + 1) % 100:02d} "
              f"Premier League + Champions League ({len(rows)} matches)")
        print(f"{'segment':<9}{'model':<16}{'log-loss':>9}{'brier':>8}{'accuracy':>10}")
        for seg in ("all", "epl", "ucl", "priced"):
            if not res.get(seg):
                continue
            for lab, r in res[seg].items():
                print(f"{seg:<9}{names[lab]:<16}{r['log_loss']:>9.4f}{r['brier']:>8.4f}"
                      f"{r['accuracy']:>9.1%}  (n={r['n']})")
        print(f"goals/game: predicted {np.mean([r['pred_tot'] for r in rows]):.2f} "
              f"vs actual {np.mean([float(r['act_tot']) for r in rows]):.2f}")
        if best_alpha is not None:
            print(f"[xg] alpha={best_alpha} beats the goals-only fit -> exported as the xG bundle")
        elif have_xg:
            print("[xg] no alpha beat the goals-only fit -> xG bundle NOT exported")
    return res


# ----------------------------------------------------------------- export --

def picker_teams(df: pd.DataFrame, last_season: int) -> dict[str, str]:
    """Canonical name -> league label, for teams shown in the web picker:
    every big-five club from the last season, plus every club that played
    UCL (proper) in the last two seasons."""
    since = pd.Timestamp(f"{last_season}-08-01")
    recent = df[df["date"] >= since]
    chosen: dict[str, str] = {}
    for code, (_, label) in BIG5.items():
        sub = recent[recent["comp"] == code]
        for t in set(sub["home_team"]) | set(sub["away_team"]):
            chosen[t] = label
    # feeder-league labels for their UCL clubs (nicer than "Champions League")
    feeder_league: dict[str, str] = {}
    for code, (_, label) in FEEDER.items():
        sub = df[df["comp"] == code]
        for t in set(sub["home_team"]) | set(sub["away_team"]):
            feeder_league[t] = label
    ucl_since = pd.Timestamp(f"{last_season - 1}-09-01")   # skip July qualifiers
    ucl = df[(df["comp"] == "UCL") & (df["date"] >= ucl_since)]
    for t in set(ucl["home_team"]) | set(ucl["away_team"]):
        chosen.setdefault(t, feeder_league.get(t, "Champions League"))
    return chosen


def export(df: pd.DataFrame, note: str, xg_alpha: float | None = None) -> None:
    now = pd.Timestamp(datetime.now().date())
    asof = now + pd.Timedelta(days=1)
    m, ratings, draw_params = fit_bundle(df, asof)
    last = current_season_start()
    if not (df["date"] >= pd.Timestamp(f"{last}-08-01")).any():
        last -= 1      # new season not underway yet: picker reflects last season
    chosen = picker_teams(df, last)

    logos = load_espn_team_logos(force=False)
    logos.update(UCL_LOGOS)
    for name, logo in load_scoreboard_logos(last, force=False).items():
        logos.setdefault(name, logo)

    teams_js, elo_js, meta_js = {}, {}, {}
    missing_logo = []
    for t in sorted(chosen):
        a, d, w = m.strength(t)
        teams_js[t] = {"a": round(a, 6), "d": round(d, 6), "w": round(w, 3)}
        elo_js[t] = round(ratings.get(t, 1500.0), 1)
        meta_js[t] = {"league": chosen[t], "logo": logos.get(t, "")}
        if not logos.get(t):
            missing_logo.append(t)
    if missing_logo:
        print(f"[warn] no logo for: {', '.join(missing_logo)}")

    data = {
        "config": {
            "max_goals": 10,
            "min_matches_for_full_trust": 8.0,
            "elo_start": 1500.0,
            "elo_home_adv": 65.0,
        },
        "base": {
            "teams": teams_js,
            "home_adv": round(m.home_adv, 4),
            "avg_goals": round(m.avg_goals, 4),
            "rho": round(m.rho, 5),
            "goal_scale": 1.0,
            "elo": elo_js,
            "draw_params": [round(draw_params[0], 4), round(draw_params[1], 1)],
        },
    }

    # Second bundle fitted on the xG-blended target — the site's "Use xG data"
    # toggle. Only exported when the back-test says it beats the goals fit.
    if xg_alpha is not None:
        mx = dc_fit(df, asof, xg_alpha)
        data["xg"] = {
            "teams": {t: {"a": round(mx.strength(t)[0], 6),
                          "d": round(mx.strength(t)[1], 6),
                          "w": round(mx.strength(t)[2], 3)} for t in sorted(chosen)},
            "home_adv": round(mx.home_adv, 4),
            "avg_goals": round(mx.avg_goals, 4),
            "rho": round(mx.rho, 5),
            "goal_scale": 1.0,
            "elo": elo_js,
            "draw_params": [round(draw_params[0], 4), round(draw_params[1], 1)],
            "alpha": xg_alpha,
        }
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d"),
        "season": f"{last}-{(last + 1) % 100:02d}",
        "note": note,
        "teams": meta_js,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("// Generated by clubs_build.py — fitted club model for the static site.\n")
        f.write("const MODEL_DATA = " + json.dumps(data, separators=(",", ":"),
                                                   ensure_ascii=False) + ";\n")
        f.write("const CLUB_META = " + json.dumps(meta, separators=(",", ":"),
                                                  ensure_ascii=False) + ";\n")
    print(f"[export] {len(teams_js)} teams -> {os.path.relpath(OUT, HERE)}")


# ------------------------------------------------------------------ names --

def names_report(df: pd.DataFrame) -> None:
    """Which UCL teams didn't fold onto a domestic-league club? Genuine
    externals (Salzburg, Shakhtar, ...) are expected; a big-five club in this
    list means a missing alias."""
    domestic = set()
    for code in LEAGUES:
        sub = df[df["comp"] == code]
        domestic |= set(sub["home_team"]) | set(sub["away_team"])
    ucl = df[df["comp"] == "UCL"]
    ucl_teams = set(ucl["home_team"]) | set(ucl["away_team"])
    external = sorted(ucl_teams - domestic)
    print(f"{len(ucl_teams)} UCL teams, {len(ucl_teams) - len(external)} matched "
          f"to a domestic league, {len(external)} external:")
    for t in external:
        n = len(ucl[(ucl["home_team"] == t) | (ucl["away_team"] == t)])
        print(f"  {t:<35s} {n:>3d} UCL matches")


# ------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--download", action="store_true",
                    help="force re-download of the two most recent seasons")
    ap.add_argument("--names", action="store_true",
                    help="print the team-name reconciliation report and exit")
    ap.add_argument("--no-backtest", action="store_true")
    args = ap.parse_args()

    df = load_all(force_current=args.download)
    print(f"[data] {len(df)} matches, {df['date'].min().date()} .. {df['date'].max().date()}")

    if args.names:
        names_report(df)
        return

    note, best_alpha = "", None
    if not args.no_backtest:
        test_season = current_season_start()
        if not (df["date"] >= pd.Timestamp(f"{test_season}-09-01")).any():
            test_season -= 1
        res = backtest(df, test_season)
        best_alpha = res["best_alpha"]
        r = res["all"]["dc"]
        note = (f"back-test on {test_season}-{(test_season + 1) % 100:02d} EPL+UCL: "
                f"log-loss {r['log_loss']:.4f}, accuracy {r['accuracy']:.1%} "
                f"(Elo baseline {res['all']['elo']['log_loss']:.4f} / "
                f"{res['all']['elo']['accuracy']:.1%})")
        if res.get("priced"):
            note += (f"; bookmaker closing odds on the same EPL matches: "
                     f"log-loss {res['priced']['market']['log_loss']:.4f} "
                     f"vs model {res['priced']['dc']['log_loss']:.4f}")

    export(df, note, best_alpha)


if __name__ == "__main__":
    main()
