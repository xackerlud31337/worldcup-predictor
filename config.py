"""
config.py
=========
All tunable parameters live here so the rest of the code stays declarative and
so you can sweep hyper-parameters (half-life, home advantage prior, date range)
from one place.

Every value has a short note explaining *why* it exists and what changing it does,
so this doubles as documentation of the model's knobs.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths / caching
# --------------------------------------------------------------------------- #
# Where downloaded raw data and the cleaned parquet cache are stored. We cache so
# that repeated runs (especially the back-test, which re-fits many times) never
# re-download the ~4 MB results file.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# martj42/international_results is the actively-maintained GitHub mirror of the
# Kaggle "International football results 1872-present" dataset. No API key needed.
# It already contains every FIFA World Cup match (labelled tournament=="FIFA World
# Cup") plus ~48k other internationals, so it serves BOTH strength-fitting and
# back-testing from a single source.
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# --------------------------------------------------------------------------- #
# Data selection
# --------------------------------------------------------------------------- #
# Matches before this date are dropped by default. Very old football (pre-war,
# amateur era) has little bearing on modern strengths and can distort the global
# average goal rate. Time-decay weighting handles recency *within* the window;
# this cutoff just trims the long irrelevant tail for speed.
MIN_DATE = "1990-01-01"

# Some competitions are far more informative about "true" strength than others.
# Friendlies are noisy (teams rest players, experiment); competitive matches carry
# more signal. We down-weight friendlies rather than dropping them, because for
# minor nations friendlies may be the *only* data we have.
#
# Weight multipliers by tournament class (matched via substring, case-insensitive):
COMPETITION_WEIGHTS = {
    "FIFA World Cup": 1.00,
    "qualification": 0.90,   # WC / continental qualifiers
    "UEFA Euro": 1.00,
    "Copa América": 1.00,
    "African Cup of Nations": 1.00,
    "AFC Asian Cup": 1.00,
    "Gold Cup": 0.90,
    "UEFA Nations League": 0.95,
    "Confederations Cup": 0.95,
    "Friendly": 0.55,
    "_default": 0.80,        # anything not listed above
}

# --------------------------------------------------------------------------- #
# Time-decay weighting  (recency)
# --------------------------------------------------------------------------- #
# Recent form matters more than ancient history. We weight each match by
#     w = 0.5 ** (age_in_days / HALF_LIFE_DAYS)
# so a match exactly one half-life old counts half as much as a brand-new one.
#
# Dixon & Coles (1997) found best predictive performance with a half-life on the
# order of ~1-2 seasons for club football. International football is sparser
# (teams play ~10 games/year), so we need a LONGER half-life so each team keeps
# enough *effective* sample. A sweep over {730,1095,1460,2190} showed that on
# 2018+2022 alone, longer is marginally better on log-loss (2190 won). But ADDING
# the 2026 back-test flipped the verdict on accuracy: 1460 gets 65% on 2026 vs 62%
# at 2190 (log-loss essentially tied), confirming that a 6-year window over-smooths
# recently-transformed squads. 1460 is the robust choice — best 3-year accuracy
# (60.1% vs 59.2%) at a negligible log-loss cost.
HALF_LIFE_DAYS = 1460.0   # ≈ 4 years (one World Cup cycle)

# --------------------------------------------------------------------------- #
# Dixon-Coles model hyper-parameters
# --------------------------------------------------------------------------- #
# Ridge (L2) penalty on attack/defence parameters. This does three jobs at once:
#   1. Fixes the otherwise-unidentified location of the log-strength parameters.
#   2. Shrinks thin-data teams toward the global average (our built-in prior for
#      new qualifiers — a team with 2 games can't move far from average).
#   3. Regularises to prevent blow-ups from lopsided scorelines (e.g. 7-0).
# Larger => more shrinkage (safer, less reactive). The back-test showed the
# original value (5.0) made the model far too timid — it pulled every team toward
# average, so predictions never left the 20-60% band and it failed to beat the
# ranking baseline. 0.5 lets genuine strength gaps show while still shrinking
# thin-data teams. Tune via validate.py's sweep (winner with half_life 2190: 0.2).
RIDGE_LAMBDA = 0.2

# Optional stronger prior: for teams with fewer than this many weighted matches we
# additionally blend their fitted strength toward the Elo-implied strength, since
# ridge-to-global-average alone can be too crude for a genuine minnow vs. a team
# that is simply new to the dataset. See model.py.
MIN_MATCHES_FOR_FULL_TRUST = 8.0

# Starting value / prior mean for the home-advantage parameter (in log space).
# exp(0.25) ≈ 1.28, i.e. the home side scores ~28% more than neutral — close to
# the long-run empirical value for international football. It is still *fitted*;
# this is just the initialisation and the mean of its weak prior.
HOME_ADV_INIT = 0.25

# The Dixon-Coles low-score dependence parameter rho is initialised near 0
# (0 == independent Poisson). It is fitted freely but bounded for stability.
RHO_INIT = -0.05
RHO_BOUNDS = (-0.2, 0.2)

# Goal-total calibration factor applied to BOTH lambdas at prediction time.
# The strengths are fit across all internationals, whose global average is pulled
# down by cagey qualifiers and mismatches vs weak defences. Competitive tournament
# games (World Cups) actually score MORE than the model's shrunk baseline predicts:
# the back-test showed predicted ~2.2 vs actual ~2.66 goals/game. Scaling lambdas by
# 1.2 makes predicted totals match reality (2.65 vs 2.66) AND lowers W/D/L log-loss
# (0.9937 -> 0.9869), because under-counting goals had made the model too draw-heavy.
# Validated in validate.py; set to 1.0 to disable. This mainly makes SCORELINES
# realistic (fewer 2-1s, more 3-1/4-1 for strong favourites).
GOAL_SCALE = 1.2

# --------------------------------------------------------------------------- #
# Elo baseline / prior
# --------------------------------------------------------------------------- #
# A self-contained Elo rating computed from match results. Used for (a) the naive
# baseline the model must beat, and (b) the strength prior for thin-data teams.
# We compute Elo instead of scraping live FIFA rankings so the whole pipeline is
# reproducible and needs no API key. Elo tracks the FIFA ranking closely in
# practice and is a fair "predict by ranking" benchmark.
ELO_START = 1500.0
ELO_K = 40.0             # update step; higher = more reactive
ELO_HOME_ADV = 65.0      # rating points added to the home side (~ +0.10 win prob)
ELO_MARGIN_SCALE = True  # scale K by goal margin (bigger wins move rating more)

# --------------------------------------------------------------------------- #
# FIFA-ranking anchor (optional; toggle)
# --------------------------------------------------------------------------- #
# The model learns strength purely from historical scorelines, so it is blind to
# current squad quality — e.g. France (real FIFA #3) can look too weak, and teams
# with little data are unstable. Seeding each team's Elo START with its current
# FIFA points (instead of a flat 1500) injects that outside knowledge and gives
# thin-data teams a sensible prior.
#
# LEAKAGE NOTE: these points are the 11 June 2026 official ranking, so they are
# only valid for CURRENT predictions and the 2026 back-test. Applying them to the
# 2018/2022 back-tests would use future information, so validate.py keeps the
# anchor OFF for those years regardless of this toggle.
USE_FIFA_ANCHOR = False   # global default; set True to seed Elo from FIFA points

# Official FIFA/Coca-Cola Men's World Ranking points, update of 11 June 2026
# (source: inside.fifa.com/fifa-world-ranking/men, top 50). Team names are already
# canonicalised to match data_loader (USA->United States, IR Iran->Iran, etc.).
FIFA_RANKING_POINTS = {
    "Argentina": 1877.27, "Spain": 1874.71, "France": 1870.70, "England": 1828.02,
    "Portugal": 1767.85, "Brazil": 1765.86, "Morocco": 1755.10, "Netherlands": 1753.57,
    "Belgium": 1742.24, "Germany": 1735.77, "Croatia": 1714.87, "Italy": 1704.73,
    "Colombia": 1698.35, "Mexico": 1687.48, "Senegal": 1684.07, "Uruguay": 1673.07,
    "United States": 1671.23, "Japan": 1661.58, "Switzerland": 1650.06, "Iran": 1619.58,
    "Denmark": 1619.47, "Turkey": 1605.73, "Ecuador": 1598.52, "Austria": 1597.40,
    "South Korea": 1591.63, "Nigeria": 1585.02, "Australia": 1579.34, "Algeria": 1571.03,
    "Egypt": 1562.37, "Canada": 1559.48, "Norway": 1557.44, "Ukraine": 1549.29,
    "Ivory Coast": 1540.87, "Panama": 1539.16, "Russia": 1529.60, "Poland": 1526.18,
    "Wales": 1516.95, "Sweden": 1509.79, "Hungary": 1506.39, "Czechia": 1505.74,
    "Paraguay": 1505.35, "Scotland": 1503.34, "Serbia": 1502.13, "Cameroon": 1481.24,
    "Tunisia": 1476.41, "DR Congo": 1474.43, "Slovakia": 1473.66, "Greece": 1473.19,
    "Venezuela": 1469.18, "Uzbekistan": 1458.73,
}

# Linear map FIFA points -> Elo seed:  elo = ELO_START + (points - REF) * SCALE.
# REF is set near the 50th-ranked team's points so that teams just outside the top
# 50 (which fall back to a flat ELO_START) join smoothly with no cliff; SCALE
# stretches the ~420-point top-50 spread onto a realistic Elo spread.
FIFA_ANCHOR_REF = 1458.0
FIFA_ANCHOR_SCALE = 1.5

# --------------------------------------------------------------------------- #
# Prediction / simulation
# --------------------------------------------------------------------------- #
# Max goals considered when building an exact scoreline matrix (used for the
# Dixon-Coles-corrected W/D/L; Skellam is used for the fast path). 10 is plenty:
# P(team scores >10 in one game) is negligible.
MAX_GOALS = 10

# Default number of Monte Carlo scenarios.
N_SIMULATIONS = 10_000

# Reference date for "recency": defaults to today at runtime, but can be pinned
# (e.g. the day before a tournament) for reproducible back-tests.
REFERENCE_DATE = None  # None -> use today's date

# --------------------------------------------------------------------------- #
# xG (expected goals) hybrid — optional refinement
# --------------------------------------------------------------------------- #
# When enabled (CLI --xg), matches that have StatsBomb event data use a blended
# target  alpha*xG + (1-alpha)*actual_goals  instead of raw goals in the fit; all
# other matches keep their real goals. xG is a less-noisy signal of performance.
# Back-testing (validate.compare_xg) shows this helps only MARGINALLY for World
# Cup prediction, because free event data covers just a few tournaments (~30 elite
# teams) — so it is OFF by default. alpha=0.7 leans on xG while keeping some of the
# scoreboard. Tournaments used are statsbomb.DEFAULT_TOURNAMENTS.
XG_ALPHA = 0.7
