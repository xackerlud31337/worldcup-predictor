"""
elo.py  —  self-contained Elo ratings
======================================
Two jobs:
  1. Provide the NAIVE BASELINE the Dixon-Coles model must beat ("predict from
     the ranking"). We compute Elo from match results rather than scraping live
     FIFA rankings so the pipeline is fully reproducible and needs no API key —
     Elo tracks the FIFA ranking closely and is the standard "strength by
     ranking" benchmark.
  2. Provide a strength PRIOR / fallback for teams with little or no data in the
     Dixon-Coles fit (new qualifiers, minnows).

Elo only models win/draw/loss (no scorelines), which is exactly what a ranking
baseline should do. We convert a rating difference into W/D/L probabilities with
a small, optionally-fitted draw model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _margin_multiplier(goal_diff: int) -> float:
    """
    Bigger wins should move ratings more, but with diminishing returns (a 5-0 is
    not 5x as informative as a 1-0). This is the well-known World-Football-Elo
    margin-of-victory multiplier.
    """
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def fifa_seed_ratings() -> dict[str, float]:
    """
    Map current FIFA ranking points onto the Elo scale (see config). Used to seed
    each team's starting rating instead of a flat 1500, so the ratings — and the
    thin-data prior built on them — reflect current squad strength.
    """
    return {team: config.ELO_START + (pts - config.FIFA_ANCHOR_REF) * config.FIFA_ANCHOR_SCALE
            for team, pts in config.FIFA_RANKING_POINTS.items()}


def compute_ratings(df: pd.DataFrame,
                    seed_ratings: dict[str, float] | None = None) -> dict[str, float]:
    """
    Walk matches in chronological order and return final Elo ratings per team.

    seed_ratings : optional per-team starting ratings (e.g. from FIFA points via
    fifa_seed_ratings). Teams not in the seed default to config.ELO_START.

    IMPORTANT for back-testing: pass only matches strictly *before* the tournament
    you are predicting, so the ratings contain no look-ahead information.
    """
    ratings: dict[str, float] = dict(seed_ratings) if seed_ratings else {}
    k = config.ELO_K
    home_adv = config.ELO_HOME_ADV

    # Iterating rows is fine here (~30k rows, one pass, runs in well under a second).
    for home, away, hg, ag, neutral in zip(
        df["home_team"], df["away_team"], df["home_goals"], df["away_goals"], df["neutral"]
    ):
        rh = ratings.get(home, config.ELO_START)
        ra = ratings.get(away, config.ELO_START)

        adv = 0.0 if neutral else home_adv
        # Expected score for the home team under the logistic Elo model.
        exp_home = 1.0 / (1.0 + 10.0 ** ((ra - (rh + adv)) / 400.0))

        if hg > ag:
            score_home = 1.0
        elif hg == ag:
            score_home = 0.5
        else:
            score_home = 0.0

        mult = _margin_multiplier(hg - ag) if config.ELO_MARGIN_SCALE else 1.0
        delta = k * mult * (score_home - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta  # zero-sum update

    return ratings


def expected_home_score(r_home: float, r_away: float, neutral: bool) -> float:
    """Elo expected score for the home team in [0, 1] (= P(win) + 0.5*P(draw))."""
    adv = 0.0 if neutral else config.ELO_HOME_ADV
    return 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + adv)) / 400.0))


def wdl_probs(
    r_home: float,
    r_away: float,
    neutral: bool = True,
    draw_base: float = 0.28,
    draw_decay: float = 220.0,
) -> tuple[float, float, float]:
    """
    Convert two Elo ratings into (P_home_win, P_draw, P_away_win).

    Elo gives only the expected score E = P_win + 0.5*P_draw, so we need a draw
    model to split it. Empirically, evenly-matched teams draw most often and the
    draw rate falls as the rating gap grows; we model that as
        P_draw = draw_base * exp(-|Δrating_effective| / draw_decay)
    then solve P_win = E - 0.5*P_draw and P_loss = 1 - P_win - P_draw.
    The two constants can be fitted from data (see `fit_draw_params`).
    """
    adv = 0.0 if neutral else config.ELO_HOME_ADV
    dr = (r_home + adv) - r_away
    e_home = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    p_draw = draw_base * np.exp(-abs(dr) / draw_decay)
    p_home = e_home - 0.5 * p_draw
    p_away = 1.0 - p_home - p_draw

    # Numerical safety: keep everything a valid probability and renormalise.
    p = np.clip(np.array([p_home, p_draw, p_away], dtype=float), 1e-6, None)
    p /= p.sum()
    return float(p[0]), float(p[1]), float(p[2])


def fit_draw_params(
    df: pd.DataFrame, ratings: dict[str, float]
) -> tuple[float, float]:
    """
    Coarse grid-search for (draw_base, draw_decay) that minimises W/D/L log-loss
    on the given matches. Keeps the baseline honestly calibrated instead of using
    hand-waved constants, without over-fitting (only two parameters, coarse grid).
    """
    rows = []
    for home, away, hg, ag, neutral in zip(
        df["home_team"], df["away_team"], df["home_goals"], df["away_goals"], df["neutral"]
    ):
        rh = ratings.get(home, config.ELO_START)
        ra = ratings.get(away, config.ELO_START)
        outcome = 0 if hg > ag else (1 if hg == ag else 2)
        rows.append((rh, ra, bool(neutral), outcome))

    best, best_ll = (0.28, 220.0), np.inf
    for base in np.linspace(0.15, 0.38, 12):
        for decay in np.linspace(120.0, 400.0, 12):
            ll = 0.0
            for rh, ra, neutral, outcome in rows:
                p = wdl_probs(rh, ra, neutral, base, decay)
                ll -= np.log(max(p[outcome], 1e-12))
            if ll < best_ll:
                best_ll, best = ll, (float(base), float(decay))
    return best


if __name__ == "__main__":
    import data_loader

    df = data_loader.load_matches(end="2022-11-19")  # before 2022 WC
    ratings = compute_ratings(df)
    top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:12]
    print("Top Elo (as of eve of 2022 WC):")
    for t, r in top:
        print(f"  {t:15s} {r:7.1f}")
    print("\nExample W/D/L  Brazil (home, neutral) vs Serbia:")
    print(wdl_probs(ratings["Brazil"], ratings["Serbia"], neutral=True))
