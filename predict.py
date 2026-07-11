"""
predict.py  —  Stage 3: single-match prediction
================================================
Turn (lambda_home, lambda_away) into Win / Draw / Loss probabilities for the home
team. Two methods are provided:

  * skellam_wdl  — FAST. The goal *difference* of two independent Poissons follows
    the Skellam distribution, so P(home win)=P(D>0), P(draw)=P(D=0), etc. One
    closed-form call, trivially vectorisable — ideal for bulk/Monte-Carlo use.
    Limitation: it treats the two Poissons as independent, so it cannot apply the
    Dixon-Coles tau correction (it slightly under-counts draws).

  * dc_matrix_wdl — ACCURATE. Builds the (max_goals+1)^2 joint scoreline matrix
    and applies the Dixon-Coles tau correction to the four low-score cells before
    summing the lower/diagonal/upper triangles. Still microseconds for an 11x11
    grid, and it captures the draw-inflation that motivated Dixon-Coles.

We use dc_matrix_wdl as the default for single matches (accuracy costs nothing at
this size) and expose skellam_wdl for the vectorised simulation path.

`predict_match` also implements the THIN-DATA FALLBACK: if either team has little
data in the fitted model, we blend its Dixon-Coles probabilities toward the Elo
(ranking) prediction, smoothly, in proportion to how little data we have.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson, skellam

import config
import elo


# --------------------------------------------------------------------------- #
# lambda -> W/D/L
# --------------------------------------------------------------------------- #
def skellam_wdl(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    """Fast W/D/L via the Skellam (Poisson-difference) distribution."""
    p_draw = skellam.pmf(0, lam_home, lam_away)
    # P(home win) = P(diff >= 1) = 1 - CDF(0); P(away win) = CDF(-1).
    p_home = 1.0 - skellam.cdf(0, lam_home, lam_away)
    p_away = skellam.cdf(-1, lam_home, lam_away)
    return _normalise(p_home, p_draw, p_away)


def dc_score_matrix(lam_home: float, lam_away: float, rho: float,
                    max_goals: int | None = None) -> np.ndarray:
    """
    Joint probability matrix M[x, y] = P(home scores x, away scores y), including
    the Dixon-Coles tau low-score correction. Rows = home goals, cols = away goals.
    """
    if max_goals is None:
        max_goals = config.MAX_GOALS
    hg = poisson.pmf(np.arange(max_goals + 1), lam_home)
    ag = poisson.pmf(np.arange(max_goals + 1), lam_away)
    M = np.outer(hg, ag)  # independent-Poisson baseline

    # Apply tau only to the 2x2 low-score corner.
    M[0, 0] *= 1.0 - lam_home * lam_away * rho
    M[0, 1] *= 1.0 + lam_home * rho
    M[1, 0] *= 1.0 + lam_away * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0.0, None)
    M /= M.sum()  # renormalise (truncation + tau)
    return M


def dc_matrix_wdl(lam_home: float, lam_away: float, rho: float = 0.0,
                  max_goals: int | None = None) -> tuple[float, float, float]:
    """Accurate W/D/L from the Dixon-Coles-corrected scoreline matrix."""
    M = dc_score_matrix(lam_home, lam_away, rho, max_goals)
    p_home = np.tril(M, -1).sum()   # home goals > away goals
    p_draw = np.trace(M)            # equal
    p_away = np.triu(M, 1).sum()    # away goals > home goals
    return _normalise(p_home, p_draw, p_away)


def _normalise(p_home, p_draw, p_away) -> tuple[float, float, float]:
    p = np.clip(np.array([p_home, p_draw, p_away], dtype=float), 1e-9, None)
    p /= p.sum()
    return float(p[0]), float(p[1]), float(p[2])


# --------------------------------------------------------------------------- #
# Full match prediction with thin-data Elo fallback
# --------------------------------------------------------------------------- #
def _data_trust(model, home: str, away: str) -> float:
    """
    Confidence in the Dixon-Coles estimate for THIS matchup, in [0, 1], driven by
    the weaker-supported of the two teams. 1.0 = both teams well observed, 0.0 =
    (near-)unseen team. Used to decide how much to lean on the Elo prior.
    """
    thr = config.MIN_MATCHES_FOR_FULL_TRUST
    _, _, wh = model.strength(home)
    _, _, wa = model.strength(away)
    return float(min(1.0, wh / thr) * min(1.0, wa / thr))


def predict_match(
    model,
    home: str,
    away: str,
    neutral: bool = True,
    method: str = "dc",
    elo_ratings: dict[str, float] | None = None,
    draw_params: tuple[float, float] | None = None,
) -> dict:
    """
    Predict W/D/L for `home` vs `away`.

    method : "dc" (Dixon-Coles scoreline matrix, default) or "skellam" (fast).
    neutral: True for World Cup games (no home advantage). Set False for a real
             home fixture.
    elo_ratings : if given, thin-data teams are blended toward their Elo (ranking)
             prediction — our fallback prior for new qualifiers / minnows.

    Returns a dict with probabilities, the expected goals, and diagnostic fields.
    """
    lam_home, lam_away = model.expected_goals(home, away, neutral=neutral)

    if method == "skellam":
        p = skellam_wdl(lam_home, lam_away)
    elif method == "dc":
        p = dc_matrix_wdl(lam_home, lam_away, model.rho)
    else:
        raise ValueError(f"unknown method {method!r} (use 'dc' or 'skellam')")

    trust = 1.0
    p_final = p
    if elo_ratings is not None:
        trust = _data_trust(model, home, away)
        if trust < 1.0:
            rh = elo_ratings.get(home, config.ELO_START)
            ra = elo_ratings.get(away, config.ELO_START)
            dp = draw_params or (0.28, 220.0)
            p_elo = elo.wdl_probs(rh, ra, neutral=neutral,
                                  draw_base=dp[0], draw_decay=dp[1])
            # Convex blend: full trust -> pure DC; no trust -> pure Elo prior.
            p_final = tuple(trust * pd + (1 - trust) * pe
                            for pd, pe in zip(p, p_elo))
            p_final = _normalise(*p_final)

    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "p_home_win": p_final[0],
        "p_draw": p_final[1],
        "p_away_win": p_final[2],
        "data_trust": trust,
        "method": method,
    }


def format_prediction(pred: dict) -> str:
    """Human-readable one-block summary of a prediction."""
    lines = [
        f"{pred['home']}  vs  {pred['away']}"
        + ("   (neutral venue)" if pred["neutral"] else "   (home fixture)"),
        f"  expected goals : {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}",
        f"  {pred['home']} win : {pred['p_home_win']*100:5.1f}%",
        f"  Draw           : {pred['p_draw']*100:5.1f}%",
        f"  {pred['away']} win : {pred['p_away_win']*100:5.1f}%",
    ]
    if pred["data_trust"] < 1.0:
        lines.append(f"  (thin data: DC/Elo blend, trust={pred['data_trust']:.2f})")
    return "\n".join(lines)


if __name__ == "__main__":
    import data_loader
    import model as model_mod

    df = data_loader.load_matches(end="2022-11-19")
    m = model_mod.fit(df)
    ratings = elo.compute_ratings(df)

    for h, a in [("Brazil", "Serbia"), ("Argentina", "Saudi Arabia"),
                 ("England", "United States"), ("Qatar", "Ecuador")]:
        pred = predict_match(m, h, a, neutral=True, elo_ratings=ratings)
        print(format_prediction(pred))
        # Cross-check fast vs accurate method.
        ps = skellam_wdl(pred["lambda_home"], pred["lambda_away"])
        print(f"  [skellam W/D/L: {ps[0]*100:.1f}/{ps[1]*100:.1f}/{ps[2]*100:.1f}]\n")
