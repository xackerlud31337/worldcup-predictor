"""
xg.py  —  Expected-Goals model from shot x/y coordinates
========================================================
Turns each shot's pitch coordinates into P(goal) — its "expected goals" (xG).

The two dominant drivers of whether a shot scores are geometric:
  * DISTANCE from the goal — closer is better.
  * ANGLE subtended by the goalposts — a wide-open view of the goal is better
    than a tight angle from the byline. We compute the actual angle between the
    lines from the shot to each post (posts at (120,36) and (120,44)).
Plus a few binary shot descriptors: header (harder), penalty (a near-fixed ~0.76),
and free-kick.

We fit a logistic regression (implemented here with scipy so the only deps stay
numpy/pandas/scipy) with an analytic gradient and light L2 regularisation, then
validate it against actual goals and cross-check against StatsBomb's own xG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

GOAL_X, GOAL_Y = 120.0, 40.0
POST1 = (120.0, 36.0)
POST2 = (120.0, 44.0)


# --------------------------------------------------------------------------- #
# Feature engineering from x/y
# --------------------------------------------------------------------------- #
def shot_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the feature matrix (no intercept column) from shot coordinates."""
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    dx = GOAL_X - x
    distance = np.sqrt(dx ** 2 + (y - GOAL_Y) ** 2)

    # Angle subtended by the two posts, via the law of cosines.
    a = np.sqrt((x - POST1[0]) ** 2 + (y - POST1[1]) ** 2)
    b = np.sqrt((x - POST2[0]) ** 2 + (y - POST2[1]) ** 2)
    c = abs(POST1[1] - POST2[1])  # goal width = 8
    cos_ang = np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b + 1e-9), -1, 1)
    angle = np.arccos(cos_ang)  # radians; larger = better view

    is_header = (df["body_part"].to_numpy() == "Head").astype(float)
    is_penalty = (df["shot_type"].to_numpy() == "Penalty").astype(float)
    is_freekick = np.char.find(df["shot_type"].to_numpy().astype(str),
                               "Free Kick") >= 0
    is_freekick = is_freekick.astype(float)

    X = np.column_stack([distance, angle, is_header, is_penalty, is_freekick])
    names = ["distance", "angle", "is_header", "is_penalty", "is_freekick"]
    return X, names


# --------------------------------------------------------------------------- #
# Logistic regression (self-contained)
# --------------------------------------------------------------------------- #
@dataclass
class XGModel:
    weights: np.ndarray     # includes bias as weights[0]
    mean: np.ndarray        # feature standardisation (excl. bias)
    std: np.ndarray
    names: list[str]

    def _design(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean) / self.std
        return np.column_stack([np.ones(len(Xs)), Xs])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = shot_features(df)
        z = self._design(X) @ self.weights
        return 1.0 / (1.0 + np.exp(-z))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_xg_model(shots: pd.DataFrame, l2: float = 1.0) -> XGModel:
    """Fit the logistic xG model on a shots DataFrame (needs x, y, is_goal, ...)."""
    X, names = shot_features(shots)
    y = shots["is_goal"].to_numpy(float)

    # Standardise features so the optimiser is well-conditioned; keep bias separate.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xs = (X - mean) / std
    D = np.column_stack([np.ones(len(Xs)), Xs])  # design matrix with intercept
    n_params = D.shape[1]

    def nll(w):
        z = D @ w
        p = _sigmoid(z)
        eps = 1e-12
        ll = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # L2 on all but the bias term.
        pen = l2 * np.dot(w[1:], w[1:])
        grad = D.T @ (p - y)
        grad[1:] += 2 * l2 * w[1:]
        return -ll + pen, grad

    res = minimize(nll, np.zeros(n_params), jac=True, method="L-BFGS-B")
    return XGModel(weights=res.x, mean=mean, std=std, names=names)


# --------------------------------------------------------------------------- #
# Per-match team xG aggregation  (the bridge to the Dixon-Coles model)
# --------------------------------------------------------------------------- #
def match_team_xg(shots: pd.DataFrame, model: XGModel) -> pd.DataFrame:
    """
    Aggregate shot xG into per-match team totals, returned in the same shape the
    results dataset uses: date, home_team, away_team, home_xg, away_xg.
    """
    s = shots.copy()
    s["xg"] = model.predict(s)
    # Sum each team's xG within each match.
    grp = s.groupby(["match_id", "date", "home_team", "away_team", "team"], as_index=False)["xg"].sum()

    rows = []
    for (mid, date, home, away), g in grp.groupby(["match_id", "date", "home_team", "away_team"]):
        hx = g.loc[g["team"] == home, "xg"].sum()
        ax = g.loc[g["team"] == away, "xg"].sum()
        rows.append({"date": date, "home_team": home, "away_team": away,
                     "home_xg": float(hx), "away_xg": float(ax)})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def blend_into_matches(matches: pd.DataFrame, match_xg: pd.DataFrame,
                       alpha: float = 0.7) -> pd.DataFrame:
    """
    Attach `home_target`/`away_target` columns to the main results DataFrame:

        target = alpha * xG + (1 - alpha) * actual_goals   (for covered matches)
        target = actual_goals                              (everything else)

    alpha in [0, 1] controls how much we trust xG over the scoreboard. xG is a
    less-noisy signal of performance (a team that carves out great chances but
    hits the post "deserved" more), so blending toward it should sharpen strength
    estimates for the elite teams that have event data — while every uncovered
    team keeps its real goals unchanged. Matched on (date, home_team, away_team).
    """
    m = matches.copy().reset_index(drop=True)
    m["_d"] = pd.to_datetime(m["date"]).dt.normalize()

    # Index each xG match by its UNORDERED team pair, so we match regardless of
    # which side the two sources call "home" (arbitrary for neutral World Cup
    # games). We then orient the xG to the results row's home/away.
    xg_by_pair: dict[frozenset, list] = {}
    for _, r in match_xg.iterrows():
        key = frozenset((r["home_team"], r["away_team"]))
        xg_by_pair.setdefault(key, []).append(
            (pd.Timestamp(r["date"]).normalize(), r["home_team"],
             float(r["home_xg"]), float(r["away_xg"])))

    home_xg = np.full(len(m), np.nan)
    away_xg = np.full(len(m), np.nan)
    tol = pd.Timedelta(days=2)  # allow small date offsets (timezones/late kickoffs)
    for i, row in m.iterrows():
        key = frozenset((row["home_team"], row["away_team"]))
        cands = xg_by_pair.get(key)
        if not cands:
            continue
        # nearest-date candidate within tolerance
        best = min(cands, key=lambda c: abs(c[0] - row["_d"]))
        if abs(best[0] - row["_d"]) > tol:
            continue
        _, xg_home_team, hxg, axg = best
        # Orient: if the xG row's home team is this row's home team, keep; else swap.
        if xg_home_team == row["home_team"]:
            home_xg[i], away_xg[i] = hxg, axg
        else:
            home_xg[i], away_xg[i] = axg, hxg

    m["home_xg"] = home_xg
    m["away_xg"] = away_xg
    has_xg = m["home_xg"].notna()
    m["home_target"] = np.where(
        has_xg, alpha * m["home_xg"] + (1 - alpha) * m["home_goals"], m["home_goals"])
    m["away_target"] = np.where(
        has_xg, alpha * m["away_xg"] + (1 - alpha) * m["away_goals"], m["away_goals"])
    m["has_xg"] = has_xg
    return m.drop(columns=["_d"])


if __name__ == "__main__":
    import statsbomb

    shots = statsbomb.load_shots()
    model = fit_xg_model(shots)

    p = model.predict(shots)
    y = shots["is_goal"].to_numpy(float)
    # Our model vs the actual goal rate and vs StatsBomb's xG.
    ll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    print("xG model coefficients (standardised):")
    for n, w in zip(["bias"] + model.names, model.weights):
        print(f"  {n:<12} {w:+.3f}")
    print(f"\ntotal predicted xG : {p.sum():.1f}   actual goals : {y.sum():.0f}")
    print(f"log-loss (per shot): {ll:.4f}")
    have_sb = shots["sb_xg"].notna()
    if have_sb.any():
        corr = np.corrcoef(p[have_sb.to_numpy()],
                           shots.loc[have_sb, "sb_xg"].to_numpy(float))[0, 1]
        print(f"correlation with StatsBomb xG: {corr:.3f}")

    # Example: a few shots at varying distance/angle.
    print("\nsanity — xG at sample spots (open-play, foot):")
    demo = pd.DataFrame({
        "x": [108, 100, 88, 114, 120], "y": [40, 40, 25, 30, 40],
        "body_part": ["Right Foot"] * 5, "shot_type": ["Open Play"] * 5,
    })
    for (_, r), xg in zip(demo.iterrows(), model.predict(demo)):
        dist = np.sqrt((120 - r.x) ** 2 + (40 - r.y) ** 2)
        print(f"  ({r.x:3.0f},{r.y:2.0f})  dist={dist:4.1f}yd  xG={xg:.3f}")

    # Per-match team xG table (bridge to Dixon-Coles).
    mx = match_team_xg(shots, model)
    print(f"\nper-match team-xG table: {len(mx)} matches")
    print(mx.head(4).to_string(index=False,
          formatters={"home_xg": "{:.2f}".format, "away_xg": "{:.2f}".format}))
