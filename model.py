"""
model.py  —  Stage 2: strength estimation (Dixon-Coles Poisson model)
=====================================================================
We fit, by weighted maximum likelihood, a bivariate-Poisson-style goals model in
the spirit of Dixon & Coles (1997).

For a match between home team i and away team j at a NON-neutral venue:
    lambda_home = avg_goals * attack(i) * defence(j) * home_adv
    lambda_away = avg_goals * attack(j) * defence(i)
At a neutral venue (most World Cup games) the home_adv factor is dropped.

We fit in log-space for numerical stability, so with
    C        = log(avg_goals)                 (fixed to the observed global rate)
    a[i]     = log attack(i)                   (a>0  => scores more than average)
    d[i]     = log defence(i)                  (d<0  => concedes less  => strong D)
    h        = log home_adv
    rho      = Dixon-Coles low-score dependence
we have
    log lambda_home = C + a[i] + d[j] + h*(venue is home)
    log lambda_away = C + a[j] + d[i]

KEY MODELLING CHOICES (so you can learn from / tune the code):
  * Time-decay weights (recency) and competition-importance weights come in via
    the per-match `weight` column from data_loader — recent, important matches
    dominate the fit.
  * The Dixon-Coles tau correction inflates/deflates the probability of the four
    low-scoring results (0-0, 1-0, 0-1, 1-1). Independent Poisson under-predicts
    draws; tau (with rho<0 typically) fixes that.
  * L2 (ridge) penalty on a[] and d[] does double duty: it makes the parameters
    identifiable AND acts as a shrink-to-global-average prior, so a team with only
    a couple of matches cannot stray far from average. This is our built-in prior
    for thin-data teams (predict.py adds an explicit Elo fallback on top).
  * We supply the ANALYTIC gradient, so the ~650-parameter fit (≈320 teams) takes
    a fraction of a second and the back-test can re-fit many times cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


@dataclass
class FittedModel:
    """Container for a fitted Dixon-Coles model plus convenience predictors."""

    teams: list[str]
    team_index: dict[str, int]
    attack: np.ndarray          # exp(a[i]) per team, ~1.0 is average
    defence: np.ndarray         # exp(d[i]) per team, <1.0 is a strong defence
    home_adv: float             # exp(h), multiplicative home-goal boost
    rho: float                  # Dixon-Coles low-score correction
    avg_goals: float            # exp(C), global goals-per-team-per-match
    goal_scale: float = 1.0     # post-fit goal-total calibration (see config.GOAL_SCALE)
    weighted_matches: dict[str, float] = field(default_factory=dict)  # data mass per team
    log_likelihood: float = 0.0

    # ---- lookups -------------------------------------------------------- #
    def has_team(self, name: str) -> bool:
        return name in self.team_index

    def strength(self, name: str) -> tuple[float, float, float]:
        """Return (attack, defence, weighted_match_count) for a team, or averages."""
        if name in self.team_index:
            i = self.team_index[name]
            return (float(self.attack[i]), float(self.defence[i]),
                    float(self.weighted_matches.get(name, 0.0)))
        return (1.0, 1.0, 0.0)  # unknown team -> league average

    def expected_goals(self, home: str, away: str, neutral: bool = True
                       ) -> tuple[float, float]:
        """
        Compute (lambda_home, lambda_away) for a matchup using the multiplicative
        form. `neutral=True` (World Cup default) drops the home-advantage factor.
        """
        atk_h, def_h, _ = self.strength(home)
        atk_a, def_a, _ = self.strength(away)
        ha = 1.0 if neutral else self.home_adv
        # goal_scale calibrates absolute goal totals (competitive games out-score the
        # shrunk global baseline); it cancels in the home/away *ratio*, so it barely
        # touches who-wins but makes totals — and thus scorelines — realistic.
        lam_home = self.avg_goals * atk_h * def_a * ha * self.goal_scale
        lam_away = self.avg_goals * atk_a * def_h * self.goal_scale
        return float(lam_home), float(lam_away)

    def rank(self, top: int | None = None) -> pd.DataFrame:
        """Teams ordered by overall strength (attack/defence ratio)."""
        df = pd.DataFrame({
            "team": self.teams,
            "attack": self.attack,
            "defence": self.defence,
        })
        # A simple scalar strength: how many goals you'd score vs how many you'd
        # concede against an average opponent -> attack/defence.
        df["strength"] = df["attack"] / df["defence"]
        df = df.sort_values("strength", ascending=False).reset_index(drop=True)
        return df.head(top) if top else df


# --------------------------------------------------------------------------- #
# Likelihood + analytic gradient
# --------------------------------------------------------------------------- #
def _unpack(theta: np.ndarray, n: int):
    a = theta[:n]
    d = theta[n:2 * n]
    h = theta[2 * n]
    rho = theta[2 * n + 1]
    return a, d, h, rho


def _tau_and_grads(x, y, lam, mu, rho):
    """
    Dixon-Coles tau(x, y) and its partials, vectorised over matches.
    Only (0,0),(0,1),(1,0),(1,1) are corrected; everywhere else tau=1.
    Returns tau, dtau/dlam, dtau/dmu, dtau/drho.
    """
    tau = np.ones_like(lam)
    dlam = np.zeros_like(lam)
    dmu = np.zeros_like(lam)
    drho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    dlam[m00] = -mu[m00] * rho
    dmu[m00] = -lam[m00] * rho
    drho[m00] = -lam[m00] * mu[m00]

    tau[m01] = 1.0 + lam[m01] * rho
    dlam[m01] = rho
    drho[m01] = lam[m01]

    tau[m10] = 1.0 + mu[m10] * rho
    dmu[m10] = rho
    drho[m10] = mu[m10]

    tau[m11] = 1.0 - rho
    drho[m11] = -1.0

    # Keep tau strictly positive so log(tau) is finite (rho bounds make this rare).
    tau = np.clip(tau, 1e-10, None)
    return tau, dlam, dmu, drho


def _make_objective(hi, ai, x, y, w, home_mask, n, C, ridge):
    """
    Build (nll, grad) closure. All match arrays are pre-indexed integer/float
    numpy arrays so each evaluation is fully vectorised.
    """
    def objective(theta):
        a, d, h, rho = _unpack(theta, n)

        eta_h = C + a[hi] + d[ai] + h * home_mask
        eta_a = C + a[ai] + d[hi]
        lam = np.exp(eta_h)
        mu = np.exp(eta_a)

        tau, dtl, dtm, dtr = _tau_and_grads(x, y, lam, mu, rho)

        # Weighted negative log-likelihood (constant log-factorial terms dropped).
        loglik = w * (x * eta_h - lam + y * eta_a - mu + np.log(tau))
        nll = -loglik.sum() + ridge * (np.dot(a, a) + np.dot(d, d))

        # ---- gradient ---- #
        # d loglik / d eta_home  (Poisson part + tau chain rule: dlogtau/deta = (dtau/dlam)*lam/tau)
        g_eta_h = w * (x - lam + (dtl * lam) / tau)
        g_eta_a = w * (y - mu + (dtm * mu) / tau)
        g_rho = w * (dtr / tau)

        grad_a = np.zeros(n)
        grad_d = np.zeros(n)
        # eta_home depends on a[home], d[away], h ; eta_away depends on a[away], d[home]
        np.add.at(grad_a, hi, g_eta_h)
        np.add.at(grad_a, ai, g_eta_a)
        np.add.at(grad_d, ai, g_eta_h)
        np.add.at(grad_d, hi, g_eta_a)
        grad_h = np.sum(g_eta_h * home_mask)
        grad_rho = np.sum(g_rho)

        # Negate (we minimise nll) and add ridge derivative on a, d.
        grad = np.empty_like(theta)
        grad[:n] = -grad_a + 2.0 * ridge * a
        grad[n:2 * n] = -grad_d + 2.0 * ridge * d
        grad[2 * n] = -grad_h
        grad[2 * n + 1] = -grad_rho
        return nll, grad

    return objective


# --------------------------------------------------------------------------- #
# Public fit
# --------------------------------------------------------------------------- #
def fit(df: pd.DataFrame, ridge: float | None = None, verbose: bool = False
        ) -> FittedModel:
    """
    Fit the Dixon-Coles model on a weighted match DataFrame.

    `df` must have columns: home_team, away_team, home_goals, away_goals, and
    (ideally) `weight` from data_loader.add_weights. Missing weights => all 1.0.
    """
    if ridge is None:
        ridge = config.RIDGE_LAMBDA

    df = df.reset_index(drop=True)
    if "weight" not in df.columns:
        df = df.assign(weight=1.0)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_index = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df["home_team"].map(team_index).to_numpy()
    ai = df["away_team"].map(team_index).to_numpy()
    # Fit target: if xG-blended target columns are present (see xg.blend_into_matches)
    # use them, else raw goals. The Poisson score x*eta - lambda is valid for a
    # CONTINUOUS target, and the Dixon-Coles tau correction only fires on exact
    # integer low scores, so it auto-disables for the (non-integer) xG rows while
    # still applying to the real-goal rows in the same fit.
    if "home_target" in df.columns and "away_target" in df.columns:
        x = df["home_target"].to_numpy(dtype=float)
        y = df["away_target"].to_numpy(dtype=float)
    else:
        x = df["home_goals"].to_numpy(dtype=float)
        y = df["away_goals"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)
    # Home advantage only applies at non-neutral venues.
    home_mask = (~df["neutral"].to_numpy(dtype=bool)).astype(float)

    # Fixed global scale C = log(weighted average goals per team per match).
    total_w = w.sum()
    avg_goals = (w * (x + y)).sum() / (2.0 * total_w)
    C = np.log(avg_goals)

    # Weighted data mass per team -> used later to flag thin-data teams.
    wmatch: dict[str, float] = {}
    wm = np.zeros(n)
    np.add.at(wm, hi, w)
    np.add.at(wm, ai, w)
    for t, i in team_index.items():
        wmatch[t] = float(wm[i])

    objective = _make_objective(hi, ai, x, y, w, home_mask, n, C, ridge)

    # Initial guess: average strengths (0 in log space), config priors for h, rho.
    theta0 = np.zeros(2 * n + 2)
    theta0[2 * n] = config.HOME_ADV_INIT
    theta0[2 * n + 1] = config.RHO_INIT

    # Only rho is bounded (for stability); attack/defence/home are free but the
    # ridge penalty keeps them in a sane range.
    bounds = [(None, None)] * (2 * n)
    bounds.append((None, None))                 # h
    bounds.append(config.RHO_BOUNDS)            # rho

    res = minimize(
        objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
    )
    if verbose:
        print(f"[model] fit success={res.success} nll={res.fun:.1f} "
              f"iters={res.nit} msg={res.message}")

    a, d, h, rho = _unpack(res.x, n)
    return FittedModel(
        teams=teams,
        team_index=team_index,
        attack=np.exp(a),
        defence=np.exp(d),
        home_adv=float(np.exp(h)),
        rho=float(rho),
        avg_goals=float(avg_goals),
        goal_scale=float(config.GOAL_SCALE),
        weighted_matches=wmatch,
        log_likelihood=float(-res.fun),
    )


# --------------------------------------------------------------------------- #
# Gradient self-check (run when executed directly)
# --------------------------------------------------------------------------- #
def _grad_check():
    """Verify the analytic gradient against finite differences on a small sample."""
    import data_loader
    df = data_loader.load_matches(start="2021-01-01", end="2022-11-19")
    df = df.reset_index(drop=True)
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home_team"].map(ti).to_numpy()
    ai = df["away_team"].map(ti).to_numpy()
    x = df["home_goals"].to_numpy(float)
    y = df["away_goals"].to_numpy(float)
    w = df["weight"].to_numpy(float)
    hm = (~df["neutral"].to_numpy(bool)).astype(float)
    C = np.log((w * (x + y)).sum() / (2 * w.sum()))
    obj = _make_objective(hi, ai, x, y, w, hm, n, C, config.RIDGE_LAMBDA)

    rng = np.random.default_rng(0)
    theta = rng.normal(0, 0.1, 2 * n + 2)
    theta[2 * n + 1] = -0.05
    f0, g = obj(theta)
    eps = 1e-6
    max_err = 0.0
    for idx in list(rng.integers(0, len(theta), 20)) + [2 * n, 2 * n + 1]:
        tp = theta.copy(); tp[idx] += eps
        tm = theta.copy(); tm[idx] -= eps
        num = (obj(tp)[0] - obj(tm)[0]) / (2 * eps)
        max_err = max(max_err, abs(num - g[idx]))
    print(f"[grad-check] max |analytic - numeric| = {max_err:.2e}  (should be ~1e-4 or less)")


if __name__ == "__main__":
    import data_loader

    _grad_check()

    df = data_loader.load_matches(end="2022-11-19")
    m = fit(df, verbose=True)
    print(f"\navg_goals={m.avg_goals:.3f}  home_adv={m.home_adv:.3f}  rho={m.rho:.3f}")
    print("\nTop 15 teams by attack/defence strength (eve of 2022 WC):")
    print(m.rank(15).to_string(index=False,
          formatters={"attack": "{:.2f}".format, "defence": "{:.2f}".format,
                      "strength": "{:.2f}".format}))

    lh, la = m.expected_goals("Brazil", "Serbia", neutral=True)
    print(f"\nBrazil vs Serbia expected goals: {lh:.2f} - {la:.2f}")
