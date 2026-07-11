"""
simulate.py  —  Stage 5: Monte Carlo simulation
================================================
Vectorised NumPy simulation so 10,000+ scenarios finish in a fraction of a second.

Three building blocks:
  * simulate_match     — sample many scorelines for ONE fixture (samples directly
                         from the Dixon-Coles joint scoreline matrix, so it keeps
                         the tau low-score correction). Returns the W/D/L split and
                         the most-likely scorelines.
  * simulate_group     — round-robin group: simulate final tables N times and
                         report each team's P(finish 1st / top-2 / etc.).
  * simulate_knockout  — single-elimination bracket: play the whole bracket N times
                         (draws resolved by a coin-flip shootout) and report each
                         team's probability of advancing / winning the trophy.

All three sample goals with a Poisson model (independent per side) except the
single-match helper, which uses the exact DC matrix. For bracket/group outcomes
the tiny tau correction is irrelevant, and Poisson sampling vectorises cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import predict as predict_mod


# --------------------------------------------------------------------------- #
# Single match
# --------------------------------------------------------------------------- #
def simulate_match(model, home: str, away: str, neutral: bool = True,
                   n: int | None = None, seed: int | None = None) -> dict:
    """
    Monte-Carlo a single fixture by sampling n scorelines from the Dixon-Coles
    joint distribution. Returns the W/D/L frequencies and the top scorelines.
    """
    if n is None:
        n = config.N_SIMULATIONS
    rng = np.random.default_rng(seed)

    lam_h, lam_a = model.expected_goals(home, away, neutral=neutral)
    M = predict_mod.dc_score_matrix(lam_h, lam_a, model.rho)  # (G+1, G+1), sums to 1
    g = M.shape[0]

    # Sample flattened cell indices in proportion to their probability -> vectorised.
    flat = rng.choice(g * g, size=n, p=M.ravel())
    hg, ag = np.divmod(flat, g)

    p_home = float(np.mean(hg > ag))
    p_draw = float(np.mean(hg == ag))
    p_away = float(np.mean(hg < ag))

    scores = pd.Series([f"{h}-{a}" for h, a in zip(hg, ag)]).value_counts(normalize=True)
    return {
        "home": home, "away": away,
        "lambda_home": lam_h, "lambda_away": lam_a,
        "p_home_win": p_home, "p_draw": p_draw, "p_away_win": p_away,
        "avg_home_goals": float(hg.mean()), "avg_away_goals": float(ag.mean()),
        "top_scores": scores.head(6).to_dict(),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Vectorised goal sampling helper
# --------------------------------------------------------------------------- #
def _sample_goals(model, atk, dfc, idx_a, idx_b, rng):
    """
    Given team-index arrays idx_a (nominal 'home') and idx_b, both shape (n_sims,),
    sample one scoreline per simulation at a NEUTRAL venue. `atk`/`dfc` are the
    per-team attack/defence arrays aligned to the same index space.
    Returns (goals_a, goals_b), each shape (n_sims,).
    """
    # Include the same goal-total calibration used in model.expected_goals.
    lam_a = model.avg_goals * atk[idx_a] * dfc[idx_b] * model.goal_scale
    lam_b = model.avg_goals * atk[idx_b] * dfc[idx_a] * model.goal_scale
    ga = rng.poisson(lam_a)
    gb = rng.poisson(lam_b)
    return ga, gb


def _team_arrays(model, teams):
    """Attack/defence arrays aligned to a local `teams` list (unknown -> average)."""
    atk = np.array([model.strength(t)[0] for t in teams])
    dfc = np.array([model.strength(t)[1] for t in teams])
    return atk, dfc


# --------------------------------------------------------------------------- #
# Round-robin group
# --------------------------------------------------------------------------- #
def simulate_group(model, teams: list[str], n: int | None = None,
                   points=(3, 1, 0), advance: int = 2, seed: int | None = None
                   ) -> pd.DataFrame:
    """
    Simulate a round-robin group N times (all teams play each other once, neutral
    venue) and return, per team: average points, and probability of finishing 1st
    and of finishing in the top `advance`.

    Ties in the table are broken randomly (a proxy for goal difference / h2h) so
    qualification probabilities stay well-defined.
    """
    if n is None:
        n = config.N_SIMULATIONS
    rng = np.random.default_rng(seed)

    k = len(teams)
    atk, dfc = _team_arrays(model, teams)
    idx = np.arange(k)
    total_pts = np.zeros((n, k))

    win_pts, draw_pts, loss_pts = points
    for i in range(k):
        for j in range(i + 1, k):
            ai = np.full(n, i)
            aj = np.full(n, j)
            gi, gj = _sample_goals(model, atk, dfc, ai, aj, rng)
            total_pts[:, i] += np.where(gi > gj, win_pts,
                                        np.where(gi == gj, draw_pts, loss_pts))
            total_pts[:, j] += np.where(gj > gi, win_pts,
                                        np.where(gj == gi, draw_pts, loss_pts))

    # Rank within each simulation; random tie-break via a tiny noise added to points.
    noisy = total_pts + rng.random((n, k)) * 1e-3
    order = np.argsort(-noisy, axis=1)          # team indices, best first, per sim
    rank_of_team = np.empty((n, k), dtype=int)
    rows = np.arange(n)[:, None]
    rank_of_team[rows, order] = np.arange(k)[None, :]

    p_first = np.mean(rank_of_team == 0, axis=0)
    p_advance = np.mean(rank_of_team < advance, axis=0)

    out = pd.DataFrame({
        "team": teams,
        "avg_points": total_pts.mean(axis=0),
        "P(1st)": p_first,
        f"P(top{advance})": p_advance,
    }).sort_values(f"P(top{advance})", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Single-elimination knockout bracket
# --------------------------------------------------------------------------- #
def simulate_knockout(model, teams: list[str], n: int | None = None,
                      seed: int | None = None) -> pd.DataFrame:
    """
    Simulate a single-elimination bracket N times. `teams` is given in bracket
    order (length must be a power of 2): teams[0] vs teams[1], teams[2] vs
    teams[3], ... winners meet in bracket order. Draws are resolved by a 50/50
    shootout. Returns each team's P(reach each round) and P(champion).

    Vectorised across simulations: each round loops over its (few) match slots and
    samples all N sims at once, so 10k runs of a 32-team bracket take well under a
    second.
    """
    if n is None:
        n = config.N_SIMULATIONS
    rng = np.random.default_rng(seed)

    k = len(teams)
    if k & (k - 1) != 0:
        raise ValueError(f"knockout needs a power-of-two number of teams, got {k}")
    atk, dfc = _team_arrays(model, teams)

    # current[:, s] = team index occupying slot s in this round, per simulation.
    current = np.tile(np.arange(k), (n, 1))
    reached = np.zeros(k)  # counts of reaching *each* round boundary (for a table)
    round_reach = {}       # round size -> per-team prob of being alive at that size

    size = k
    while size > 1:
        round_reach[size] = np.bincount(current.ravel(), minlength=k) / n
        winners = np.empty((n, size // 2), dtype=int)
        for s in range(0, size, 2):
            a = current[:, s]
            b = current[:, s + 1]
            ga, gb = _sample_goals(model, atk, dfc, a, b, rng)
            a_wins = ga > gb
            b_wins = gb > ga
            tie = ~(a_wins | b_wins)
            coin = rng.random(n) < 0.5           # shootout
            take_a = a_wins | (tie & coin)
            winners[:, s // 2] = np.where(take_a, a, b)
        current = winners
        size //= 2

    champions = current[:, 0]
    p_champ = np.bincount(champions, minlength=k) / n

    # Assemble a readable table: probability of reaching each round + winning.
    labels = {2: "P(final)", 4: "P(semi)", 8: "P(quarter)", 16: "P(round16)"}
    data = {"team": teams, "P(champion)": p_champ}
    for rsize, prob in round_reach.items():
        if rsize in labels and rsize != k:  # skip the trivial "everyone starts" round
            data[labels[rsize]] = prob
    out = pd.DataFrame(data).sort_values("P(champion)", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import time
    import data_loader
    import model as model_mod

    df = data_loader.load_matches(end="2022-11-19")
    m = model_mod.fit(df)

    print("--- Single match: Brazil vs Serbia (10k sims) ---")
    t0 = time.time()
    sm = simulate_match(m, "Brazil", "Serbia", n=10_000, seed=1)
    print(f"  W/D/L: {sm['p_home_win']*100:.1f}/{sm['p_draw']*100:.1f}/"
          f"{sm['p_away_win']*100:.1f}   avg score {sm['avg_home_goals']:.2f}-"
          f"{sm['avg_away_goals']:.2f}")
    print(f"  most likely scores: {sm['top_scores']}")
    print(f"  ({time.time()-t0:.3f}s)")

    print("\n--- Group stage (Group G 2022: Brazil, Switzerland, Serbia, Cameroon) ---")
    t0 = time.time()
    grp = simulate_group(m, ["Brazil", "Switzerland", "Serbia", "Cameroon"],
                         n=10_000, seed=1)
    print(grp.to_string(index=False, formatters={
        "avg_points": "{:.2f}".format, "P(1st)": "{:.2f}".format,
        "P(top2)": "{:.2f}".format}))
    print(f"  ({time.time()-t0:.3f}s)")

    print("\n--- Knockout bracket (8 top teams, 10k sims) ---")
    t0 = time.time()
    bracket = ["Brazil", "South Korea", "Netherlands", "United States",
               "Argentina", "Australia", "France", "Poland"]
    ko = simulate_knockout(m, bracket, n=10_000, seed=1)
    print(ko.to_string(index=False, formatters={
        c: "{:.3f}".format for c in ko.columns if c != "team"}))
    print(f"  ({time.time()-t0:.3f}s)")
