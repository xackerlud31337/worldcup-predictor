"""
validate.py  —  Stage 4: back-testing (do this before trusting anything)
========================================================================
We back-test on the 2018 and 2022 World Cups with a strict train/test split:
fit ONLY on matches before the tournament's first game, then predict every match
of that tournament. No look-ahead.

We report, for both the Dixon-Coles model and a naive Elo (ranking) baseline:
  * multiclass log-loss   (lower is better; punishes confident wrong calls)
  * multiclass Brier score (lower is better; mean squared prob error)
  * accuracy               (% of matches whose most-likely outcome was correct)
and a calibration table (when we say ~p, does it happen ~p of the time?).

Outcome encoding everywhere: 0 = home win, 1 = draw, 2 = away win.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import data_loader
import elo
import model as model_mod
import predict as predict_mod


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def outcome_code(home_goals: int, away_goals: int) -> int:
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-probability assigned to the actual outcomes."""
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs the one-hot truth (0..2)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == outcomes))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5
                      ) -> pd.DataFrame:
    """
    Pool ALL predicted probabilities (home/draw/away) against whether that outcome
    happened, bin by predicted probability, and compare predicted vs observed. A
    well-calibrated model has observed ≈ predicted in every bin.
    """
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    p_flat = probs.ravel()
    y_flat = onehot.ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_flat, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": f"{bins[b]:.2f}-{bins[b+1]:.2f}",
            "n": int(mask.sum()),
            "avg_predicted": float(p_flat[mask].mean()),
            "observed_freq": float(y_flat[mask].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# One tournament back-test
# --------------------------------------------------------------------------- #
def backtest_tournament(year: int, tournament: str = "FIFA World Cup",
                        ridge: float | None = None, half_life: float | None = None,
                        use_fifa: bool = False, verbose: bool = True) -> dict:
    """
    Fit before `year`'s tournament, predict its matches, return metrics + probs.

    use_fifa : seed Elo ratings from current FIFA points (config). Only valid when
    the FIFA ranking predates the tournament — safe for 2026, anachronistic (and so
    ignored by `run`) for 2018/2022.
    """
    all_df = data_loader.load_matches(min_date=config.MIN_DATE, with_weights=False)
    test = all_df[(all_df["tournament"] == tournament)
                  & (all_df["date"].dt.year == year)].copy()
    if test.empty:
        raise ValueError(f"no {tournament} matches found for {year}")

    first_day = test["date"].min()
    cutoff = first_day - pd.Timedelta(days=1)   # train strictly before kickoff

    # Training set: weighted, with the decay clock pinned to the eve of the cup.
    train = data_loader.load_matches(
        end=str(cutoff.date()), reference_date=cutoff, half_life_days=half_life
    )
    if verbose:
        print(f"\n=== {tournament} {year} ===")
        print(f"train: {len(train)} matches up to {cutoff.date()} | "
              f"test: {len(test)} matches"
              + ("  [FIFA anchor ON]" if use_fifa else ""))

    m = model_mod.fit(train, ridge=ridge)
    seeds = elo.fifa_seed_ratings() if use_fifa else None
    ratings = elo.compute_ratings(train, seed_ratings=seeds)
    draw_params = elo.fit_draw_params(
        train[train["date"] >= cutoff - pd.Timedelta(days=365 * 4)], ratings
    )

    dc_probs, elo_probs, outcomes, test_rows = [], [], [], []
    for _, row in test.iterrows():
        neutral = bool(row["neutral"])
        pred = predict_mod.predict_match(
            m, row["home_team"], row["away_team"], neutral=neutral,
            elo_ratings=ratings, draw_params=draw_params,
        )
        dc_probs.append([pred["p_home_win"], pred["p_draw"], pred["p_away_win"]])

        rh = ratings.get(row["home_team"], config.ELO_START)
        ra = ratings.get(row["away_team"], config.ELO_START)
        elo_probs.append(list(elo.wdl_probs(rh, ra, neutral=neutral,
                                            draw_base=draw_params[0],
                                            draw_decay=draw_params[1])))
        outcomes.append(outcome_code(row["home_goals"], row["away_goals"]))
        test_rows.append((row["home_team"], row["away_team"],
                          int(row["home_goals"]), int(row["away_goals"])))

    dc_probs = np.array(dc_probs)
    elo_probs = np.array(elo_probs)
    outcomes = np.array(outcomes)

    return {
        "year": year,
        "test_rows": test_rows,
        "dc": {"probs": dc_probs,
               "log_loss": log_loss(dc_probs, outcomes),
               "brier": brier(dc_probs, outcomes),
               "accuracy": accuracy(dc_probs, outcomes)},
        "elo": {"probs": elo_probs,
                "log_loss": log_loss(elo_probs, outcomes),
                "brier": brier(elo_probs, outcomes),
                "accuracy": accuracy(elo_probs, outcomes)},
        "outcomes": outcomes,
    }


def _attach_xg_targets(train: pd.DataFrame, cutoff: pd.Timestamp, alpha: float):
    """
    Return a copy of `train` with xG-blended target columns, LEAK-FREE: both the
    xG model and the per-match xG are built only from shots dated before `cutoff`.
    Returns (train_with_targets, n_matches_with_xg). Falls back to plain goals if
    no prior event data exists.
    """
    import statsbomb
    import xg as xg_mod

    shots = statsbomb.load_shots()
    shots = shots[pd.to_datetime(shots["date"]) < cutoff]
    if shots.empty:
        return train.assign(home_target=train["home_goals"].astype(float),
                            away_target=train["away_goals"].astype(float),
                            has_xg=False), 0
    xg_model = xg_mod.fit_xg_model(shots)
    match_xg = xg_mod.match_team_xg(shots, xg_model)
    blended = xg_mod.blend_into_matches(train, match_xg, alpha=alpha)
    return blended, int(blended["has_xg"].sum())


def backtest_with_xg(year: int, alpha: float = 0.7, ridge: float | None = None,
                     half_life: float | None = None, verbose: bool = True) -> dict:
    """
    Same split as backtest_tournament, but fit the Dixon-Coles model on xG-blended
    targets (leak-free). Returns metrics for the hybrid model.
    """
    all_df = data_loader.load_matches(min_date=config.MIN_DATE, with_weights=False)
    test = all_df[(all_df["tournament"] == "FIFA World Cup")
                  & (all_df["date"].dt.year == year)].copy()
    first_day = test["date"].min()
    cutoff = first_day - pd.Timedelta(days=1)

    train = data_loader.load_matches(end=str(cutoff.date()), reference_date=cutoff,
                                     half_life_days=half_life)
    train_xg, n_xg = _attach_xg_targets(train, cutoff, alpha)
    if verbose:
        print(f"[xg] WC{year}: {n_xg} training matches carry xG "
              f"(alpha={alpha}); rest use real goals")

    m = model_mod.fit(train_xg, ridge=ridge)
    ratings = elo.compute_ratings(train)
    draw_params = elo.fit_draw_params(
        train[train["date"] >= cutoff - pd.Timedelta(days=365 * 4)], ratings)

    probs, outcomes = [], []
    for _, row in test.iterrows():
        neutral = bool(row["neutral"])
        pred = predict_mod.predict_match(m, row["home_team"], row["away_team"],
                                         neutral=neutral, elo_ratings=ratings,
                                         draw_params=draw_params)
        probs.append([pred["p_home_win"], pred["p_draw"], pred["p_away_win"]])
        outcomes.append(outcome_code(row["home_goals"], row["away_goals"]))
    probs = np.array(probs); outcomes = np.array(outcomes)
    return {"year": year, "n_xg": n_xg,
            "log_loss": log_loss(probs, outcomes),
            "brier": brier(probs, outcomes),
            "accuracy": accuracy(probs, outcomes)}


def compare_xg(years=(2018, 2022), alpha: float = 0.7) -> None:
    """Head-to-head: goals-only Dixon-Coles vs xG-hybrid, on the back-test."""
    print("\n" + "=" * 70)
    print(f"xG-HYBRID vs GOALS-ONLY  (alpha={alpha}: target=alpha*xG+(1-alpha)*goals)")
    print("=" * 70)
    print(f"{'Tournament':<12}{'Model':<16}{'xG-matches':>11}{'LogLoss':>10}"
          f"{'Brier':>9}{'Acc%':>7}")
    print("-" * 70)
    for y in years:
        base = backtest_tournament(y, verbose=False)["dc"]
        hyb = backtest_with_xg(y, alpha=alpha, verbose=False)
        print(f"{'WC ' + str(y):<12}{'goals-only':<16}{'-':>11}"
              f"{base['log_loss']:>10.4f}{base['brier']:>9.4f}{base['accuracy']*100:>6.1f}%")
        print(f"{'':<12}{'xG-hybrid':<16}{hyb['n_xg']:>11}"
              f"{hyb['log_loss']:>10.4f}{hyb['brier']:>9.4f}{hyb['accuracy']*100:>6.1f}%")
        d = base["log_loss"] - hyb["log_loss"]
        verdict = "xG helps" if d > 0 else "xG hurts" if d < 0 else "tie"
        print(f"{'':<12}{'-> ' + verdict:<16}{'':>11}{d:>+10.4f} log-loss delta")
        print("-" * 70)


def _print_metrics_table(results: list[dict]) -> None:
    print("\n" + "=" * 66)
    print("BACK-TEST SUMMARY  (Dixon-Coles model vs naive Elo/ranking baseline)")
    print("=" * 66)
    header = f"{'Tournament':<14}{'Model':<14}{'LogLoss':>10}{'Brier':>10}{'Acc%':>8}"
    print(header)
    print("-" * 66)
    agg = {"dc": [], "elo": []}
    for r in results:
        for key, label in (("dc", "Dixon-Coles"), ("elo", "Elo baseline")):
            d = r[key]
            print(f"{'WC ' + str(r['year']):<14}{label:<14}"
                  f"{d['log_loss']:>10.4f}{d['brier']:>10.4f}"
                  f"{d['accuracy']*100:>7.1f}%")
            agg[key].append((d["log_loss"], d["brier"], d["accuracy"]))
        print("-" * 66)

    # Pooled averages across tournaments.
    for key, label in (("dc", "Dixon-Coles"), ("elo", "Elo baseline")):
        arr = np.array(agg[key])
        print(f"{'ALL (mean)':<14}{label:<14}"
              f"{arr[:,0].mean():>10.4f}{arr[:,1].mean():>10.4f}"
              f"{arr[:,2].mean()*100:>7.1f}%")
    print("=" * 66)
    dc = np.array(agg["dc"]); el = np.array(agg["elo"])
    better = dc[:, 0].mean() < el[:, 0].mean()
    verdict = "ADDS VALUE" if better else "does NOT beat baseline"
    print(f"Verdict (by log-loss): Dixon-Coles model {verdict} vs the ranking baseline.")


def report_misses(result: dict, top: int = 8) -> None:
    """
    Print the matches the model got most wrong — the actual results it assigned the
    lowest probability to. These upsets are where the squad-vs-record gap bites, so
    they show how much reality diverged from the historical record.
    """
    probs = result["dc"]["probs"]
    outcomes = result["outcomes"]
    rows = result["test_rows"]
    p_actual = probs[np.arange(len(outcomes)), outcomes]  # prob given to true result
    order = np.argsort(p_actual)[:top]                    # most surprising first
    label = {0: "home win", 1: "draw", 2: "away win"}
    print(f"\nHardest misses — WC {result['year']} "
          f"(actual result the model rated least likely):")
    print(f"  {'match':<34}{'score':>7}   {'model said H/D/A':>18}  p(actual)")
    for i in order:
        h, a, hg, ag = rows[i]
        ph, pd_, pa = probs[i]
        print(f"  {h + ' vs ' + a:<34}{f'{hg}-{ag}':>7}   "
              f"{ph*100:4.0f}/{pd_*100:3.0f}/{pa*100:3.0f} %   "
              f"{p_actual[i]*100:5.1f}%  [{label[outcomes[i]]}]")


def run(years=(2018, 2022, 2026), ridge: float | None = None,
        half_life: float | None = None, use_fifa_2026: bool = False) -> list[dict]:
    results = []
    for y in years:
        # FIFA anchor is only leak-safe for 2026 (ranking predates that cup only).
        uf = use_fifa_2026 and (y == 2026)
        results.append(backtest_tournament(y, ridge=ridge, half_life=half_life,
                                           use_fifa=uf))
    _print_metrics_table(results)

    # Pooled calibration across all back-test matches (Dixon-Coles).
    all_probs = np.vstack([r["dc"]["probs"] for r in results])
    all_out = np.concatenate([r["outcomes"] for r in results])
    print("\nCALIBRATION (Dixon-Coles, all back-test matches pooled)")
    print("When we predict a probability, how often does it actually happen?")
    ct = calibration_table(all_probs, all_out, n_bins=5)
    print(ct.to_string(index=False, formatters={
        "avg_predicted": "{:.3f}".format, "observed_freq": "{:.3f}".format}))

    # Hardest misses for the most recent tournament (squad-vs-record diagnostic).
    if results:
        report_misses(results[-1])
    return results


if __name__ == "__main__":
    run()
