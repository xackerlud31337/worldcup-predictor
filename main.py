#!/usr/bin/env python3
"""
main.py  —  CLI entry point
===========================
Fit the Dixon-Coles model on all data up to today (recency-weighted) and answer
questions from the command line.

Examples
--------
  # W/D/L for a neutral-venue match (World Cup default)
  python main.py Brazil Argentina

  # treat the first team as playing at home
  python main.py England Wales --home

  # Monte-Carlo scoreline distribution for one match
  python main.py Brazil Argentina --sim 10000

  # simulate a round-robin group (qualification odds)
  python main.py --group Brazil Switzerland Serbia Cameroon

  # simulate a single-elimination bracket (teams in bracket order, power of two)
  python main.py --knockout Brazil Korea Netherlands USA Argentina Australia France Poland

  # show the strongest teams right now
  python main.py --rank 20

  # run the 2018 & 2022 back-test / validation
  python main.py --backtest
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from datetime import date

import config
import data_loader
import elo
import model as model_mod
import predict as predict_mod
import simulate as simulate_mod


# --------------------------------------------------------------------------- #
# Fit-once, cache-to-disk so repeated CLI calls are instant
# --------------------------------------------------------------------------- #
def _cache_key(use_xg: bool, use_fifa: bool) -> str:
    """Hash the settings that affect the fit so the cache invalidates on change."""
    raw = (f"{config.MIN_DATE}|{config.HALF_LIFE_DAYS}|{config.RIDGE_LAMBDA}"
           f"|gs{config.GOAL_SCALE}|xg{use_xg}:{config.XG_ALPHA}"
           f"|fifa{use_fifa}|{date.today()}")
    clean = data_loader._clean_cache_path()
    mtime = os.path.getmtime(clean) if os.path.exists(clean) else 0
    return hashlib.md5(f"{raw}|{mtime}".encode()).hexdigest()[:12]


def build_current_model(force: bool = False, use_xg: bool = False,
                        use_fifa: bool | None = None):
    """Return (fitted_model, elo_ratings, draw_params), cached to disk per config."""
    if use_fifa is None:
        use_fifa = config.USE_FIFA_ANCHOR
    config.USE_FIFA_ANCHOR = use_fifa   # so the build below sees the effective value
    data_loader._ensure_cache_dir()
    cache = os.path.join(config.CACHE_DIR, f"fit_{_cache_key(use_xg, use_fifa)}.pkl")
    if os.path.exists(cache) and not force:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    print("[main] fitting model on all data up to today ...")
    df = data_loader.load_matches()  # all data, recency-weighted to today

    if use_xg:
        # Blend StatsBomb xG into the covered (elite-tournament) matches. No leak
        # concern here: we are predicting the future, not a held-out past event.
        import statsbomb
        import xg as xg_mod
        print("[main] --xg: loading event data & blending expected goals ...")
        shots = statsbomb.load_shots()
        xg_model = xg_mod.fit_xg_model(shots)
        match_xg = xg_mod.match_team_xg(shots, xg_model)
        df = xg_mod.blend_into_matches(df, match_xg, alpha=config.XG_ALPHA)
        print(f"[main]   {int(df['has_xg'].sum())} matches upgraded to xG "
              f"(alpha={config.XG_ALPHA})")

    m = model_mod.fit(df)
    seeds = elo.fifa_seed_ratings() if config.USE_FIFA_ANCHOR else None
    if seeds:
        print(f"[main] FIFA anchor ON: seeding Elo from {len(seeds)} ranked teams")
    ratings = elo.compute_ratings(df, seed_ratings=seeds)
    # Fit the baseline draw model on the last few years for the Elo fallback.
    import pandas as pd
    recent = df[df["date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=365 * 4)]
    draw_params = elo.fit_draw_params(recent, ratings)

    bundle = (m, ratings, draw_params)
    with open(cache, "wb") as fh:
        pickle.dump(bundle, fh)
    return bundle


# --------------------------------------------------------------------------- #
# Team-name resolution (fuzzy matching for typos)
# --------------------------------------------------------------------------- #
def resolve_team(name: str, known) -> str:
    """
    Map a user-typed team name to a known team.

    * exact / case-insensitive / known-alias match -> return it silently
    * close match (difflib ratio >= 0.8) -> auto-correct with an [info] notice
    * otherwise -> print the nearest candidates and STOP (SystemExit)

    This stops typos like "Equador"/"Columbia" from silently falling through to
    the league-average prior and producing garbage. Real-but-sparse teams match
    exactly here, so their genuine thin-data Elo fallback is left untouched.
    """
    import difflib

    # Honour the canonical alias map first (e.g. "IR Iran" -> "Iran").
    canon = data_loader.canonicalize(name)
    known_list = list(known)
    lower = {k.lower(): k for k in known_list}

    if canon in known:
        return canon
    if canon.lower() in lower:
        return lower[canon.lower()]

    close = difflib.get_close_matches(canon.lower(), list(lower.keys()), n=3, cutoff=0.8)
    if close:
        corrected = lower[close[0]]
        print(f"[info] '{name}' -> assuming '{corrected}'")
        return corrected

    # No confident match: suggest the nearest names and refuse to guess.
    nearest = [lower[c] for c in
               difflib.get_close_matches(canon.lower(), list(lower.keys()), n=5, cutoff=0.4)]
    hint = ("  Did you mean: " + ", ".join(nearest) + "?") if nearest else \
           "  (no similar team found; check spelling)"
    raise SystemExit(f"[error] team '{name}' not recognised.\n{hint}")


def resolve_all(names, known) -> list[str]:
    return [resolve_team(n, known) for n in names]


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_predict(args, m, ratings, draw_params):
    home = resolve_team(args.teams[0], m.teams)
    away = resolve_team(args.teams[1], m.teams)
    neutral = not args.home
    pred = predict_mod.predict_match(
        m, home, away, neutral=neutral, method=args.method,
        elo_ratings=ratings, draw_params=draw_params,
    )
    print(predict_mod.format_prediction(pred))

    if args.sim:
        print(f"\nMonte-Carlo scoreline distribution ({args.sim:,} sims):")
        sm = simulate_mod.simulate_match(m, home, away, neutral=neutral, n=args.sim)
        print(f"  W/D/L: {sm['p_home_win']*100:.1f} / {sm['p_draw']*100:.1f} / "
              f"{sm['p_away_win']*100:.1f}")
        print(f"  avg score: {sm['avg_home_goals']:.2f} - {sm['avg_away_goals']:.2f}")
        print("  most likely scorelines:")
        for score, prob in sm["top_scores"].items():
            print(f"    {score}: {prob*100:.1f}%")


def cmd_group(args, m, ratings, draw_params):
    teams = resolve_all(args.group, m.teams)
    print(f"Group simulation ({args.n_sims:,} sims): {', '.join(teams)}")
    table = simulate_mod.simulate_group(m, teams, n=args.n_sims)
    print(table.to_string(index=False, formatters={
        c: "{:.2f}".format for c in table.columns if c != "team"}))


def cmd_knockout(args, m, ratings, draw_params):
    teams = resolve_all(args.knockout, m.teams)
    print(f"Knockout simulation ({args.n_sims:,} sims), bracket order:")
    print("  " + " | ".join(teams))
    table = simulate_mod.simulate_knockout(m, teams, n=args.n_sims)
    print(table.to_string(index=False, formatters={
        c: "{:.3f}".format for c in table.columns if c != "team"}))


def cmd_rank(args, m, ratings, draw_params):
    print(f"Top {args.rank} teams by current attack/defence strength:")
    r = m.rank(args.rank)
    print(r.to_string(index=False, formatters={
        "attack": "{:.2f}".format, "defence": "{:.2f}".format,
        "strength": "{:.2f}".format}))


def cmd_backtest(args):
    import validate
    validate.run()


# --------------------------------------------------------------------------- #
# Arg parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="World Cup match-result predictor (Dixon-Coles Poisson model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Outcome probabilities are for the FIRST team (home). "
               "Matches are treated as neutral-venue unless --home is given.",
    )
    p.add_argument("teams", nargs="*", help="two team names for a single prediction")
    p.add_argument("--home", action="store_true",
                   help="treat the first team as playing at home (default: neutral)")
    p.add_argument("--method", choices=["dc", "skellam"], default="dc",
                   help="probability method: dc=Dixon-Coles matrix (default), skellam=fast")
    p.add_argument("--sim", type=int, metavar="N",
                   help="also Monte-Carlo N scorelines for the single match")
    p.add_argument("--group", nargs="+", metavar="TEAM",
                   help="simulate a round-robin group of these teams")
    p.add_argument("--knockout", nargs="+", metavar="TEAM",
                   help="simulate a single-elim bracket (teams in bracket order, 2^k)")
    p.add_argument("--rank", type=int, metavar="N", help="print the top-N teams")
    p.add_argument("--backtest", action="store_true",
                   help="run the 2018 & 2022 validation back-test")
    p.add_argument("--n-sims", type=int, default=config.N_SIMULATIONS,
                   help=f"sims for group/knockout (default {config.N_SIMULATIONS})")
    p.add_argument("--refit", action="store_true", help="ignore the cached fit")
    p.add_argument("--xg", action="store_true",
                   help="blend StatsBomb expected-goals into elite-team strengths "
                        "(downloads event data on first use; marginal effect)")
    p.add_argument("--fifa", action="store_true",
                   help="seed Elo / thin-data prior from current FIFA ranking points")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Back-test doesn't need the 'current' model bundle.
    if args.backtest:
        cmd_backtest(args)
        return

    use_fifa = True if args.fifa else None  # None -> config default
    m, ratings, draw_params = build_current_model(force=args.refit, use_xg=args.xg,
                                                  use_fifa=use_fifa)

    if args.group:
        cmd_group(args, m, ratings, draw_params)
    elif args.knockout:
        cmd_knockout(args, m, ratings, draw_params)
    elif args.rank:
        cmd_rank(args, m, ratings, draw_params)
    elif len(args.teams) == 2:
        cmd_predict(args, m, ratings, draw_params)
    elif len(args.teams) == 0:
        build_parser().print_help()
    else:
        raise SystemExit("Provide exactly TWO team names, or use "
                         "--group/--knockout/--rank/--backtest.")


if __name__ == "__main__":
    main()
