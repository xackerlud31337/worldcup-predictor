# SportMath

**Live site: [xackerlud31337.github.io/worldcup-predictor](https://xackerlud31337.github.io/worldcup-predictor/)** —
football, UFC and NBA predictors running entirely in the browser.

## World Cup Match-Result Predictor

A Dixon-Coles Poisson model that predicts **Win / Draw / Loss** probabilities for
international football matches, validated by back-testing on the 2018 and 2022
World Cups. Built with only `pandas`, `numpy`, and `scipy`.

## What it does

- Learns a per-team **attack** and **defence** strength from ~32k historical
  international results, weighting recent and more-important matches more heavily.
- Turns any matchup into two expected-goal rates (λ_home, λ_away) and then into
  W/D/L probabilities, using the Dixon-Coles low-score correction so draws aren't
  under-predicted.
- Falls back to an Elo (ranking) prior for teams with little data.
- Simulates single matches, groups, and knockout brackets with vectorised Monte
  Carlo (10k+ scenarios in milliseconds).
- **Back-tests itself** against a naive ranking baseline before you trust it.

## Data

Source: [`martj42/international_results`](https://github.com/martj42/international_results)
— the maintained GitHub mirror of the Kaggle *"International football results
1872-present"* dataset. No API key. It already contains every FIFA World Cup match
(labelled `FIFA World Cup`) plus ~48k other internationals, so it serves both
strength-fitting *and* back-testing. The file is downloaded once and cached under
`data_cache/` (raw CSV + a cleaned CSV); nothing is re-fetched on later runs.

> openfootball's `worldcup.json` is redundant here — the results CSV already has
> all World Cup scorelines with a neutral-venue flag — so it is not used, keeping
> the pipeline to one source. Swapping in another source only means editing
> `data_loader._build_clean`.

## HTTP API

There is a self-hosted FastAPI backend (`api.py`) exposing the model as JSON
over HTTP (`GET /teams`, `POST /predict`). See **[API.md](API.md)** for full
documentation with examples.

## Install & run

```bash
pip install -r requirements.txt

# W/D/L for a neutral-venue match (World Cup default); probabilities are for team 1
python main.py Brazil Argentina

# first team at home instead of neutral
python main.py England Wales --home

# add a Monte-Carlo scoreline distribution
python main.py Brazil Argentina --sim 10000

# strongest teams right now
python main.py --rank 20

# simulate a round-robin group (qualification odds)
python main.py --group Brazil Switzerland Serbia Cameroon

# simulate a single-elim bracket (teams in bracket order, power-of-two count)
python main.py --knockout Argentina Australia Netherlands "United States" \
                          France Poland England Senegal

# run the 2018 / 2022 / 2026 back-test + hardest-misses report
python main.py --backtest

# seed strengths from the current FIFA ranking (optional; see notes)
python main.py France Spain --fifa
```

The first run fits the model (~1–2 s) and caches the fit; later runs are instant
until the data or `config.py` changes. Use `--refit` to force a refit.

**Typos are handled**: misspelled inputs are auto-corrected against the known team
list (`Equador -> Ecuador`, with an `[info]` notice), and an unrecognised name stops
with the nearest suggestions instead of silently producing garbage.

**`--fifa` (optional):** seeds each team's Elo/thin-data prior from the current FIFA
ranking points (hard-coded in `config.py`, update of 11 June 2026). In back-testing
it made **no measurable difference** — no World Cup team is data-sparse enough to
lean on the prior — so it is **off by default**. Kept as a toggle for experimenting
and for genuine debutants.

## Back-test result

Fitting only on data *before* each tournament and predicting its matches:

| Tournament | Model        | Log-loss | Brier  | Accuracy |
|------------|--------------|---------:|-------:|---------:|
| WC 2018    | Dixon-Coles  |   0.9478 | 0.5618 |    60.9% |
| WC 2022    | Dixon-Coles  |   1.0249 | 0.5970 |    54.7% |
| WC 2026*   | Dixon-Coles  |   0.8556 | 0.5068 |    64.6% |
| **Mean**   | **Dixon-Coles** | **0.9428** | **0.5552** | **60.1%** |
| Mean       | Elo baseline |   0.9770 | 0.5690 |    55.8% |

\*2026 = group stage + early knockouts present in the dataset so far.

The model beats the ranking baseline on log-loss/Brier across all three cups. It is
well-calibrated in the mid-range, though the 2026 data shows it is slightly
*over*-confident on heavy favourites (its worst misses are all favourites dropping
points — Spain 0-0 Cape Verde, Portugal 1-1 DR Congo). `python main.py --backtest`
reproduces these numbers and prints the hardest misses.

## How the model works (the short version)

For a match between home `i` and away `j` (home advantage applied only at a
non-neutral venue):

```
λ_home = avg_goals · attack(i) · defence(j) · home_adv
λ_away = avg_goals · attack(j) · defence(i)
```

- **Strengths** are fitted by weighted maximum likelihood in log-space
  (`attack = exp(a)`, `defence = exp(d)`), with an **analytic gradient** so the
  ~650-parameter fit takes a fraction of a second.
- **Recency**: each match is weighted by `0.5 ** (age / half_life)` (default
  half-life = 4 years) times a competition-importance factor (friendlies count
  less than World Cup games).
- **Dixon-Coles τ correction** adjusts the four low-scoring results (0-0, 1-0,
  0-1, 1-1) so draws are modelled correctly.
- **Thin-data prior**: an L2 (ridge) penalty shrinks under-observed teams toward
  the global average; on top of that, predictions for low-data teams are blended
  toward an Elo-ranking prediction.
- **λ → W/D/L**: the fast path is the **Skellam** distribution (difference of two
  Poissons); the default path sums the Dixon-Coles-corrected scoreline matrix
  (so it keeps τ). Both are exposed.
- **Goal-total calibration** (`GOAL_SCALE`, default 1.2): strengths are fit across
  all internationals, whose average is dragged down by cagey qualifiers, so raw
  λ's under-count goals at competitive tournaments (back-test: predicted 2.2 vs
  actual 2.66 goals/game). Scaling both λ's by 1.2 makes totals match reality *and*
  lowers W/D/L log-loss (0.9937 → 0.9869), because under-counting goals had made the
  model too draw-heavy. It cancels in the home/away ratio, so it mainly makes
  **scorelines** realistic without distorting who-wins.

> **Reading the output:** trust the **expected goals** and the **win/draw/loss %**,
> not the single "most likely scoreline". Football is high-variance — even for a
> heavy favourite the single most common score is only ~10% likely, so a 4-1 result
> when the top line reads 2-1 is the model being right about a *range*, not wrong.

## Project layout

| File            | Stage | Responsibility |
|-----------------|-------|----------------|
| `config.py`     | —     | All tunable parameters (half-life, home adv, ridge, dates, xG alpha) |
| `data_loader.py`| 1     | Download/cache/clean results, canonical team names, weighting |
| `elo.py`        | —     | Self-contained Elo: naive baseline **and** thin-data prior |
| `model.py`      | 2     | Dixon-Coles fit → attack/defence/home_adv/ρ (analytic gradient) |
| `predict.py`    | 3     | λ → W/D/L via Skellam or DC matrix; Elo fallback blend |
| `validate.py`   | 4     | 2018/2022 back-test, baseline comparison, calibration, xG comparison |
| `simulate.py`   | 5     | Vectorised Monte Carlo: match, group, knockout |
| `statsbomb.py`  | xG    | Download StatsBomb event data, extract shots with x/y coordinates |
| `xg.py`         | xG    | Expected-goals model from shot geometry; blend xG into the fit |
| `main.py`       | —     | CLI wiring it all together |

Each module has a runnable `__main__` smoke test (e.g. `python model.py`).

## Optional: xG (expected-goals) hybrid

Off by default; enable with `--xg`. Instead of raw goals, matches that have event
data use a blended target `alpha·xG + (1-alpha)·goals` in the fit.

- **Source**: [StatsBomb Open Data](https://github.com/statsbomb/open-data) (free,
  no key) — shot-by-shot x/y coordinates for recent World Cups, Euros, Copa América.
- **xG model** (`xg.py`): a self-contained logistic regression on each shot's
  **distance** and **angle** to goal (plus header / penalty / free-kick flags).
  It reproduces StatsBomb's own xG at **0.923 correlation** and is calibrated
  (predicted xG totals match actual goals).
- **The honest verdict**: back-testing (`python -c "import validate; validate.compare_xg()"`)
  shows xG helps only **marginally** — WC 2022 log-loss 1.0246 → 1.0233, accuracy
  54.7% → 56.2%. Why: free event data covers only ~5 tournaments (~30 elite teams),
  so ~99% of the 32k training matches — and every minnow/qualifier — still use real
  goals. It sharpens the elite teams slightly; it can't help a team with no event
  data (e.g. most of Africa/Asia/CONCACAF). It is therefore **opt-in, not default**.

```bash
python main.py France Spain --xg     # both elite -> small shift
python statsbomb.py                   # download + inspect the shot data
python xg.py                          # train + validate the xG model
```

First `--xg` run downloads ~260 event files (threaded, ~1 min) and caches compact
shot tables under `data_cache/statsbomb/`.

## Tuning

Everything worth changing is in `config.py`. The `half_life` and `ridge`
parameters were chosen from the sweep in the back-test; to re-tune:

```python
import numpy as np, validate as v
for ridge in [0.25, 0.5, 1.0]:
    for hl in [730, 1460, 2190]:
        res = [v.backtest_tournament(y, ridge=ridge, half_life=hl, verbose=False)
               for y in (2018, 2022)]
        print(ridge, hl, np.mean([r['dc']['log_loss'] for r in res]))
```

## Notes / possible next steps

Kept deliberately simple (team strengths only) per the brief. If back-testing
motivates it, natural additions are: separate home/away strengths, a bivariate
Poisson correlation term, opponent-confederation adjustments, or player-availability
features. Add them only if they improve the back-test.
