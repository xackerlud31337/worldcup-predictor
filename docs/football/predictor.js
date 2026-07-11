/*
 * predictor.js — browser port of the Python prediction logic
 * ===========================================================
 * Mirrors predict.py, simulate.py and elo.wdl_probs, running on the fitted
 * parameters exported by export_model.py into model_data.js (MODEL_DATA).
 * Same math, same defaults:
 *
 *   lambda_home = avg_goals * attack(home) * defence(away) * home_adv * goal_scale
 *   lambda_away = avg_goals * attack(away) * defence(home) * goal_scale
 *
 * then W/D/L via the Dixon-Coles-corrected scoreline matrix (or Skellam), with
 * the thin-data Elo blend on top, and Monte-Carlo scoreline sampling from the
 * same DC matrix.
 */

const CFG = MODEL_DATA.config;

function getBundle(useXg) {
  return (useXg && MODEL_DATA.xg) ? MODEL_DATA.xg : MODEL_DATA.base;
}

/* ---------------------------------------------------------------- *
 * model.FittedModel equivalents
 * ---------------------------------------------------------------- */
// (attack, defence, weighted_match_count); unknown team -> league average.
function strength(bundle, team) {
  const t = bundle.teams[team];
  return t ? [t.a, t.d, t.w] : [1.0, 1.0, 0.0];
}

function expectedGoals(bundle, home, away, neutral) {
  const [atkH, defH] = strength(bundle, home);
  const [atkA, defA] = strength(bundle, away);
  const ha = neutral ? 1.0 : bundle.home_adv;
  const lamHome = bundle.avg_goals * atkH * defA * ha * bundle.goal_scale;
  const lamAway = bundle.avg_goals * atkA * defH * bundle.goal_scale;
  return [lamHome, lamAway];
}

/* ---------------------------------------------------------------- *
 * predict.py: lambda -> W/D/L
 * ---------------------------------------------------------------- */
// Poisson pmf for k = 0..maxK, computed iteratively (no factorials needed).
function poissonPmf(lam, maxK) {
  const p = new Array(maxK + 1);
  p[0] = Math.exp(-lam);
  for (let k = 1; k <= maxK; k++) p[k] = p[k - 1] * lam / k;
  return p;
}

function normalise3(pHome, pDraw, pAway) {
  const p = [Math.max(pHome, 1e-9), Math.max(pDraw, 1e-9), Math.max(pAway, 1e-9)];
  const s = p[0] + p[1] + p[2];
  return [p[0] / s, p[1] / s, p[2] / s];
}

// Joint P(home scores x, away scores y) with the Dixon-Coles tau correction
// on the four low-score cells. Flat array, M[x * (g+1) + y].
function dcScoreMatrix(lamHome, lamAway, rho, maxGoals = CFG.max_goals) {
  const g = maxGoals + 1;
  const hg = poissonPmf(lamHome, maxGoals);
  const ag = poissonPmf(lamAway, maxGoals);
  const M = new Float64Array(g * g);
  for (let x = 0; x < g; x++)
    for (let y = 0; y < g; y++) M[x * g + y] = hg[x] * ag[y];

  M[0] *= 1.0 - lamHome * lamAway * rho;          // (0,0)
  M[1] *= 1.0 + lamHome * rho;                    // (0,1)
  M[g] *= 1.0 + lamAway * rho;                    // (1,0)
  M[g + 1] *= 1.0 - rho;                          // (1,1)

  let sum = 0;
  for (let i = 0; i < M.length; i++) { if (M[i] < 0) M[i] = 0; sum += M[i]; }
  for (let i = 0; i < M.length; i++) M[i] /= sum;
  return M;
}

function dcMatrixWdl(lamHome, lamAway, rho) {
  const M = dcScoreMatrix(lamHome, lamAway, rho);
  const g = CFG.max_goals + 1;
  let pHome = 0, pDraw = 0, pAway = 0;
  for (let x = 0; x < g; x++)
    for (let y = 0; y < g; y++) {
      const p = M[x * g + y];
      if (x > y) pHome += p; else if (x === y) pDraw += p; else pAway += p;
    }
  return normalise3(pHome, pDraw, pAway);
}

// Skellam W/D/L = triangle sums of the independent-Poisson grid (no tau).
// A generous truncation stands in for scipy's closed form; the tail beyond
// 30 goals is far below the 1e-9 clip used in normalisation.
function skellamWdl(lamHome, lamAway) {
  const maxK = 30;
  const hg = poissonPmf(lamHome, maxK);
  const ag = poissonPmf(lamAway, maxK);
  let pHome = 0, pDraw = 0, pAway = 0;
  for (let x = 0; x <= maxK; x++)
    for (let y = 0; y <= maxK; y++) {
      const p = hg[x] * ag[y];
      if (x > y) pHome += p; else if (x === y) pDraw += p; else pAway += p;
    }
  return normalise3(pHome, pDraw, pAway);
}

/* ---------------------------------------------------------------- *
 * elo.wdl_probs — the thin-data fallback prior
 * ---------------------------------------------------------------- */
function eloWdlProbs(rHome, rAway, neutral, drawBase, drawDecay) {
  const adv = neutral ? 0.0 : CFG.elo_home_adv;
  const dr = (rHome + adv) - rAway;
  const eHome = 1.0 / (1.0 + Math.pow(10.0, -dr / 400.0));

  const pDraw = drawBase * Math.exp(-Math.abs(dr) / drawDecay);
  const pHome = eHome - 0.5 * pDraw;
  const pAway = 1.0 - pHome - pDraw;

  const p = [Math.max(pHome, 1e-6), Math.max(pDraw, 1e-6), Math.max(pAway, 1e-6)];
  const s = p[0] + p[1] + p[2];
  return [p[0] / s, p[1] / s, p[2] / s];
}

// Confidence in the DC estimate for this matchup, in [0, 1], driven by the
// weaker-supported of the two teams (predict._data_trust).
function dataTrust(bundle, home, away) {
  const thr = CFG.min_matches_for_full_trust;
  const wh = strength(bundle, home)[2];
  const wa = strength(bundle, away)[2];
  return Math.min(1.0, wh / thr) * Math.min(1.0, wa / thr);
}

/* ---------------------------------------------------------------- *
 * predict.predict_match
 * ---------------------------------------------------------------- */
function predictMatch(bundle, home, away, neutral = true, method = "dc") {
  const [lamHome, lamAway] = expectedGoals(bundle, home, away, neutral);

  const p = method === "skellam"
    ? skellamWdl(lamHome, lamAway)
    : dcMatrixWdl(lamHome, lamAway, bundle.rho);

  const trust = dataTrust(bundle, home, away);
  let pFinal = p;
  if (trust < 1.0) {
    const rh = bundle.elo[home] ?? CFG.elo_start;
    const ra = bundle.elo[away] ?? CFG.elo_start;
    const [base, decay] = bundle.draw_params;
    const pElo = eloWdlProbs(rh, ra, neutral, base, decay);
    // Convex blend: full trust -> pure DC; no trust -> pure Elo prior.
    pFinal = normalise3(
      trust * p[0] + (1 - trust) * pElo[0],
      trust * p[1] + (1 - trust) * pElo[1],
      trust * p[2] + (1 - trust) * pElo[2],
    );
  }

  return {
    home, away, neutral,
    lambda_home: lamHome, lambda_away: lamAway,
    p_home_win: pFinal[0], p_draw: pFinal[1], p_away_win: pFinal[2],
    data_trust: trust, method,
  };
}

/* ---------------------------------------------------------------- *
 * simulate.simulate_match — Monte-Carlo scorelines from the DC matrix
 * ---------------------------------------------------------------- */
function simulateMatch(bundle, home, away, neutral = true, n = 10000) {
  const [lamH, lamA] = expectedGoals(bundle, home, away, neutral);
  const M = dcScoreMatrix(lamH, lamA, bundle.rho);
  const g = CFG.max_goals + 1;

  // Cumulative distribution over the flattened cells -> inverse-CDF sampling.
  const cum = new Float64Array(M.length);
  let acc = 0;
  for (let i = 0; i < M.length; i++) { acc += M[i]; cum[i] = acc; }

  const counts = new Uint32Array(M.length);
  for (let s = 0; s < n; s++) {
    const u = Math.random() * acc;
    // Binary search for the first cell with cum >= u.
    let lo = 0, hi = M.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < u) lo = mid + 1; else hi = mid;
    }
    counts[lo]++;
  }

  let pHome = 0, pDraw = 0, pAway = 0, sumH = 0, sumA = 0;
  const scored = [];
  for (let i = 0; i < counts.length; i++) {
    if (!counts[i]) continue;
    const hg = Math.floor(i / g), ag = i % g, c = counts[i];
    if (hg > ag) pHome += c; else if (hg === ag) pDraw += c; else pAway += c;
    sumH += hg * c;
    sumA += ag * c;
    scored.push([`${hg}-${ag}`, c / n]);
  }
  scored.sort((a, b) => b[1] - a[1]);
  const topScores = Object.fromEntries(scored.slice(0, 6));

  return {
    home, away,
    lambda_home: lamH, lambda_away: lamA,
    p_home_win: pHome / n, p_draw: pDraw / n, p_away_win: pAway / n,
    avg_home_goals: sumH / n, avg_away_goals: sumA / n,
    top_scores: topScores,
    n,
  };
}

/* ---------------------------------------------------------------- *
 * api.predict equivalent — same response shape the web UI consumed
 * ---------------------------------------------------------------- */
// Exact top-k scorelines straight from the DC probability matrix —
// deterministic, so repeated runs always show the same chart.
function topScorelines(bundle, home, away, neutral = true, k = 6) {
  const [lamH, lamA] = expectedGoals(bundle, home, away, neutral);
  const M = dcScoreMatrix(lamH, lamA, bundle.rho);
  const g = CFG.max_goals + 1;
  const all = [];
  for (let x = 0; x < g; x++)
    for (let y = 0; y < g; y++) all.push([`${x}-${y}`, M[x * g + y]]);
  all.sort((a, b) => b[1] - a[1]);
  return Object.fromEntries(all.slice(0, k));
}

// Probabilities grouped by victory margin — "France by 2+" style aggregates
// that don't fragment a strong side's dominance across many exact scorelines.
function marginBreakdown(bundle, home, away, neutral = true) {
  const [lamH, lamA] = expectedGoals(bundle, home, away, neutral);
  const M = dcScoreMatrix(lamH, lamA, bundle.rho);
  const g = CFG.max_goals + 1;
  const out = { home3: 0, home2: 0, home1: 0, draw: 0, away1: 0, away2: 0, away3: 0 };
  for (let x = 0; x < g; x++)
    for (let y = 0; y < g; y++) {
      const p = M[x * g + y], d = x - y;
      if (d >= 3) out.home3 += p;
      else if (d === 2) out.home2 += p;
      else if (d === 1) out.home1 += p;
      else if (d === 0) out.draw += p;
      else if (d === -1) out.away1 += p;
      else if (d === -2) out.away2 += p;
      else out.away3 += p;
    }
  return out;
}

// Common betting markets with model probabilities. 1X2-derived markets use the
// final blended probabilities; goals markets come from the DC scoreline matrix.
function bettingMarkets(bundle, home, away, neutral, pH, pD, pA) {
  const [lamH, lamA] = expectedGoals(bundle, home, away, neutral);
  const M = dcScoreMatrix(lamH, lamA, bundle.rho);
  const g = CFG.max_goals + 1;
  let o15 = 0, o25 = 0, o35 = 0, btts = 0;
  for (let x = 0; x < g; x++)
    for (let y = 0; y < g; y++) {
      const p = M[x * g + y], t = x + y;
      if (t >= 2) o15 += p;
      if (t >= 3) o25 += p;
      if (t >= 4) o35 += p;
      if (x >= 1 && y >= 1) btts += p;
    }
  return [
    { label: `${home} or draw (1X)`, p: pH + pD },
    { label: `${away} or draw (X2)`, p: pD + pA },
    { label: `Either team wins (12)`, p: pH + pA },
    { label: `${home} draw-no-bet`, p: pH / (pH + pA) },
    { label: `${away} draw-no-bet`, p: pA / (pH + pA) },
    { label: "Over 1.5 goals", p: o15 },
    { label: "Over 2.5 goals", p: o25 },
    { label: "Under 2.5 goals", p: 1 - o25 },
    { label: "Over 3.5 goals", p: o35 },
    { label: "Under 3.5 goals", p: 1 - o35 },
    { label: "Both teams score — yes", p: btts },
    { label: "Both teams score — no", p: 1 - btts },
  ].sort((a, b) => b.p - a.p);
}

function runPrediction({ home_team, away_team, use_xg = false,
                         home_advantage = false, method = "dc" }) {
  const bundle = getBundle(use_xg);
  const neutral = !home_advantage;

  const pred = predictMatch(bundle, home_team, away_team, neutral, method);

  return {
    home: home_team,
    away: away_team,
    neutral,
    method,
    use_xg,
    probabilities: {
      home_win: pred.p_home_win,
      draw: pred.p_draw,
      away_win: pred.p_away_win,
    },
    expected_goals: {
      home: pred.lambda_home,
      away: pred.lambda_away,
    },
    data_trust: pred.data_trust,
    margins: marginBreakdown(bundle, home_team, away_team, neutral),
    markets: bettingMarkets(bundle, home_team, away_team, neutral,
                            pred.p_home_win, pred.p_draw, pred.p_away_win),
    simulation: {
      top_scores: topScorelines(bundle, home_team, away_team, neutral),
    },
  };
}

function listTeams() {
  return Object.keys(MODEL_DATA.base.teams);
}
