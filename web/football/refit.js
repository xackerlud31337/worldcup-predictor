/*
 * refit.js — the CLI's --refit, in the browser
 * ============================================
 * Downloads the latest results CSV from the same GitHub mirror the Python
 * pipeline uses and rebuilds the whole model bundle client-side, mirroring
 * data_loader.py (clean + canonicalise + weight), model.fit (same weighted
 * Dixon-Coles likelihood and analytic gradient, optimised with Adam instead
 * of scipy's L-BFGS), elo.compute_ratings and elo.fit_draw_params.
 *
 * The result is a bundle with the exact shape of MODEL_DATA.base, so
 * predictor.js can use it unchanged.
 */

const RESULTS_URL =
  "https://raw.githubusercontent.com/martj42/international_results/master/results.csv";

// ---- constants mirrored from config.py ---------------------------------- //
const REFIT_CFG = {
  min_date: "1990-01-01",
  half_life_days: 1460.0,
  ridge: 0.2,
  home_adv_init: 0.25,
  rho_init: -0.05,
  rho_bounds: [-0.2, 0.2],
  goal_scale: 1.2,
  elo_k: 40.0,
  elo_home_adv: 65.0,
  elo_start: 1500.0,
};

const COMPETITION_WEIGHTS = [
  ["fifa world cup", 1.00],
  ["qualification", 0.90],
  ["uefa euro", 1.00],
  ["copa américa", 1.00],
  ["african cup of nations", 1.00],
  ["afc asian cup", 1.00],
  ["gold cup", 0.90],
  ["uefa nations league", 0.95],
  ["confederations cup", 0.95],
  ["friendly", 0.55],
];
const COMPETITION_DEFAULT = 0.80;

// data_loader.CANONICAL_TEAM_NAMES — one spelling per modern national team.
const CANONICAL_TEAM_NAMES = {
  "West Germany": "Germany", "East Germany": "Germany",
  "Soviet Union": "Russia", "CIS": "Russia",
  "Czechoslovakia": "Czechia", "Czech Republic": "Czechia",
  "Yugoslavia": "Serbia", "Serbia and Montenegro": "Serbia", "FR Yugoslavia": "Serbia",
  "Zaïre": "DR Congo", "Zaire": "DR Congo", "Congo DR": "DR Congo",
  "IR Iran": "Iran",
  "Korea Republic": "South Korea", "Korea DPR": "North Korea",
  "China PR": "China",
  "USA": "United States",
  "Cape Verde Islands": "Cape Verde", "Cabo Verde": "Cape Verde",
  "Türkiye": "Turkey", "Turkiye": "Turkey",
  "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
  "Kyrgyz Republic": "Kyrgyzstan",
  "The Gambia": "Gambia",
  "Curacao": "Curaçao",
  "St Kitts and Nevis": "Saint Kitts and Nevis",
  "St Lucia": "Saint Lucia",
  "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
  "Brunei Darussalam": "Brunei",
  "Republic of Ireland": "Ireland",
};

function canonicalizeTeam(name) {
  const t = name.trim();
  return CANONICAL_TEAM_NAMES[t] || t;
}

/* ---------------------------------------------------------------- *
 * CSV parsing + cleaning (data_loader._build_clean)
 * ---------------------------------------------------------------- */
// Minimal CSV parser with quoted-field support (tournament/city names).
function parseCsvLine(line) {
  const out = [];
  let cur = "", inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQ) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; } else inQ = false;
      } else cur += ch;
    } else if (ch === '"') inQ = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

function cleanMatches(csvText) {
  const lines = csvText.split("\n");
  const header = parseCsvLine(lines[0].trim());
  const col = Object.fromEntries(header.map((h, i) => [h, i]));
  const minTime = new Date(REFIT_CFG.min_date + "T00:00:00Z").getTime();

  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const f = parseCsvLine(line);
    const dateStr = f[col.date];
    const hg = parseInt(f[col.home_score], 10);
    const ag = parseInt(f[col.away_score], 10);
    if (!dateStr || Number.isNaN(hg) || Number.isNaN(ag)) continue;
    const time = new Date(dateStr + "T00:00:00Z").getTime();
    if (Number.isNaN(time) || time < minTime) continue;
    rows.push({
      time,
      home: canonicalizeTeam(f[col.home_team]),
      away: canonicalizeTeam(f[col.away_team]),
      hg, ag,
      tournament: f[col.tournament] || "",
      neutral: String(f[col.neutral]).toUpperCase() === "TRUE",
    });
  }
  rows.sort((a, b) => a.time - b.time); // Elo needs chronological order
  return rows;
}

// data_loader.add_weights: competition importance x exponential time decay.
function addWeights(rows) {
  const now = Date.now();
  const compCache = new Map();
  for (const r of rows) {
    let cw = compCache.get(r.tournament);
    if (cw === undefined) {
      const t = r.tournament.toLowerCase();
      cw = COMPETITION_DEFAULT;
      for (const [key, w] of COMPETITION_WEIGHTS)
        if (t.includes(key)) { cw = w; break; }
      compCache.set(r.tournament, cw);
    }
    const ageDays = Math.max(0, Math.floor((now - r.time) / 86400000));
    r.weight = cw * Math.pow(0.5, ageDays / REFIT_CFG.half_life_days);
  }
}

/* ---------------------------------------------------------------- *
 * model.fit — weighted Dixon-Coles MLE with the analytic gradient
 * ---------------------------------------------------------------- */
function fitDixonColes(rows, onProgress) {
  const teams = [...new Set(rows.flatMap(r => [r.home, r.away]))].sort();
  const index = new Map(teams.map((t, i) => [t, i]));
  const n = teams.length;
  const m = rows.length;

  const hi = new Int32Array(m), ai = new Int32Array(m);
  const x = new Float64Array(m), y = new Float64Array(m);
  const w = new Float64Array(m), hm = new Float64Array(m);
  let totalW = 0, goalW = 0;
  for (let k = 0; k < m; k++) {
    const r = rows[k];
    hi[k] = index.get(r.home); ai[k] = index.get(r.away);
    x[k] = r.hg; y[k] = r.ag; w[k] = r.weight;
    hm[k] = r.neutral ? 0.0 : 1.0;
    totalW += r.weight;
    goalW += r.weight * (r.hg + r.ag);
  }
  const avgGoals = goalW / (2 * totalW);
  const C = Math.log(avgGoals);
  const ridge = REFIT_CFG.ridge;

  // theta = [a(n), d(n), h, rho]
  const P = 2 * n + 2;
  const theta = new Float64Array(P);
  theta[2 * n] = REFIT_CFG.home_adv_init;
  theta[2 * n + 1] = REFIT_CFG.rho_init;
  const grad = new Float64Array(P);

  function objective() {
    grad.fill(0);
    const h = theta[2 * n], rho = theta[2 * n + 1];
    let nll = 0, gH = 0, gRho = 0;
    for (let k = 0; k < m; k++) {
      const iH = hi[k], iA = ai[k];
      const etaH = C + theta[iH] + theta[n + iA] + h * hm[k];
      const etaA = C + theta[iA] + theta[n + iH];
      const lam = Math.exp(etaH), mu = Math.exp(etaA);
      const xk = x[k], yk = y[k], wk = w[k];

      // Dixon-Coles tau on the four low-score cells, plus its partials.
      let tau = 1.0, dtl = 0.0, dtm = 0.0, dtr = 0.0;
      if (xk === 0 && yk === 0) {
        tau = 1.0 - lam * mu * rho; dtl = -mu * rho; dtm = -lam * rho; dtr = -lam * mu;
      } else if (xk === 0 && yk === 1) {
        tau = 1.0 + lam * rho; dtl = rho; dtr = lam;
      } else if (xk === 1 && yk === 0) {
        tau = 1.0 + mu * rho; dtm = rho; dtr = mu;
      } else if (xk === 1 && yk === 1) {
        tau = 1.0 - rho; dtr = -1.0;
      }
      if (tau < 1e-10) tau = 1e-10;

      nll -= wk * (xk * etaH - lam + yk * etaA - mu + Math.log(tau));

      const gEtaH = wk * (xk - lam + (dtl * lam) / tau);
      const gEtaA = wk * (yk - mu + (dtm * mu) / tau);
      grad[iH] -= gEtaH;          // d nll / d a[home]
      grad[iA] -= gEtaA;          // d nll / d a[away]
      grad[n + iA] -= gEtaH;      // d nll / d d[away]
      grad[n + iH] -= gEtaA;      // d nll / d d[home]
      gH += gEtaH * hm[k];
      gRho += wk * (dtr / tau);
    }
    for (let i = 0; i < 2 * n; i++) {
      nll += ridge * theta[i] * theta[i];
      grad[i] += 2.0 * ridge * theta[i];
    }
    grad[2 * n] = -gH;
    grad[2 * n + 1] = -gRho;
    return nll;
  }

  // Full-batch Adam (the likelihood is smooth and ridge-regularised, so this
  // converges to the same optimum L-BFGS finds, just in more steps).
  const ITERS = 1500;
  const mAdam = new Float64Array(P), vAdam = new Float64Array(P);
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const [rhoLo, rhoHi] = REFIT_CFG.rho_bounds;
  let nll = 0;
  for (let t = 1; t <= ITERS; t++) {
    nll = objective();
    const lr = t <= 500 ? 0.05 : (t <= 1000 ? 0.01 : 0.002);
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    for (let i = 0; i < P; i++) {
      mAdam[i] = b1 * mAdam[i] + (1 - b1) * grad[i];
      vAdam[i] = b2 * vAdam[i] + (1 - b2) * grad[i] * grad[i];
      theta[i] -= lr * (mAdam[i] / c1) / (Math.sqrt(vAdam[i] / c2) + eps);
    }
    if (theta[2 * n + 1] < rhoLo) theta[2 * n + 1] = rhoLo;
    if (theta[2 * n + 1] > rhoHi) theta[2 * n + 1] = rhoHi;
    if (onProgress && t % 100 === 0) onProgress(t, ITERS, nll);
  }

  // Weighted data mass per team (thin-data trust in predict).
  const wm = new Float64Array(n);
  for (let k = 0; k < m; k++) { wm[hi[k]] += w[k]; wm[ai[k]] += w[k]; }

  const teamParams = {};
  for (let i = 0; i < n; i++) {
    teamParams[teams[i]] = {
      a: Math.exp(theta[i]),
      d: Math.exp(theta[n + i]),
      w: wm[i],
    };
  }
  return {
    teams: teamParams,
    avg_goals: avgGoals,
    home_adv: Math.exp(theta[2 * n]),
    rho: theta[2 * n + 1],
    goal_scale: REFIT_CFG.goal_scale,
    nll,
  };
}

/* ---------------------------------------------------------------- *
 * elo.compute_ratings + fit_draw_params
 * ---------------------------------------------------------------- */
function marginMultiplier(goalDiff) {
  const g = Math.abs(goalDiff);
  if (g <= 1) return 1.0;
  if (g === 2) return 1.5;
  return (11.0 + g) / 8.0;
}

function computeEloRatings(rows) {
  const ratings = {};
  const K = REFIT_CFG.elo_k, START = REFIT_CFG.elo_start, ADV = REFIT_CFG.elo_home_adv;
  for (const r of rows) {
    const rh = ratings[r.home] ?? START;
    const ra = ratings[r.away] ?? START;
    const adv = r.neutral ? 0.0 : ADV;
    const expHome = 1.0 / (1.0 + Math.pow(10.0, (ra - (rh + adv)) / 400.0));
    const scoreHome = r.hg > r.ag ? 1.0 : (r.hg === r.ag ? 0.5 : 0.0);
    const delta = K * marginMultiplier(r.hg - r.ag) * (scoreHome - expHome);
    ratings[r.home] = rh + delta;
    ratings[r.away] = ra - delta;
  }
  return ratings;
}

function fitDrawParams(rows, ratings) {
  const cutoff = Date.now() - 365 * 4 * 86400000;
  const recent = [];
  for (const r of rows) {
    if (r.time < cutoff) continue;
    recent.push([
      ratings[r.home] ?? REFIT_CFG.elo_start,
      ratings[r.away] ?? REFIT_CFG.elo_start,
      r.neutral,
      r.hg > r.ag ? 0 : (r.hg === r.ag ? 1 : 2),
    ]);
  }
  const linspace = (a, b, k) =>
    Array.from({ length: k }, (_, i) => a + (b - a) * i / (k - 1));

  let best = [0.28, 220.0], bestLL = Infinity;
  for (const base of linspace(0.15, 0.38, 12)) {
    for (const decay of linspace(120.0, 400.0, 12)) {
      let ll = 0;
      for (const [rh, ra, neutral, outcome] of recent) {
        const adv = neutral ? 0.0 : REFIT_CFG.elo_home_adv;
        const dr = (rh + adv) - ra;
        const eHome = 1.0 / (1.0 + Math.pow(10.0, -dr / 400.0));
        const pDraw = base * Math.exp(-Math.abs(dr) / decay);
        const p = [Math.max(eHome - 0.5 * pDraw, 1e-6),
                   Math.max(pDraw, 1e-6),
                   Math.max(1.0 - eHome - 0.5 * pDraw, 1e-6)];
        const s = p[0] + p[1] + p[2];
        ll -= Math.log(Math.max(p[outcome] / s, 1e-12));
      }
      if (ll < bestLL) { bestLL = ll; best = [base, decay]; }
    }
  }
  return best;
}

/* ---------------------------------------------------------------- *
 * Public entry point: main.build_current_model(force=True), in JS
 * ---------------------------------------------------------------- */
async function refitModel(onStatus) {
  const say = onStatus || (() => {});
  say("Downloading latest results (~4 MB)…");
  const resp = await fetch(RESULTS_URL);
  if (!resp.ok) throw new Error(`download failed (HTTP ${resp.status})`);
  const csv = await resp.text();

  say("Preparing match data…");
  await nextFrame();
  const rows = cleanMatches(csv);
  addWeights(rows);

  say(`Fitting Dixon-Coles model on ${rows.length.toLocaleString()} matches…`);
  await nextFrame();
  const bundle = fitDixonColes(rows, null);

  say("Computing Elo ratings & draw model…");
  await nextFrame();
  bundle.elo = computeEloRatings(rows);
  bundle.draw_params = fitDrawParams(rows, bundle.elo);
  bundle.fitted_at = new Date().toISOString();
  return bundle;
}

function nextFrame() {
  return new Promise(res => setTimeout(res, 15)); // let the status text paint
}
