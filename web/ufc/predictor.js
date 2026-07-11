/*
 * ufc/predictor.js — in-browser prediction on the fitted UFC bundle
 * =================================================================
 * Win probability comes from the calibrated Elo logistic. Method and round
 * distributions blend three signals with pseudo-count smoothing:
 *   how the winner tends to win  +  how the loser tends to lose  +  the
 *   league-wide base rates — so a 3-fight rookie doesn't show a 100% KO rate.
 */

const UFC_CFG = UFC_DATA.config;

function ufcFighter(name) {
  return UFC_DATA.fighters[name];
}

function listFighters(division) {
  return Object.keys(UFC_DATA.fighters)
    .filter((n) => !division || UFC_DATA.fighters[n].div === division)
    .sort();
}

function logLayoffYears(last) {
  const days = Math.max((Date.now() - new Date(last).getTime()) / 86400000, 0);
  return Math.log1p(days / 365);
}

// Win probability: rating gap adjusted for ring rust and age. The
// coefficients were fitted on historical bouts and verified on a held-out
// backtest (see ufc_build.py); without them we fall back to pure Elo.
function winProb(a, b) {
  const f = UFC_CFG.form;
  if (!f) return 1 / (1 + Math.pow(10, -(a.elo - b.elo) / UFC_CFG.elo_scale));
  const ageA = agePct(a.dob), ageB = agePct(b.dob);
  const z = f.w_elo * ((a.elo - b.elo) / 100)
          + f.w_layoff * (logLayoffYears(a.last) - logLayoffYears(b.last))
          + f.w_age * ((ageA && ageB ? ageA - ageB : 0) / 10);
  return 1 / (1 + Math.exp(-z));
}

// Method distribution [ko, sub, dec] given `winner` beats `loser`.
function methodDist(winner, loser) {
  const g = UFC_DATA.global_wm;
  const gSum = g[0] + g[1] + g[2];
  const out = [0, 0, 0];
  let sum = 0;
  for (let i = 0; i < 3; i++) {
    // winner's finishing tendency + how the loser tends to get beaten + prior
    out[i] = winner.wm[i] + 0.5 * loser.lm[i] + 4 * (g[i] / gSum);
    sum += out[i];
  }
  return out.map((x) => x / sum);
}

// Round distribution for a finish, truncated to the scheduled rounds.
const ROUND_PRIOR = [0.42, 0.27, 0.17, 0.09, 0.05]; // league-wide finish rounds
function roundDist(winner, rounds) {
  const raw = [];
  let sum = 0;
  for (let r = 0; r < rounds; r++) {
    raw[r] = (winner.rh[r] || 0) + 5 * ROUND_PRIOR[r];
    sum += raw[r];
  }
  return raw.map((x) => x / sum);
}

function agePct(dob) {
  if (!dob) return null;
  return (Date.now() - new Date(dob).getTime()) / (365.25 * 86400000);
}

/*
 * The full prediction object the UI renders.
 * rounds: 3 (regular) or 5 (main event / title).
 */
function predictFight(nameA, nameB, rounds = 3) {
  const A = ufcFighter(nameA), B = ufcFighter(nameB);
  const pA = winProb(A, B), pB = 1 - pA;

  const mA = methodDist(A, B);            // [ko, sub, dec] if A wins
  const mB = methodDist(B, A);
  const rA = roundDist(A, rounds);        // finish round | A finishes
  const rB = roundDist(B, rounds);

  // P(fight is finished) = P(A wins & finishes) + P(B wins & finishes)
  const finA = pA * (mA[0] + mA[1]);
  const finB = pB * (mB[0] + mB[1]);
  const pFinish = finA + finB;

  // finish-round mixture over both fighters
  const roundMix = [];
  for (let r = 0; r < rounds; r++) {
    roundMix[r] = (finA * rA[r] + finB * rB[r]) / (pFinish || 1);
  }

  // markets, sorted by probability in the UI
  const markets = [
    { label: `${nameA} to win`, p: pA },
    { label: `${nameB} to win`, p: pB },
    { label: `${nameA} by KO/TKO`, p: pA * mA[0] },
    { label: `${nameA} by submission`, p: pA * mA[1] },
    { label: `${nameA} by decision`, p: pA * mA[2] },
    { label: `${nameB} by KO/TKO`, p: pB * mB[0] },
    { label: `${nameB} by submission`, p: pB * mB[1] },
    { label: `${nameB} by decision`, p: pB * mB[2] },
    { label: "Fight goes the distance", p: 1 - pFinish },
    { label: "Fight doesn't go the distance", p: pFinish },
  ];
  // over/under 1.5 and 2.5 rounds (finish before the midpoint of R2 / R3,
  // approximated as R1 finishes + half of R2 finishes, etc.)
  const u15 = pFinish * (roundMix[0] + 0.5 * (roundMix[1] || 0));
  const u25 = pFinish * (roundMix[0] + (roundMix[1] || 0) + 0.5 * (roundMix[2] || 0));
  markets.push({ label: "Under 1.5 rounds", p: u15 });
  markets.push({ label: "Over 1.5 rounds", p: 1 - u15 });
  markets.push({ label: "Under 2.5 rounds", p: u25 });
  markets.push({ label: "Over 2.5 rounds", p: 1 - u25 });
  markets.sort((x, y) => y.p - x.p);

  return {
    a: A, b: B, pA, pB,
    methodA: mA, methodB: mB,
    pFinish, roundMix, rounds,
    markets,
  };
}
