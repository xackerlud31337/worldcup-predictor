/*
 * nba/predictor.js — in-browser prediction on the fitted NBA bundle
 * =================================================================
 * One coherent normal model: the point margin is N(slope * elo_diff, sigma),
 * so the win probability, fair spread and victory-margin buckets all come
 * from the same curve and can never disagree. The game total comes from the
 * teams' exponentially-weighted scoring rates (points scored / allowed per
 * game), and the predicted score reconciles the two: total from the rates,
 * margin from the Elo gap.
 */

const NBA_CFG = NBA_DATA.config;

function nbaTeam(name) {
  return NBA_DATA.teams[name];
}

function listTeams(conf) {
  return Object.keys(NBA_DATA.teams)
    .filter((n) => !conf || NBA_DATA.teams[n].conf === conf)
    .sort();
}

function normCdf(x) {
  // Abramowitz-Stegun erf approximation, plenty for display purposes
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-(x * x) / 2);
  return x >= 0 ? 0.5 * (1 + erf) : 0.5 * (1 - erf);
}

/*
 * venue: "home" (team A at home), "away" (team B at home), "neutral".
 */
function predictGame(nameA, nameB, venue = "home") {
  const A = nbaTeam(nameA), B = nbaTeam(nameB);
  const adv = venue === "home" ? NBA_CFG.hfa : venue === "away" ? -NBA_CFG.hfa : 0;
  const diff = A.elo + adv - B.elo;

  const mu = NBA_CFG.slope * diff;            // expected margin, A minus B
  const sigma = NBA_CFG.sigma;
  const pA = normCdf(mu / sigma);
  const pB = 1 - pA;

  // game total from the scoring rates: A's offence meets B's defence & v.v.
  const total = (A.off + B.def + B.off + A.def) / 2;
  const scoreA = (total + mu) / 2;
  const scoreB = (total - mu) / 2;

  // victory-margin buckets from the same normal (margin 0 can't happen)
  const seg = (lo, hi) => normCdf((hi - mu) / sigma) - normCdf((lo - mu) / sigma);
  const margins = [
    [`${nameA} by 16+`, 1 - normCdf((15.5 - mu) / sigma)],
    [`${nameA} by 11–15`, seg(10.5, 15.5)],
    [`${nameA} by 6–10`, seg(5.5, 10.5)],
    [`${nameA} by 1–5`, seg(0, 5.5)],
    [`${nameB} by 1–5`, seg(-5.5, 0)],
    [`${nameB} by 6–10`, seg(-10.5, -5.5)],
    [`${nameB} by 11–15`, seg(-15.5, -10.5)],
    [`${nameB} by 16+`, normCdf((-15.5 - mu) / sigma)],
  ];

  // fair betting lines: spread and total rounded to the usual half point
  const spread = Math.round(mu - 0.5) + 0.5;   // nearest half-point line
  const line = Math.round(total * 2) / 2 + (Math.round(total * 2) % 2 === 0 ? 0.5 : 0);
  const sT = NBA_CFG.sigma_total;
  const pOver = 1 - normCdf((line - total) / sT);
  const sgn = (x) => (x > 0 ? "-" : "+") + Math.abs(x).toFixed(1);

  const markets = [
    { label: `${nameA} to win (moneyline)`, p: pA },
    { label: `${nameB} to win (moneyline)`, p: pB },
    { label: `${nameA} ${sgn(spread)} (spread)`, p: 1 - normCdf((spread - mu) / sigma) },
    { label: `${nameB} ${sgn(-spread)} (spread)`, p: normCdf((spread - mu) / sigma) },
    { label: `${nameA} wins by 10+`, p: 1 - normCdf((9.5 - mu) / sigma) },
    { label: `${nameB} wins by 10+`, p: normCdf((-9.5 - mu) / sigma) },
    { label: `Over ${line.toFixed(1)} points`, p: pOver },
    { label: `Under ${line.toFixed(1)} points`, p: 1 - pOver },
    { label: `Over ${(line + 8).toFixed(1)} points`, p: 1 - normCdf((line + 8 - total) / sT) },
    { label: `Under ${(line - 8).toFixed(1)} points`, p: normCdf((line - 8 - total) / sT) },
  ].sort((x, y) => y.p - x.p);

  return {
    a: A, b: B, pA, pB,
    mu, sigma, total,
    scoreA, scoreB,
    spread, line,
    margins, markets, venue,
  };
}
