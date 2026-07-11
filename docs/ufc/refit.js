/*
 * ufc/refit.js — ufc_build.py's data pipeline, in the browser
 * ===========================================================
 * Downloads the community UFC dataset (Greco1899/scrape_ufc_stats, the same
 * source the Python build uses) and rebuilds ratings, records, profiles and
 * career stats client-side. The win-probability coefficients (ring rust and
 * age) are NOT refitted here; they come from the backtested Python build and
 * stay as shipped.
 */

const UFC_RAW = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/";

// mirrored from ufc_build.py
const R_ELO_START = 1500.0;
const R_ELO_K = 56.0;
const R_FINISH_MULT = 1.2;
const R_MIN_FIGHTS = 2;
const R_MIN_LAST = new Date("2015-01-01").getTime();

function rParseCsv(text) {
  // line-based CSV with quoted-field support; returns array of arrays
  const rows = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
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
    rows.push(out.map((c) => c.trim()));
  }
  return rows;
}

function rMethodClass(m) {
  m = (m || "").toLowerCase();
  if (m.startsWith("ko/tko") || m.startsWith("tko")) return "ko";
  if (m.startsWith("submission")) return "sub";
  if (m.startsWith("decision")) return "dec";
  return "other";
}

function rCleanDivision(wc) {
  for (const junk of ["UFC", "Title", "Bout", "Interim", "Tournament",
                      "Ultimate Fighter", "Australia", "UK", "vs.",
                      "Latin America", "China", "Brazil"]) {
    wc = wc.split(junk).join("");
  }
  wc = wc.replace(/\s+/g, " ").trim().replace(/^[0-9 ]+/, "");
  return wc || "Unknown";
}

function rParseOf(s) {
  const m = (s || "").split(" of ");
  return m.length === 2 ? [parseInt(m[0], 10) || 0, parseInt(m[1], 10) || 0] : [0, 0];
}

function rParseHeight(s) {
  const m = (s || "").match(/(\d+)'\s*(\d+)?/);
  return m ? parseInt(m[1], 10) * 12 + (parseInt(m[2], 10) || 0) : null;
}

function rParseMmss(s) {
  const m = (s || "").match(/(\d+):(\d+)/);
  return m ? parseInt(m[1], 10) * 60 + parseInt(m[2], 10) : 0;
}

async function rFetch(name, say) {
  say(`Downloading ${name}…`);
  const resp = await fetch(UFC_RAW + name);
  if (!resp.ok) throw new Error(`${name}: HTTP ${resp.status}`);
  return resp.text();
}

async function refitUfc(onStatus) {
  const say = onStatus || (() => {});

  const evTxt = await rFetch("ufc_event_details.csv", say);
  const resTxt = await rFetch("ufc_fight_results.csv", say);
  const tottTxt = await rFetch("ufc_fighter_tott.csv", say);
  say("Downloading fight stats (~7 MB)…");
  const statsText = await rFetch("ufc_fight_stats.csv", () => {});

  say("Rebuilding the model…");
  await new Promise((r) => setTimeout(r, 15)); // let the status paint

  // event -> date
  const evRows = rParseCsv(evTxt);
  const evCol = Object.fromEntries(evRows[0].map((h, i) => [h, i]));
  const evDate = {};
  for (let i = 1; i < evRows.length; i++) {
    const t = new Date(evRows[i][evCol.DATE]).getTime();
    if (!Number.isNaN(t)) evDate[evRows[i][evCol.EVENT]] = t;
  }

  // bouts
  const rRows = rParseCsv(resTxt);
  const rCol = Object.fromEntries(rRows[0].map((h, i) => [h, i]));
  const fights = [];
  for (let i = 1; i < rRows.length; i++) {
    const f = rRows[i];
    const date = evDate[f[rCol.EVENT]];
    const bout = f[rCol.BOUT] || "";
    const outcome = f[rCol.OUTCOME];
    if (date === undefined || !bout.includes(" vs. ")) continue;
    if (!["W/L", "L/W", "D/D", "NC/NC"].includes(outcome)) continue;
    const rnd = parseInt(f[rCol.ROUND], 10);
    if (Number.isNaN(rnd)) continue;
    const [a, b] = bout.split(" vs. ").map((x) => x.trim());
    fights.push({
      date, a, b, outcome,
      division: rCleanDivision(f[rCol.WEIGHTCLASS] || ""),
      method: rMethodClass(f[rCol.METHOD]),
      round: rnd,
      secs: (rnd - 1) * 300 + rParseMmss(f[rCol.TIME]),
      event: f[rCol.EVENT], bout,
    });
  }
  fights.sort((x, y) => x.date - y.date);

  // Elo pass (same constants as the Python build)
  const elo = {};
  for (const f of fights) {
    if (f.outcome === "NC/NC") continue;
    const ra = elo[f.a] ?? R_ELO_START, rb = elo[f.b] ?? R_ELO_START;
    const expA = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    const scoreA = { "W/L": 1.0, "L/W": 0.0, "D/D": 0.5 }[f.outcome];
    const k = R_ELO_K * (f.method === "ko" || f.method === "sub" ? R_FINISH_MULT : 1.0);
    const d = k * (scoreA - expA);
    elo[f.a] = ra + d;
    elo[f.b] = rb - d;
  }

  // per-fight stat totals
  const sRows = rParseCsv(statsText);
  const sCol = Object.fromEntries(sRows[0].map((h, i) => [h, i]));
  const totals = {};
  for (let i = 1; i < sRows.length; i++) {
    const r = sRows[i];
    const key = r[sCol.EVENT] + "|" + r[sCol.BOUT] + "|" + r[sCol.FIGHTER];
    const t = totals[key] || (totals[key] = [0, 0, 0, 0, 0, 0]);
    const [sl, sa] = rParseOf(r[sCol["SIG.STR."]]);
    const [tl, ta] = rParseOf(r[sCol.TD]);
    t[0] += sl; t[1] += sa; t[2] += tl; t[3] += ta;
    t[4] += parseInt(r[sCol["SUB.ATT"]], 10) || 0;
    t[5] += parseInt(r[sCol.KD], 10) || 0;
  }

  // career aggregates
  const M = { ko: 0, sub: 1, dec: 2 };
  const F = {};
  const get = (n) => F[n] || (F[n] = {
    w: 0, l: 0, d: 0, wm: [0, 0, 0], lm: [0, 0, 0], rh: [0, 0, 0, 0, 0],
    secs: 0, sigL: 0, sigA: 0, absL: 0, tdL: 0, tdA: 0, sub: 0, kd: 0,
    div: "Unknown", last: 0, n: 0,
  });
  for (const f of fights) {
    if (f.outcome === "NC/NC") continue;
    for (const [me, opp, res] of [[f.a, f.b, f.outcome],
                                  [f.b, f.a, f.outcome.split("/").reverse().join("/")]]) {
      const st = get(me);
      st.n++; st.secs += f.secs; st.div = f.division; st.last = f.date;
      const mine = totals[f.event + "|" + f.bout + "|" + me];
      const theirs = totals[f.event + "|" + f.bout + "|" + opp];
      if (mine) { st.sigL += mine[0]; st.sigA += mine[1]; st.tdL += mine[2]; st.tdA += mine[3]; st.sub += mine[4]; st.kd += mine[5]; }
      if (theirs) st.absL += theirs[0];
      if (res === "D/D") st.d++;
      else if (res === "W/L") {
        st.w++;
        if (f.method in M) st.wm[M[f.method]]++;
        if (f.method === "ko" || f.method === "sub") st.rh[Math.min(f.round, 5) - 1]++;
      } else {
        st.l++;
        if (f.method in M) st.lm[M[f.method]]++;
      }
    }
  }

  // tale of the tape
  const tRows = rParseCsv(tottTxt);
  const tCol = Object.fromEntries(tRows[0].map((h, i) => [h, i]));
  const tott = {};
  for (let i = 1; i < tRows.length; i++) {
    const r = tRows[i];
    const dob = new Date(r[tCol.DOB]);
    tott[r[tCol.FIGHTER]] = {
      ht: rParseHeight(r[tCol.HEIGHT]),
      rc: parseFloat((r[tCol.REACH] || "").replace('"', "")) || null,
      st: r[tCol.STANCE] || null,
      dob: Number.isNaN(dob.getTime()) ? null : dob.toISOString().slice(0, 10),
    };
  }

  // roster filter + bundle (same shape as ufc_model_data.js)
  const fighters = {};
  const divWm = {};
  for (const [name, st] of Object.entries(F)) {
    if (st.n < R_MIN_FIGHTS || st.last < R_MIN_LAST) continue;
    const mins = st.secs / 60 || 1;
    const t = tott[name] || {};
    fighters[name] = {
      div: st.div,
      elo: Math.round((elo[name] ?? R_ELO_START) * 10) / 10,
      rec: [st.w, st.l, st.d],
      wm: st.wm, lm: st.lm, rh: st.rh,
      slpm: st.sigL / mins,
      sacc: st.sigA ? st.sigL / st.sigA : 0,
      sapm: st.absL / mins,
      td15: st.tdL / (mins / 15),
      tdacc: st.tdA ? st.tdL / st.tdA : 0,
      sub15: st.sub / (mins / 15),
      kd15: st.kd / (mins / 15),
      mins,
      ht: t.ht ?? null, rc: t.rc ?? null, st: t.st ?? null, dob: t.dob ?? null,
      last: new Date(st.last).toISOString().slice(0, 10),
    };
    const dw = divWm[st.div] || (divWm[st.div] = [0, 0, 0]);
    for (let i = 0; i < 3; i++) dw[i] += st.wm[i];
  }

  const divisions = [...new Set(Object.values(fighters).map((f) => f.div))]
    .filter((d) => d !== "Unknown").sort();
  const globalWm = [0, 1, 2].map((i) =>
    Object.values(divWm).reduce((s, v) => s + v[i], 0));

  return { fighters, divisions, global_wm: globalWm, bouts: fights.length };
}
