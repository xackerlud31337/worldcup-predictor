// All predictions run locally in the browser (predictor.js + club_model_data.js) —
// no backend needed, so the site can be hosted as static files on GitHub Pages.
// Club crests and the upcoming-fixtures box come from ESPN's CORS-open API,
// the same source clubs_build.py fits the model from.

function crestUrl(team){
  const m = CLUB_META.teams[team];
  return (m && m.logo) ? m.logo : null;
}
function crestImg(team, cls="flag"){
  const url = crestUrl(team);
  if (url) return `<img class="${cls}" src="${url}" alt="${team}" loading="lazy">`;
  const initials = team.split(/\s+/).map(w => w[0]).join("").slice(0, 3).toUpperCase();
  return `<span class="${cls} placeholder">${initials}</span>`;
}
function leagueOf(team){
  const m = CLUB_META.teams[team];
  return m ? m.league : "";
}

let teams = [], home = null, away = null;

function loadTeams(){
  try {
    teams = listTeams();
    // Premier League first (it's in the page title), then the rest A-Z by league.
    const order = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"];
    teams.sort((a, b) => {
      const la = order.indexOf(leagueOf(a)), lb = order.indexOf(leagueOf(b));
      const ra = la === -1 ? 99 : la, rb = lb === -1 ? 99 : lb;
      if (ra !== rb) return ra - rb;
      const lg = leagueOf(a).localeCompare(leagueOf(b));
      if (ra === 99 && lg !== 0) return lg;
      return a.localeCompare(b);
    });
    buildList("home"); buildList("away");
  } catch(e){ showStatus("Could not load the model data (club_model_data.js missing?)", true); }
}

function pickerEls(which){
  const root = document.getElementById(`${which}-picker`);
  return {
    root,
    toggle: root.querySelector(".picker-toggle"),
    menu: root.querySelector(".picker-menu"),
    search: root.querySelector(".picker-search"),
    list: root.querySelector(".picker-list"),
    flag: root.querySelector(".picker-toggle .flag"),
    name: root.querySelector(".picker-name"),
  };
}

function buildList(which, filter=""){
  const els = pickerEls(which);
  els.list.innerHTML = "";
  const f = filter.trim().toLowerCase();
  teams.filter(t => t.toLowerCase().includes(f) ||
                    leagueOf(t).toLowerCase().includes(f)).forEach(t => {
    const li = document.createElement("li");
    li.className = "picker-item";
    li.setAttribute("role","option");
    li.innerHTML = `${crestImg(t)} <span>${t}</span>
      <span class="picker-league">${leagueOf(t)}</span>`;
    li.onclick = () => selectTeam(which, t);
    els.list.appendChild(li);
  });
}

function selectTeam(which, team){
  if (which === "home") home = team; else away = team;
  unavailable[which].clear();
  const els = pickerEls(which);
  els.flag.outerHTML = crestImg(team, "flag");
  els.name.textContent = team;
  els.name.classList.remove("muted");
  els.menu.hidden = true;
  document.getElementById("calc-btn").disabled = !(home && away);
  renderSquadsPanel();
}

function wirePicker(which){
  const els = pickerEls(which);
  els.toggle.onclick = () => {
    els.menu.hidden = !els.menu.hidden;
    if (!els.menu.hidden){ els.search.value=""; buildList(which); els.search.focus(); }
  };
  els.search.oninput = () => buildList(which, els.search.value);
}

document.addEventListener("click", (e) => {
  ["home","away"].forEach(w => {
    const els = pickerEls(w);
    if (!els.root.contains(e.target)) els.menu.hidden = true;
  });
});

document.getElementById("swap-btn").onclick = () => {
  const h = home, a = away; home = null; away = null;
  if (a) selectTeam("home", a);
  if (h) selectTeam("away", h);
};

function recalc(scroll = true){
  if (!home || !away) return;
  if (home === away){ showStatus("Pick two different clubs.", true); return; }

  const body = {
    home_team: home,
    away_team: away,
    method: document.getElementById("method").value,
    home_advantage: document.getElementById("home-adv").checked,
    adjust: currentAdjust(),
  };

  showStatus("Calculating…");
  document.getElementById("calc-btn").disabled = true;

  // Let the status paint before the (synchronous) simulation runs.
  setTimeout(() => {
    try {
      const d = runPrediction(body);
      renderResults(d, scroll);
      hideStatus();
    } catch(e){ showStatus("Prediction failed: " + e.message, true); }
    finally { document.getElementById("calc-btn").disabled = false; }
  }, 20);
}

document.getElementById("calc-btn").onclick = () => recalc(true);

function pct(x){ return (x*100).toFixed(1) + "%"; }

/* ---------- squads & availability (players_data.js) ---------- */
// Toggled-out players, per side. Reset whenever that side's club changes.
const unavailable = { home: new Set(), away: new Set() };

function squadOf(team){
  const d = (typeof PLAYER_DATA !== "undefined") && PLAYER_DATA.teams[team];
  return d || null;
}

// Availability multipliers for one side: attack scale (<=1) from lost
// npxG+xA share, defensive leak (>=1) from missing GK/D starters.
function sideAdjust(team, outSet){
  const d = squadOf(team);
  if (!d || !outSet.size) return { atk: 1, leak: 1 };
  const cfg = PLAYER_DATA.config;
  let lost = 0, leak = 1;
  d.players.forEach((p) => {
    if (!outSet.has(p.n)) return;
    const repl = cfg.repl_factor * (cfg.prior[p.pos] || 0);
    lost += Math.max(0, p.r - repl) * p.s;
    const dl = cfg.def_leak[p.pos];
    if (dl) leak *= 1 + dl * p.s;
  });
  return {
    atk: Math.max(0.5, (d.s0 - lost) / d.s0),
    leak: Math.min(1.5, leak),
  };
}

function currentAdjust(){
  const h = sideAdjust(home, unavailable.home);
  const a = sideAdjust(away, unavailable.away);
  return { ha: h.atk, hd: h.leak, aa: a.atk, ad: a.leak };
}

function roleLabel(share){
  return share >= 0.6 ? "starter" : (share >= 0.3 ? "rotation" : "squad");
}

function renderSquad(which){
  const team = which === "home" ? home : away;
  const box = document.querySelector(`#squad-${which}`);
  const nameEl = box.querySelector(".squad-name");
  const listEl = box.querySelector(".squad-list");
  if (!team){ nameEl.textContent = ""; listEl.innerHTML = ""; updateImpact(which); return; }

  nameEl.innerHTML = `${crestImg(team)} ${team}`;
  const d = squadOf(team);
  if (!d){
    listEl.innerHTML = `<p class="squad-none">No player data for ${team} — Understat covers the big-five leagues only. The team rating still applies.</p>`;
    updateImpact(which);
    return;
  }
  const outSet = unavailable[which];
  listEl.innerHTML = "";
  d.players.forEach((p) => {
    const row = document.createElement("label");
    row.className = "squad-row" + (outSet.has(p.n) ? " out" : "");
    row.innerHTML = `<input type="checkbox" ${outSet.has(p.n) ? "" : "checked"}>
      <span class="squad-pos ${p.pos}">${p.pos}</span>
      <span class="squad-player">${p.n}</span>
      <span class="squad-role">${roleLabel(p.s)}</span>`;
    row.querySelector("input").onchange = (e) => {
      if (e.target.checked) outSet.delete(p.n); else outSet.add(p.n);
      row.classList.toggle("out", !e.target.checked);
      updateImpact(which);
      if (!document.getElementById("results").hidden) recalc(false);
    };
    listEl.appendChild(row);
  });
  updateImpact(which);
}

function updateImpact(which){
  const team = which === "home" ? home : away;
  const el = document.querySelector(`#squad-${which} .squad-impact`);
  if (!team){ el.hidden = true; return; }
  const adj = sideAdjust(team, unavailable[which]);
  if (adj.atk === 1 && adj.leak === 1){ el.hidden = true; return; }
  const bits = [];
  if (adj.atk < 1) bits.push(`attack −${((1 - adj.atk) * 100).toFixed(0)}%`);
  if (adj.leak > 1) bits.push(`concedes +${((adj.leak - 1) * 100).toFixed(0)}%`);
  el.hidden = false;
  el.textContent = `${team} without those players: ${bits.join(" · ")}`;
}

function renderSquadsPanel(){
  document.getElementById("squads").hidden = !(home || away);
  renderSquad("home");
  renderSquad("away");
}


function renderResults(d, scroll = true){
  const pHome = d.probabilities.home_win;
  const pDraw = d.probabilities.draw;
  const pAway = d.probabilities.away_win;
  const xgHome = d.expected_goals.home;
  const xgAway = d.expected_goals.away;
  const scores = d.simulation.top_scores;
  const trust = d.data_trust;

  document.getElementById("result-title").innerHTML =
    `${crestImg(home,"flag")} ${home} <span class="vsr">vs</span> ${away} ${crestImg(away,"flag")}`;
  const outCount = unavailable.home.size + unavailable.away.size;
  document.getElementById("venue-note").textContent =
    (document.getElementById("home-adv").checked ? `${home} playing at home` : "Neutral venue")
    + (outCount ? ` · ${outCount} player${outCount > 1 ? "s" : ""} marked unavailable` : "");

  setSeg("home", pHome); setSeg("draw", pDraw); setSeg("away", pAway);
  document.getElementById("leg-home-name").textContent = home;
  document.getElementById("leg-away-name").textContent = away;
  document.getElementById("leg-home-pct").textContent = pct(pHome);
  document.getElementById("leg-draw-pct").textContent = pct(pDraw);
  document.getElementById("leg-away-pct").textContent = pct(pAway);

  document.getElementById("xg-home").textContent = xgHome.toFixed(2);
  document.getElementById("xg-away").textContent = xgAway.toFixed(2);
  document.getElementById("xg-home-flag").src = crestUrl(home) || "";
  document.getElementById("xg-away-flag").src = crestUrl(away) || "";

  const chart = document.getElementById("score-chart");
  chart.innerHTML = "";
  const entries = Object.entries(scores);
  const max = Math.max(...entries.map(([,v]) => v), 0.0001);
  entries.forEach(([score, p]) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `<span class="score-label">${score}</span>
      <span class="score-track"><span class="score-fill" style="width:${(p/max*100).toFixed(1)}%"></span></span>
      <span class="score-pct">${(p*100).toFixed(1)}%</span>`;
    chart.appendChild(row);
  });

  const mg = d.margins;
  const marginRows = [
    [`${home} by 3+`, mg.home3, "home"],
    [`${home} by 2`, mg.home2, "home"],
    [`${home} by 1`, mg.home1, "home"],
    ["Draw", mg.draw, "draw"],
    [`${away} by 1`, mg.away1, "away"],
    [`${away} by 2`, mg.away2, "away"],
    [`${away} by 3+`, mg.away3, "away"],
  ];
  const mc = document.getElementById("margin-chart");
  mc.innerHTML = "";
  const mmax = Math.max(...marginRows.map((r) => r[1]), 0.0001);
  marginRows.forEach(([label, p, side]) => {
    const row = document.createElement("div");
    row.className = "margin-row";
    row.innerHTML = `<span class="margin-label">${label}</span>
      <span class="margin-track"><span class="margin-fill ${side}" style="width:${(p/mmax*100).toFixed(1)}%"></span></span>
      <span class="margin-pct">${(p*100).toFixed(1)}%</span>`;
    mc.appendChild(row);
  });

  const ml = document.getElementById("market-list");
  ml.innerHTML = "";
  d.markets.forEach(({ label, p }) => {
    const fair = p > 0.005 ? (1 / p).toFixed(2) : "99+";
    const row = document.createElement("div");
    row.className = "market-row";
    row.innerHTML = `<span class="market-label">${label}</span>
      <span class="market-pct">${(p * 100).toFixed(1)}%</span>
      <span class="market-odds">${fair}</span>`;
    ml.appendChild(row);
  });

  const tn = document.getElementById("trust-note");
  if (trust < 1){ tn.hidden = false; tn.textContent = "Note: limited recent top-level data for one of these clubs, so this estimate leans on its Elo rating and is less certain."; }
  else tn.hidden = true;

  document.getElementById("results").hidden = false;
  if (scroll) document.getElementById("results").scrollIntoView({behavior:"smooth", block:"start"});
}

function setSeg(which, p){
  const seg = document.getElementById(`seg-${which}`);
  seg.style.width = (p*100).toFixed(1) + "%";
  seg.querySelector("span").textContent = p >= 0.08 ? pct(p) : "";
}

function showStatus(msg, isErr=false){
  const s = document.getElementById("status");
  s.hidden = false; s.textContent = msg;
  s.classList.toggle("error", isErr);
}
function hideStatus(){ document.getElementById("status").hidden = true; }

/* ---------- info popup ---------- */
const infoOverlay = document.getElementById("info-overlay");
document.getElementById("info-btn").onclick = () => { infoOverlay.hidden = false; };
document.getElementById("info-close").onclick = () => { infoOverlay.hidden = true; };
infoOverlay.addEventListener("click", (e) => {
  if (e.target === infoOverlay) infoOverlay.hidden = true;
});
/* ---------- donate popup ---------- */
const donateOverlay = document.getElementById("donate-overlay");
document.getElementById("donate-btn").onclick = () => { donateOverlay.hidden = false; };
document.getElementById("donate-close").onclick = () => { donateOverlay.hidden = true; };
donateOverlay.addEventListener("click", (e) => {
  if (e.target === donateOverlay) donateOverlay.hidden = true;
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { infoOverlay.hidden = true; donateOverlay.hidden = true; }
});

document.querySelectorAll(".copy-btn[data-copy]").forEach((btn) => {
  btn.onclick = async () => {
    try {
      await navigator.clipboard.writeText(btn.dataset.copy);
    } catch (e) {
      const ta = document.createElement("textarea");
      ta.value = btn.dataset.copy;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    const old = btn.textContent;
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = old; btn.classList.remove("copied"); }, 1500);
  };
});

/* ---------- upcoming fixtures (ESPN, EPL + Champions League) ---------- */
// The model's team names ARE ESPN display names (clubs_build.py canonicalises
// to them), so most fixtures match directly; the fuzzy fallback catches
// accent/punctuation variants.
const FIXTURE_COMPS = [
  { slug: "eng.1", label: "Premier League" },
  { slug: "uefa.champions", label: "Champions League" },
];

const _normCache = {};
function normName(s){
  if (_normCache[s]) return _normCache[s];
  const n = s.normalize("NFD").replace(/[̀-ͯ]/g, "")
             .toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
  return (_normCache[s] = n);
}
let _teamByNorm = null;
function modelTeamName(name){
  if (!name) return null;
  if (CLUB_META.teams[name]) return name;
  if (!_teamByNorm){
    _teamByNorm = {};
    teams.forEach(t => { _teamByNorm[normName(t)] = t; });
  }
  return _teamByNorm[normName(name)] || null;
}

function espnDates(daysAhead){
  const fmt = (d) => `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,"0")}${String(d.getDate()).padStart(2,"0")}`;
  const now = new Date();
  const end = new Date(now.getTime() + daysAhead * 864e5);
  return `${fmt(now)}-${fmt(end)}`;
}

async function loadFixtures(){
  const box = document.getElementById("fixtures-list");
  try {
    const range = espnDates(14);
    const results = await Promise.all(FIXTURE_COMPS.map(async ({ slug, label }) => {
      const resp = await fetch(
        `https://site.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard?dates=${range}&limit=100`);
      if (!resp.ok) return [];
      const data = await resp.json();
      return (data.events || [])
        .filter((e) => !e.status.type.completed)
        .map((e) => ({ ev: e, league: label }));
    }));
    const fixtures = results.flat()
      .sort((a, b) => a.ev.date.localeCompare(b.ev.date))
      .slice(0, 8);
    if (!fixtures.length){
      box.innerHTML = `<p class="fixtures-status">No Premier League or Champions League matches in the next two weeks — off-season. Pick any matchup above.</p>`;
      return;
    }
    box.innerHTML = fixtures.map(fixtureRow).join("");
    box.querySelectorAll(".fx-load").forEach((btn, i) => {
      btn.onclick = () => loadFixtureIntoPickers(fixtures[i]);
    });
  } catch (e) {
    box.innerHTML = `<p class="fixtures-status">Couldn't load upcoming fixtures.</p>`;
  }
}

function fixtureSides(ev){
  const comp = ev.competitions[0];
  const bySide = {};
  comp.competitors.forEach((c) => { bySide[c.homeAway] = c.team.displayName; });
  return { home: bySide.home, away: bySide.away, neutral: !!comp.neutralSite };
}

function loadFixtureIntoPickers({ ev }){
  const s = fixtureSides(ev);
  const h = modelTeamName(s.home);
  const a = modelTeamName(s.away);
  if (!h || !a){
    showStatus(`Sorry, couldn't match "${s.home}" or "${s.away}" to a club in the model.`, true);
    return;
  }
  selectTeam("home", h);
  selectTeam("away", a);
  document.getElementById("home-adv").checked = !s.neutral;
  document.querySelector(".matchup-card").scrollIntoView({ behavior: "smooth", block: "start" });
  document.getElementById("calc-btn").click();
}

function fixtureRow({ ev, league }){
  const s = fixtureSides(ev);
  const t = new Date(ev.date);
  let when = "Time TBD";
  if (!Number.isNaN(t.getTime())) {
    const hhmm = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    when = t.toDateString() === new Date().toDateString()
      ? `Today &middot; ${hhmm}`
      : `${t.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })} &middot; ${hhmm}`;
  }
  const live = ev.status.type.state === "in";
  return `<div class="fixture">
    <div class="fx-league">${league} &middot; ${when}${live ? ' <span class="fx-live">LIVE</span>' : ""}</div>
    <div class="fx-row">
      <span class="fx-team">${s.home || "?"}</span>
      <span class="fx-mid">vs</span>
      <span class="fx-team away">${s.away || "?"}</span>
    </div>
    <button type="button" class="fx-load">&#127942; Predict this match</button>
  </div>`;
}

/* ---------- visit counter (CounterAPI) ---------- */
async function loadVisitCount() {
  try {
    const resp = await fetch("https://api.counterapi.dev/v1/xackerlud-worldcup/clubs-visits/up");
    const data = await resp.json();
    if (data.count) {
      const el = document.getElementById("visit-count");
      el.textContent = `🏆 ${Number(data.count).toLocaleString()} visits`;
      el.hidden = false;
    }
  } catch (e) { /* cosmetic — fail silently */ }
}

wirePicker("home");
wirePicker("away");
loadTeams();
loadFixtures();
loadVisitCount();
