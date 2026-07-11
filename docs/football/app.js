// All predictions run locally in the browser (predictor.js + model_data.js) —
// no backend needed, so the site can be hosted as static files on GitHub Pages.

const ISO = {
  "Afghanistan":"af","Albania":"al","Algeria":"dz","Andorra":"ad","Angola":"ao",
  "Anguilla":"ai","Antigua and Barbuda":"ag","Argentina":"ar","Armenia":"am","Aruba":"aw",
  "Australia":"au","Austria":"at","Azerbaijan":"az","Bahamas":"bs","Bahrain":"bh",
  "Bangladesh":"bd","Barbados":"bb","Belarus":"by","Belgium":"be","Belize":"bz",
  "Benin":"bj","Bermuda":"bm","Bhutan":"bt","Bolivia":"bo","Bosnia and Herzegovina":"ba",
  "Botswana":"bw","Brazil":"br","Brunei":"bn","Bulgaria":"bg","Burkina Faso":"bf",
  "Burundi":"bi","Cambodia":"kh","Cameroon":"cm","Canada":"ca","Cape Verde":"cv",
  "Central African Republic":"cf","Chad":"td","Chile":"cl","China":"cn","Chinese Taipei":"tw",
  "Colombia":"co","Comoros":"km","Congo":"cg","Cook Islands":"ck","Costa Rica":"cr",
  "Croatia":"hr","Cuba":"cu","Curaçao":"cw","Cyprus":"cy","Czechia":"cz",
  "Denmark":"dk","Djibouti":"dj","Dominica":"dm","Dominican Republic":"do","DR Congo":"cd",
  "Ecuador":"ec","Egypt":"eg","El Salvador":"sv","England":"gb-eng","Equatorial Guinea":"gq",
  "Eritrea":"er","Estonia":"ee","Eswatini":"sz","Ethiopia":"et","Faroe Islands":"fo",
  "Fiji":"fj","Finland":"fi","France":"fr","Gabon":"ga","Gambia":"gm",
  "Georgia":"ge","Germany":"de","Ghana":"gh","Gibraltar":"gi","Greece":"gr",
  "Grenada":"gd","Guam":"gu","Guatemala":"gt","Guinea":"gn","Guinea-Bissau":"gw",
  "Guyana":"gy","Haiti":"ht","Honduras":"hn","Hong Kong":"hk","Hungary":"hu",
  "Iceland":"is","India":"in","Indonesia":"id","Iran":"ir","Iraq":"iq",
  "Ireland":"ie","Israel":"il","Italy":"it","Ivory Coast":"ci","Jamaica":"jm",
  "Japan":"jp","Jordan":"jo","Kazakhstan":"kz","Kenya":"ke","Kosovo":"xk",
  "Kuwait":"kw","Kyrgyzstan":"kg","Laos":"la","Latvia":"lv","Lebanon":"lb",
  "Lesotho":"ls","Liberia":"lr","Libya":"ly","Liechtenstein":"li","Lithuania":"lt",
  "Luxembourg":"lu","Macau":"mo","Madagascar":"mg","Malawi":"mw","Malaysia":"my",
  "Maldives":"mv","Mali":"ml","Malta":"mt","Mauritania":"mr","Mauritius":"mu",
  "Mexico":"mx","Moldova":"md","Mongolia":"mn","Montenegro":"me","Montserrat":"ms",
  "Morocco":"ma","Mozambique":"mz","Myanmar":"mm","Namibia":"na","Nepal":"np",
  "Netherlands":"nl","New Caledonia":"nc","New Zealand":"nz","Nicaragua":"ni","Niger":"ne",
  "Nigeria":"ng","North Korea":"kp","North Macedonia":"mk","Northern Ireland":"gb-nir","Norway":"no",
  "Oman":"om","Pakistan":"pk","Palestine":"ps","Panama":"pa","Papua New Guinea":"pg",
  "Paraguay":"py","Peru":"pe","Philippines":"ph","Poland":"pl","Portugal":"pt",
  "Puerto Rico":"pr","Qatar":"qa","Romania":"ro","Russia":"ru","Rwanda":"rw",
  "Saint Kitts and Nevis":"kn","Saint Lucia":"lc","Saint Vincent and the Grenadines":"vc","Samoa":"ws","San Marino":"sm",
  "São Tomé and Príncipe":"st","Saudi Arabia":"sa","Scotland":"gb-sct","Senegal":"sn","Serbia":"rs",
  "Seychelles":"sc","Sierra Leone":"sl","Singapore":"sg","Slovakia":"sk","Slovenia":"si",
  "Solomon Islands":"sb","Somalia":"so","South Africa":"za","South Korea":"kr","South Sudan":"ss",
  "Spain":"es","Sri Lanka":"lk","Sudan":"sd","Suriname":"sr","Sweden":"se",
  "Switzerland":"ch","Syria":"sy","Tahiti":"pf","Tajikistan":"tj","Tanzania":"tz",
  "Thailand":"th","Timor-Leste":"tl","Togo":"tg","Tonga":"to","Trinidad and Tobago":"tt",
  "Tunisia":"tn","Turkey":"tr","Turkmenistan":"tm","Turks and Caicos Islands":"tc","Uganda":"ug",
  "Ukraine":"ua","United Arab Emirates":"ae","United States":"us","Uruguay":"uy","Uzbekistan":"uz",
  "Vanuatu":"vu","Venezuela":"ve","Vietnam":"vn","Wales":"gb-wls","Yemen":"ye",
  "Zambia":"zm","Zimbabwe":"zw"
};

function flagUrl(team){ const c = ISO[team]; return c ? `https://flagcdn.com/w80/${c}.png` : null; }
function flagImg(team, cls="flag"){
  const url = flagUrl(team);
  return url ? `<img class="${cls}" src="${url}" alt="${team}">` : `<span class="${cls} placeholder"></span>`;
}

let teams = [], home = null, away = null;

function loadTeams(){
  try {
    teams = listTeams();
    teams.sort();
    buildList("home"); buildList("away");
  } catch(e){ showStatus("Could not load the model data (model_data.js missing?)", true); }
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
  teams.filter(t => t.toLowerCase().includes(f)).forEach(t => {
    const li = document.createElement("li");
    li.className = "picker-item";
    li.setAttribute("role","option");
    li.innerHTML = `${flagImg(t)} <span>${t}</span>`;
    li.onclick = () => selectTeam(which, t);
    els.list.appendChild(li);
  });
}

function selectTeam(which, team){
  if (which === "home") home = team; else away = team;
  const els = pickerEls(which);
  els.flag.outerHTML = flagImg(team, "flag");
  els.name.textContent = team;
  els.name.classList.remove("muted");
  els.menu.hidden = true;
  document.getElementById("calc-btn").disabled = !(home && away);
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

document.getElementById("calc-btn").onclick = async () => {
  if (!home || !away) return;
  if (home === away){ showStatus("Pick two different teams.", true); return; }

  const body = {
    home_team: home,
    away_team: away,
    method: document.getElementById("method").value,
    home_advantage: document.getElementById("home-adv").checked,
    use_xg: document.getElementById("use-xg").checked,
  };

  showStatus("Calculating…");
  document.getElementById("calc-btn").disabled = true;

  // Let the status paint before the (synchronous) simulation runs.
  setTimeout(() => {
    try {
      const d = runPrediction(body);
      renderResults(d);
      hideStatus();
    } catch(e){ showStatus("Prediction failed: " + e.message, true); }
    finally { document.getElementById("calc-btn").disabled = false; }
  }, 20);
};

function pct(x){ return (x*100).toFixed(1) + "%"; }

function renderResults(d){
  const pHome = d.probabilities.home_win;
  const pDraw = d.probabilities.draw;
  const pAway = d.probabilities.away_win;
  const xgHome = d.expected_goals.home;
  const xgAway = d.expected_goals.away;
  const scores = d.simulation.top_scores;
  const trust = d.data_trust;

  document.getElementById("result-title").innerHTML =
    `${flagImg(home,"flag")} ${home} <span class="vsr">vs</span> ${away} ${flagImg(away,"flag")}`;
  document.getElementById("venue-note").textContent =
    document.getElementById("home-adv").checked ? `${home} playing at home` : "Neutral venue";

  setSeg("home", pHome); setSeg("draw", pDraw); setSeg("away", pAway);
  document.getElementById("leg-home-name").textContent = home;
  document.getElementById("leg-away-name").textContent = away;
  document.getElementById("leg-home-pct").textContent = pct(pHome);
  document.getElementById("leg-draw-pct").textContent = pct(pDraw);
  document.getElementById("leg-away-pct").textContent = pct(pAway);

  document.getElementById("xg-home").textContent = xgHome.toFixed(2);
  document.getElementById("xg-away").textContent = xgAway.toFixed(2);
  document.getElementById("xg-home-flag").src = flagUrl(home) || "";
  document.getElementById("xg-away-flag").src = flagUrl(away) || "";

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
  if (trust < 1){ tn.hidden = false; tn.textContent = "Note: limited recent data for one of these teams, so this estimate is less certain."; }
  else tn.hidden = true;

  document.getElementById("results").hidden = false;
  document.getElementById("results").scrollIntoView({behavior:"smooth", block:"start"});
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

/* ---------- refit (refit.js) ---------- */
document.getElementById("refit-btn").onclick = async () => {
  const btn = document.getElementById("refit-btn");
  const oldLabel = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = "Refitting…";
  try {
    const bundle = await refitModel(showStatus);
    MODEL_DATA.base = bundle;
    loadTeams(); // refresh pickers with any new teams
    btn.innerHTML = "&#10003; Up to date";
    btn.classList.add("done");
    showStatus("Model rebuilt on the latest results. Predictions now include recent matches.");
    setTimeout(hideStatus, 6000);
  } catch (e) {
    btn.innerHTML = oldLabel;
    showStatus("Refit failed: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
};

/* ---------- upcoming World Cup fixtures (TheSportsDB free API) ---------- */
const WORLD_CUP_LEAGUE_ID = 4429;

async function loadFixtures() {
  const box = document.getElementById("fixtures-list");
  try {
    const resp = await fetch(
      `https://www.thesportsdb.com/api/v1/json/123/eventsnextleague.php?id=${WORLD_CUP_LEAGUE_ID}`);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    const events = (data.events || []).filter((e) =>
      !["FT", "AET", "PEN", "Match Finished"].includes(e.strStatus));
    if (!events.length) {
      box.innerHTML = `<p class="fixtures-status">No upcoming World Cup matches scheduled.</p>`;
      return;
    }
    events.sort((a, b) => (a.strTimestamp || "").localeCompare(b.strTimestamp || ""));
    const shown = events.slice(0, 8);
    box.innerHTML = shown.map(fixtureRow).join("");
    box.querySelectorAll(".fx-load").forEach((btn, i) => {
      btn.onclick = () => loadFixtureIntoPickers(shown[i]);
    });
  } catch (e) {
    box.innerHTML = `<p class="fixtures-status">Couldn't load upcoming fixtures.</p>`;
  }
}

// Fixture team names come from TheSportsDB; map them to the model's spelling
// (refit.js's canonical map handles "USA" -> "United States" etc.).
function modelTeamName(name) {
  const t = typeof canonicalizeTeam === "function" ? canonicalizeTeam(name || "") : (name || "");
  return teams.includes(t) ? t : null;
}

function loadFixtureIntoPickers(e) {
  const h = modelTeamName(e.strHomeTeam);
  const a = modelTeamName(e.strAwayTeam);
  if (!h || !a) {
    showStatus(`Sorry, couldn't match "${e.strHomeTeam}" or "${e.strAwayTeam}" to a team in the model.`, true);
    return;
  }
  selectTeam("home", h);
  selectTeam("away", a);
  document.querySelector(".matchup-card").scrollIntoView({ behavior: "smooth", block: "start" });
  document.getElementById("calc-btn").click();
}

function fixtureRow(e) {
  // strTimestamp is UTC; show the viewer's local date + kick-off time.
  const t = e.strTimestamp ? new Date(e.strTimestamp + "Z") : null;
  let when = "Time TBD";
  if (t && !Number.isNaN(t.getTime())) {
    const hhmm = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    when = t.toDateString() === new Date().toDateString()
      ? `Today &middot; ${hhmm}`
      : `${t.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })} &middot; ${hhmm}`;
  }

  const live = e.intHomeScore != null && e.intAwayScore != null;
  const mid = live
    ? `<span class="fx-mid score">${e.intHomeScore}&ndash;${e.intAwayScore}</span>`
    : `<span class="fx-mid">vs</span>`;

  return `<div class="fixture">
    <div class="fx-league">${when}${live ? ' <span class="fx-live">LIVE</span>' : ""}</div>
    <div class="fx-row">
      <span class="fx-team">${e.strHomeTeam || "?"}</span>
      ${mid}
      <span class="fx-team away">${e.strAwayTeam || "?"}</span>
    </div>
    <button type="button" class="fx-load">&#9917; Predict this match</button>
  </div>`;
}

/* ---------- visit counter (CounterAPI) ---------- */
async function loadVisitCount() {
  try {
    const resp = await fetch("https://api.counterapi.dev/v1/xackerlud-worldcup/visits/up");
    const data = await resp.json();
    if (data.count) {
      const el = document.getElementById("visit-count");
      el.textContent = `⚽ ${Number(data.count).toLocaleString()} visits`;
      el.hidden = false;
    }
  } catch (e) { /* cosmetic — fail silently */ }
}

wirePicker("home");
wirePicker("away");
if (typeof MODEL_DATA !== "undefined" && !MODEL_DATA.xg) {
  document.getElementById("use-xg").disabled = true; // exported without the xG bundle
}
loadTeams();
loadFixtures();
loadVisitCount();