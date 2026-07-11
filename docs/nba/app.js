// NBA page UI — mirrors the UFC app: pickers, prediction, glass panels.
// All prediction math lives in predictor.js on top of NBA_DATA.

let teamA = null, teamB = null;

function pickerEls(which) {
  const root = document.getElementById(`${which}-picker`);
  return {
    root,
    toggle: root.querySelector(".picker-toggle"),
    menu: root.querySelector(".picker-menu"),
    search: root.querySelector(".picker-search"),
    list: root.querySelector(".picker-list"),
    avatar: root.querySelector(".picker-toggle .avatar"),
    name: root.querySelector(".picker-name"),
  };
}

function currentConf() {
  return document.getElementById("conference").value;
}

function buildList(which, filter = "") {
  const els = pickerEls(which);
  els.list.innerHTML = "";
  const f = filter.trim().toLowerCase();
  const names = listTeams(currentConf()).filter((n) => n.toLowerCase().includes(f));
  if (!names.length) {
    const li = document.createElement("li");
    li.className = "picker-item empty";
    li.textContent = "No teams found";
    els.list.appendChild(li);
    return;
  }
  names.forEach((n) => {
    const td = nbaTeam(n);
    const li = document.createElement("li");
    li.className = "picker-item";
    li.setAttribute("role", "option");
    li.innerHTML = `<span class="avatar ${which === "away" ? "red" : ""}">${td.abbr}</span>
      <span>${n} <small style="color:var(--muted)">&middot; ${td.rec[0]}-${td.rec[1]}</small></span>`;
    li.onclick = () => selectTeam(which, n);
    els.list.appendChild(li);
  });
}

function selectTeam(which, name) {
  if (which === "home") teamA = name; else teamB = name;
  const els = pickerEls(which);
  els.avatar.textContent = nbaTeam(name).abbr;
  els.name.textContent = name;
  els.name.classList.remove("muted");
  els.menu.hidden = true;
  document.getElementById("calc-btn").disabled = !(teamA && teamB);
}

function wirePicker(which) {
  const els = pickerEls(which);
  els.toggle.onclick = () => {
    els.menu.hidden = !els.menu.hidden;
    if (!els.menu.hidden) { els.search.value = ""; buildList(which); els.search.focus(); }
  };
  els.search.oninput = () => buildList(which, els.search.value);
}

document.addEventListener("click", (e) => {
  ["home", "away"].forEach((w) => {
    const els = pickerEls(w);
    if (!els.root.contains(e.target)) els.menu.hidden = true;
  });
});

document.getElementById("swap-btn").onclick = () => {
  const a = teamA, b = teamB;
  teamA = null; teamB = null;
  if (b) selectTeam("home", b);
  if (a) selectTeam("away", a);
  // keep the venue pointing at the same physical court
  const v = document.getElementById("venue");
  if (v.value !== "neutral") v.value = v.value === "home" ? "away" : "home";
};

/* ---------- rendering ---------- */

function pct(x) { return (x * 100).toFixed(1) + "%"; }

function fmtStreak(s) {
  return s > 0 ? `W${s}` : `L${-s}`;
}

document.getElementById("calc-btn").onclick = () => {
  if (!teamA || !teamB) return;
  if (teamA === teamB) { showStatus("Pick two different teams.", true); return; }
  hideStatus();
  renderResults(predictGame(teamA, teamB, document.getElementById("venue").value));
};

function renderResults(r) {
  const A = r.a, B = r.b;

  document.getElementById("result-title").innerHTML =
    `<span class="avatar big">${A.abbr}</span> ${teamA}
     <span class="vsr">vs</span> ${teamB} <span class="avatar big red">${B.abbr}</span>`;
  const venueTxt = r.venue === "home" ? `at ${teamA}`
    : r.venue === "away" ? `at ${teamB}` : "neutral court";
  document.getElementById("venue-note").textContent =
    `${A.conf === B.conf ? A.conf + "ern Conference" : "Inter-conference"} matchup · ${venueTxt}`;

  // win bar (no draws in the NBA — overtime settles everything)
  const segH = document.getElementById("seg-home"), segA = document.getElementById("seg-away");
  segH.style.width = (r.pA * 100).toFixed(1) + "%";
  segA.style.width = (r.pB * 100).toFixed(1) + "%";
  segH.querySelector("span").textContent = pct(r.pA);
  segA.querySelector("span").textContent = pct(r.pB);
  document.getElementById("leg-home-name").textContent = teamA;
  document.getElementById("leg-away-name").textContent = teamB;
  document.getElementById("leg-home-pct").textContent = pct(r.pA);
  document.getElementById("leg-away-pct").textContent = pct(r.pB);

  // team sheet
  const tape = document.getElementById("tape");
  const rows = [
    [`Record (${NBA_CFG.season})`, `${A.rec[0]}-${A.rec[1]}`, `${B.rec[0]}-${B.rec[1]}`],
    ["Elo rating", A.elo.toFixed(0), B.elo.toFixed(0)],
    ["Home record", `${A.home_rec[0]}-${A.home_rec[1]}`, `${B.home_rec[0]}-${B.home_rec[1]}`],
    ["Road record", `${A.away_rec[0]}-${A.away_rec[1]}`, `${B.away_rec[0]}-${B.away_rec[1]}`],
    ["Last 10", `${A.l10}-${10 - A.l10}`, `${B.l10}-${10 - B.l10}`],
    ["Streak", fmtStreak(A.streak), fmtStreak(B.streak)],
    ["Last game", A.last, B.last],
  ];
  tape.innerHTML = rows.map(([label, va, vb]) =>
    `<div class="tape-row"><span class="tape-a">${va}</span>
     <span class="tape-label">${label}</span><span class="tape-b">${vb}</span></div>`).join("");

  // predicted score + margins
  document.getElementById("pred-score").innerHTML =
    `<span class="score a">${Math.round(r.scoreA)}</span>
     <span class="score-sep">–</span>
     <span class="score b">${Math.round(r.scoreB)}</span>`;
  document.getElementById("score-note").textContent =
    `fair spread ${teamA} ${r.spread > 0 ? "-" : "+"}${Math.abs(r.spread).toFixed(1)} · total ${r.line.toFixed(1)} pts`;
  document.getElementById("margin-list").innerHTML = r.margins.map(([label, p]) =>
    `<div class="method-row"><span class="method-label">${label}</span>
     <span class="method-pct">${pct(p)}</span></div>`).join("");

  // stat comparison — mirrored bars
  const stats = [
    ["Points per game", A.ppg, B.ppg, 1],
    ["Points allowed (lower = better)", A.oppg, B.oppg, 1],
    ["Net rating (avg margin)", A.ppg - A.oppg, B.ppg - B.oppg, 1],
    ["Recent scoring form", A.off, B.off, 1],
    ["Recent defence (lower = better)", A.def, B.def, 1],
    ["Win %", A.rec[0] / Math.max(A.rec[0] + A.rec[1], 1) * 100,
      B.rec[0] / Math.max(B.rec[0] + B.rec[1], 1) * 100, 0],
  ];
  const sc = document.getElementById("stat-compare");
  sc.innerHTML = stats.map(([label, va, vb, dp]) => {
    const lo = Math.min(va, vb, 0);
    const max = Math.max(va - lo, vb - lo, 0.001);
    const fmt = (v) => label === "Win %" ? v.toFixed(0) + "%" : v.toFixed(dp);
    return `<div class="stat-row">
      <span class="stat-val left">${fmt(va)}</span>
      <span class="stat-track left"><span class="stat-fill a" style="width:${((va - lo) / max * 100).toFixed(0)}%"></span></span>
      <span class="stat-label">${label}</span>
      <span class="stat-track"><span class="stat-fill b" style="width:${((vb - lo) / max * 100).toFixed(0)}%"></span></span>
      <span class="stat-val">${fmt(vb)}</span>
    </div>`;
  }).join("");

  // fair odds
  const mkl = document.getElementById("market-list");
  mkl.innerHTML = "";
  r.markets.forEach(({ label, p }) => {
    const fair = p > 0.005 ? (1 / p).toFixed(2) : "99+";
    const row = document.createElement("div");
    row.className = "market-row";
    row.innerHTML = `<span class="market-label">${label}</span>
      <span class="market-pct">${pct(p)}</span>
      <span class="market-odds">${fair}</span>`;
    mkl.appendChild(row);
  });

  // staleness warning (mostly for the offseason)
  const tn = document.getElementById("trust-note");
  const monthsOut = (Date.now() - new Date(Math.max(
    new Date(A.last), new Date(B.last)))) / (30.44 * 86400000);
  if (monthsOut > 1.5) {
    tn.hidden = false;
    tn.textContent = `Take with extra salt: the season is over, so ratings and rosters are
      frozen at ${NBA_CFG.season} form. Trades, drafts and injuries since then aren't in the model.`;
  } else tn.hidden = true;

  document.getElementById("results").hidden = false;
  document.getElementById("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showStatus(msg, isErr = false) {
  const s = document.getElementById("status");
  s.hidden = false; s.textContent = msg;
  s.classList.toggle("error", isErr);
}
function hideStatus() { document.getElementById("status").hidden = true; }

/* ---------- info & donate popups ---------- */
const infoOverlay = document.getElementById("info-overlay");
document.getElementById("info-btn").onclick = () => { infoOverlay.hidden = false; };
document.getElementById("info-close").onclick = () => { infoOverlay.hidden = true; };
infoOverlay.addEventListener("click", (e) => {
  if (e.target === infoOverlay) infoOverlay.hidden = true;
});

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
    try { await navigator.clipboard.writeText(btn.dataset.copy); }
    catch (e) {
      const ta = document.createElement("textarea");
      ta.value = btn.dataset.copy;
      document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); ta.remove();
    }
    const old = btn.textContent;
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = old; btn.classList.remove("copied"); }, 1500);
  };
});

/* ---------- upcoming NBA games (ESPN, the same source the model is built on) ---------- */

function loadGameIntoPickers(homeName, awayName) {
  selectTeam("home", homeName);
  selectTeam("away", awayName);
  document.getElementById("venue").value = "home";
  document.querySelector(".matchup-card").scrollIntoView({ behavior: "smooth", block: "start" });
  document.getElementById("calc-btn").click();
}

function fmtGameDate(iso) {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? "Date TBD"
    : t.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" }) +
      " · " + t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function loadGames() {
  const box = document.getElementById("fixtures-list");
  const ymd = (d) => d.toISOString().slice(0, 10).replace(/-/g, "");
  const now = new Date();
  const to = new Date(now.getTime() + 60 * 86400000);
  try {
    const resp = await fetch("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/" +
      `scoreboard?dates=${ymd(now)}-${ymd(to)}&limit=40`);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const events = ((await resp.json()).events || []).filter((e) =>
      e.competitions?.[0]?.status?.type?.name === "STATUS_SCHEDULED");
    if (!events.length) {
      box.innerHTML = `<p class="fixtures-status">Offseason &mdash; no games on the calendar yet.
        The new season's schedule lands here in August, tip-off is in October.</p>`;
      return;
    }

    let html = "";
    const wired = [];
    events.slice(0, 6).forEach((ev, i) => {
      const comp = ev.competitions[0];
      const home = comp.competitors.find((c) => c.homeAway === "home");
      const away = comp.competitors.find((c) => c.homeAway === "away");
      const hName = home?.team?.displayName, aName = away?.team?.displayName;
      const tag = ev.season?.type === 1 ? "Preseason" :
        ev.season?.type === 3 ? "Playoffs" : "Regular season";
      let action = "";
      if (nbaTeam(hName) && nbaTeam(aName)) {
        action = `<button type="button" class="fx-load" data-game="${i}">&#127936; Predict this game</button>`;
        wired.push({ i, h: hName, a: aName });
      }
      html += `<div class="fixture${i === 0 ? " fx-event-head" : ""}">
        <div class="fx-league">${fmtGameDate(ev.date)} &middot; ${tag}</div>
        <div class="fx-fight${i === 0 ? "" : " small"}">${aName} <span class="fx-vs">at</span> ${hName}</div>
        ${action}
      </div>`;
    });

    box.innerHTML = html;
    wired.forEach(({ i, h, a }) => {
      box.querySelector(`.fx-load[data-game="${i}"]`).onclick = () => loadGameIntoPickers(h, a);
    });
  } catch (e) {
    box.innerHTML = `<p class="fixtures-status">Couldn't load upcoming games.</p>`;
  }
}

/* ---------- visit counter (shared with the rest of SportMath) ---------- */
async function loadVisitCount() {
  try {
    const resp = await fetch("https://api.counterapi.dev/v1/xackerlud-worldcup/visits/up");
    const data = await resp.json();
    if (data.count) {
      const el = document.getElementById("visit-count");
      el.textContent = `🏀 ${Number(data.count).toLocaleString()} visits`;
      el.hidden = false;
    }
  } catch (e) { /* cosmetic */ }
}

/* ---------- boot ---------- */
document.getElementById("conference").onchange = () => { buildList("home"); buildList("away"); };

wirePicker("home");
wirePicker("away");
buildList("home");
buildList("away");
loadGames();
loadVisitCount();
