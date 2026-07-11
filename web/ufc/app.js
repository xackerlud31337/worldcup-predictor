// UFC page UI — mirrors the football app: pickers, prediction, glass panels.
// All prediction math lives in predictor.js on top of UFC_DATA.

let fighterA = null, fighterB = null;

function initials(name) {
  return name.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase();
}

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

function currentDivision() {
  return document.getElementById("division").value;
}

function buildList(which, filter = "") {
  const els = pickerEls(which);
  els.list.innerHTML = "";
  const f = filter.trim().toLowerCase();
  const names = listFighters(currentDivision()).filter((n) => n.toLowerCase().includes(f));
  if (!names.length) {
    const li = document.createElement("li");
    li.className = "picker-item empty";
    li.textContent = "No fighters found";
    els.list.appendChild(li);
    return;
  }
  names.slice(0, 400).forEach((n) => {
    const fd = ufcFighter(n);
    const li = document.createElement("li");
    li.className = "picker-item";
    li.setAttribute("role", "option");
    li.innerHTML = `<span class="avatar ${which === "away" ? "red" : ""}">${initials(n)}</span>
      <span>${n} <small style="color:var(--muted)">&middot; ${fd.rec[0]}-${fd.rec[1]}${fd.rec[2] ? "-" + fd.rec[2] : ""}</small></span>`;
    li.onclick = () => selectFighter(which, n);
    els.list.appendChild(li);
  });
}

function selectFighter(which, name) {
  if (which === "home") fighterA = name; else fighterB = name;
  const els = pickerEls(which);
  els.avatar.textContent = initials(name);
  els.name.textContent = name;
  els.name.classList.remove("muted");
  els.menu.hidden = true;
  document.getElementById("calc-btn").disabled = !(fighterA && fighterB);
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
  const a = fighterA, b = fighterB;
  fighterA = null; fighterB = null;
  if (b) selectFighter("home", b);
  if (a) selectFighter("away", a);
};

/* ---------- rendering ---------- */

function pct(x) { return (x * 100).toFixed(1) + "%"; }

function fmtAge(dob) {
  const a = agePct(dob);
  return a ? a.toFixed(0) : "?";
}

function fmtHeight(inches) {
  if (!inches) return "?";
  return `${Math.floor(inches / 12)}'${Math.round(inches % 12)}"`;
}

document.getElementById("calc-btn").onclick = () => {
  if (!fighterA || !fighterB) return;
  if (fighterA === fighterB) { showStatus("Pick two different fighters.", true); return; }
  hideStatus();
  const rounds = document.getElementById("five-rounds").checked ? 5 : 3;
  renderResults(predictFight(fighterA, fighterB, rounds));
};

function renderResults(r) {
  const A = r.a, B = r.b;

  document.getElementById("result-title").innerHTML =
    `<span class="avatar big">${initials(fighterA)}</span> ${fighterA}
     <span class="vsr">vs</span> ${fighterB} <span class="avatar big red">${initials(fighterB)}</span>`;
  document.getElementById("venue-note").textContent =
    `${A.div}${A.div !== B.div ? " vs " + B.div : ""} · scheduled for ${r.rounds} rounds`;

  // win bar (no draw segment — MMA draws are folded into the two sides)
  const segH = document.getElementById("seg-home"), segA = document.getElementById("seg-away");
  segH.style.width = (r.pA * 100).toFixed(1) + "%";
  segA.style.width = (r.pB * 100).toFixed(1) + "%";
  segH.querySelector("span").textContent = pct(r.pA);
  segA.querySelector("span").textContent = pct(r.pB);
  document.getElementById("leg-home-name").textContent = fighterA;
  document.getElementById("leg-away-name").textContent = fighterB;
  document.getElementById("leg-home-pct").textContent = pct(r.pA);
  document.getElementById("leg-away-pct").textContent = pct(r.pB);

  // tale of the tape
  const tape = document.getElementById("tape");
  const rows = [
    ["Record", `${A.rec[0]}-${A.rec[1]}${A.rec[2] ? "-" + A.rec[2] : ""}`, `${B.rec[0]}-${B.rec[1]}${B.rec[2] ? "-" + B.rec[2] : ""}`],
    ["Elo rating", A.elo.toFixed(0), B.elo.toFixed(0)],
    ["Age", fmtAge(A.dob), fmtAge(B.dob)],
    ["Height", fmtHeight(A.ht), fmtHeight(B.ht)],
    ["Reach", A.rc ? A.rc + '"' : "?", B.rc ? B.rc + '"' : "?"],
    ["Stance", A.st || "?", B.st || "?"],
    ["Last fight", A.last, B.last],
  ];
  tape.innerHTML = rows.map(([label, va, vb]) =>
    `<div class="tape-row"><span class="tape-a">${va}</span>
     <span class="tape-label">${label}</span><span class="tape-b">${vb}</span></div>`).join("");

  // method & round
  const ml = document.getElementById("method-list");
  const finA = r.pA * (r.methodA[0] + r.methodA[1]);
  const finB = r.pB * (r.methodB[0] + r.methodB[1]);
  const methodRows = [
    [`${fighterA} by KO/TKO`, r.pA * r.methodA[0]],
    [`${fighterA} by submission`, r.pA * r.methodA[1]],
    [`${fighterA} by decision`, r.pA * r.methodA[2]],
    [`${fighterB} by KO/TKO`, r.pB * r.methodB[0]],
    [`${fighterB} by submission`, r.pB * r.methodB[1]],
    [`${fighterB} by decision`, r.pB * r.methodB[2]],
    ["Goes the distance", 1 - r.pFinish],
  ];
  for (let i = 0; i < r.rounds; i++) {
    methodRows.push([`Finish in round ${i + 1}`, r.pFinish * r.roundMix[i]]);
  }
  ml.innerHTML = methodRows.map(([label, p]) =>
    `<div class="method-row"><span class="method-label">${label}</span>
     <span class="method-pct">${pct(p)}</span></div>`).join("");

  // stat comparison — mirrored bars
  const stats = [
    ["SLpM", A.slpm, B.slpm, false],
    ["Strike acc.", A.sacc * 100, B.sacc * 100, true],
    ["SApM (lower = better)", A.sapm, B.sapm, false],
    ["TD / 15 min", A.td15, B.td15, false],
    ["TD acc.", A.tdacc * 100, B.tdacc * 100, true],
    ["Subs / 15 min", A.sub15, B.sub15, false],
    ["KD / 15 min", A.kd15, B.kd15, false],
  ];
  const sc = document.getElementById("stat-compare");
  sc.innerHTML = stats.map(([label, va, vb, isPct]) => {
    const max = Math.max(va, vb, 0.001);
    const fmt = (v) => isPct ? v.toFixed(0) + "%" : v.toFixed(2).replace(/\.00$/, "");
    return `<div class="stat-row">
      <span class="stat-val left">${fmt(va)}</span>
      <span class="stat-track left"><span class="stat-fill a" style="width:${(va / max * 100).toFixed(0)}%"></span></span>
      <span class="stat-label">${label}</span>
      <span class="stat-track"><span class="stat-fill b" style="width:${(vb / max * 100).toFixed(0)}%"></span></span>
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

  // staleness / small-sample warnings
  const tn = document.getElementById("trust-note");
  const warnings = [];
  for (const [name, f] of [[fighterA, A], [fighterB, B]]) {
    const yearsOut = (Date.now() - new Date(f.last).getTime()) / (365.25 * 86400000);
    if (yearsOut > 2) warnings.push(`${name} hasn't fought in ${yearsOut.toFixed(1)} years`);
    if (f.rec[0] + f.rec[1] + f.rec[2] < 5) warnings.push(`${name} has few UFC fights on record`);
  }
  if (warnings.length) {
    tn.hidden = false;
    tn.textContent = "Take with extra salt: " + warnings.join("; ") +
      ". Ratings and profiles may not reflect their current form.";
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

/* ---------- upcoming UFC events (TheSportsDB + Wikipedia fight card) ---------- */

// diacritics-insensitive fighter lookup ("Benoît" matches "Benoit");
// rebuilt after a data refresh
let NORM_ROSTER = {};
function buildNormRoster() {
  NORM_ROSTER = {};
  for (const n of Object.keys(UFC_DATA.fighters)) {
    NORM_ROSTER[n.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()] = n;
  }
}
buildNormRoster();
function rosterName(wikiName) {
  return NORM_ROSTER[(wikiName || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase()] || null;
}

function eventWhen(e) {
  // Date only: the feed's clock time is the venue listing (doors/early
  // prelims), not the main card, so showing it as a precise local time
  // would mislead. Fight cards span many hours anyway.
  const t = e.dateEvent ? new Date(e.dateEvent + "T12:00:00Z") : null;
  return t && !Number.isNaN(t.getTime())
    ? t.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" })
    : "Date TBD";
}

// Top bouts (main event first) parsed from the Wikipedia page of a numbered
// UFC event — its REST API allows browser requests from anywhere.
async function fetchCard(eventName) {
  const m = (eventName || "").match(/^UFC (\d+)/);
  if (!m) return [];
  const resp = await fetch(`https://en.wikipedia.org/api/rest_v1/page/html/UFC_${m[1]}`);
  if (!resp.ok) return [];
  const doc = new DOMParser().parseFromString(await resp.text(), "text/html");
  const bouts = [];
  for (const tr of doc.querySelectorAll("tr")) {
    const cells = [...tr.querySelectorAll("td,th")].map((c) => c.textContent.trim());
    if (cells.length >= 4 && /^vs\.?$/i.test(cells[2] || "")) {
      bouts.push({ wc: cells[0], a: cells[1], b: cells[3] });
    }
  }
  return bouts;
}

function loadFightIntoPickers(a, b, fiveRounds) {
  selectFighter("home", a);
  selectFighter("away", b);
  document.getElementById("five-rounds").checked = fiveRounds;
  document.querySelector(".matchup-card").scrollIntoView({ behavior: "smooth", block: "start" });
  document.getElementById("calc-btn").click();
}

async function loadEvents() {
  const box = document.getElementById("fixtures-list");
  try {
    const resp = await fetch("https://www.thesportsdb.com/api/v1/json/123/eventsnextleague.php?id=4443");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const events = (await resp.json()).events || [];
    if (!events.length) {
      box.innerHTML = `<p class="fixtures-status">No upcoming UFC events found.</p>`;
      return;
    }

    const next = events[0];
    let bouts = [];
    try { bouts = (await fetchCard(next.strEvent)).slice(0, 2); } catch (e) { /* card is a bonus */ }

    let html = `<div class="fixture fx-event-head">
      <div class="fx-league">${eventWhen(next)}</div>
      <div class="fx-fight">${next.strEvent || "?"}</div>
      ${next.strVenue ? `<div class="fx-venue">${next.strVenue}${next.strCity ? ", " + next.strCity : ""}</div>` : ""}
    </div>`;

    const wired = [];
    bouts.forEach((bt, i) => {
      const a = rosterName(bt.a), b = rosterName(bt.b);
      const tag = i === 0 ? "Main event" : "Co-main event";
      let action;
      if (a && b) {
        action = `<button type="button" class="fx-load" data-bout="${i}">&#129354; Predict this fight</button>`;
        wired.push({ i, a, b, five: i === 0 });
      } else {
        action = `<div class="fx-note">${!a ? bt.a : bt.b} isn't in the model yet (no UFC record)</div>`;
      }
      html += `<div class="fixture">
        <div class="fx-league">${tag} &middot; ${bt.wc}</div>
        <div class="fx-fight">${bt.a} <span class="fx-vs">vs</span> ${bt.b}</div>
        ${action}
      </div>`;
    });

    for (const e of events.slice(1, 4)) {
      html += `<div class="fixture">
        <div class="fx-league">${eventWhen(e)}</div>
        <div class="fx-fight small">${e.strEvent || "?"}</div>
      </div>`;
    }

    box.innerHTML = html;
    wired.forEach(({ i, a, b, five }) => {
      box.querySelector(`.fx-load[data-bout="${i}"]`).onclick = () => loadFightIntoPickers(a, b, five);
    });
  } catch (e) {
    box.innerHTML = `<p class="fixtures-status">Couldn't load upcoming events.</p>`;
  }
}

/* ---------- fetch latest fight data (refit.js) ---------- */
document.getElementById("refit-btn").onclick = async () => {
  const btn = document.getElementById("refit-btn");
  const oldLabel = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = "Fetching…";
  try {
    const fresh = await refitUfc(showStatus);
    UFC_DATA.fighters = fresh.fighters;
    UFC_DATA.divisions = fresh.divisions;
    UFC_DATA.global_wm = fresh.global_wm;
    buildNormRoster();
    buildList("home"); buildList("away");
    btn.innerHTML = "&#10003; Up to date";
    btn.classList.add("done");
    showStatus(`Rebuilt from ${fresh.bouts.toLocaleString()} bouts. Records and ratings now match the newest available data.`);
    setTimeout(hideStatus, 7000);
  } catch (e) {
    btn.innerHTML = oldLabel;
    showStatus("Fetch failed: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
};

/* ---------- visit counter (shared with the rest of SportMath) ---------- */
async function loadVisitCount() {
  try {
    const resp = await fetch("https://api.counterapi.dev/v1/xackerlud-worldcup/visits/up");
    const data = await resp.json();
    if (data.count) {
      const el = document.getElementById("visit-count");
      el.textContent = `🥊 ${Number(data.count).toLocaleString()} visits`;
      el.hidden = false;
    }
  } catch (e) { /* cosmetic */ }
}

/* ---------- boot ---------- */
const divSel = document.getElementById("division");
UFC_DATA.divisions.forEach((d) => {
  const o = document.createElement("option");
  o.value = d; o.textContent = d;
  divSel.appendChild(o);
});
divSel.onchange = () => { buildList("home"); buildList("away"); };

wirePicker("home");
wirePicker("away");
buildList("home");
buildList("away");
loadEvents();
loadVisitCount();
