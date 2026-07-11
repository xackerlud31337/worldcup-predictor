"""
ufc_build.py  —  fit the UFC model and dump it to web/ufc/ufc_model_data.js
===========================================================================
Data source: community scrape of ufcstats.com (Greco1899/scrape_ufc_stats).
CSVs are cached in data_cache/ufc/; pass --download to refresh them.

The model, mirroring the football pipeline's offline-fit / in-browser-predict
split:
  * per-fighter Elo fitted chronologically over every UFC bout
    (finishes move ratings more than decisions),
  * a calibrated logistic Elo->win-probability scale,
  * per-fighter career rates (sig. strikes landed/absorbed per minute,
    accuracy, takedowns & submissions per 15 min, knockdowns),
  * win/loss method profiles (KO / submission / decision) and a
    finish-round histogram for method & round predictions,
  * tale-of-the-tape physicals (height, reach, stance, DOB).

Re-run whenever you want the site to pick up fresh fights:
  python3 ufc_build.py --download
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import urllib.request
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache", "ufc")
OUT = os.path.join(HERE, "web", "ufc", "ufc_model_data.js")

RAW_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = ["ufc_event_details.csv", "ufc_fight_results.csv",
         "ufc_fight_stats.csv", "ufc_fighter_tott.csv"]

ELO_START = 1500.0
ELO_K = 56.0
FINISH_MULT = 1.2
MIN_FIGHTS = 2             # roster filter
MIN_LAST_FIGHT = "2015-01-01"


# ---------------------------------------------------------------- parsing --

def download() -> None:
    os.makedirs(CACHE, exist_ok=True)
    for f in FILES:
        print(f"  downloading {f} ...")
        urllib.request.urlretrieve(RAW_BASE + f, os.path.join(CACHE, f))


def read_csv(name: str) -> list[dict]:
    with open(os.path.join(CACHE, name), newline="", encoding="utf-8") as fh:
        return [{k.strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh)]


def parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%B %d, %Y")
    except ValueError:
        try:
            return datetime.strptime(s, "%b %d, %Y")
        except ValueError:
            return None


def parse_height(s: str) -> float | None:
    # 5' 11" -> inches
    if "'" not in s:
        return None
    try:
        ft, rest = s.split("'")
        inches = rest.replace('"', "").strip()
        return int(ft) * 12 + (int(inches) if inches else 0)
    except ValueError:
        return None


def parse_reach(s: str) -> float | None:
    s = s.replace('"', "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_of(s: str) -> tuple[int, int]:
    # "23 of 38" -> (23, 38)
    try:
        a, b = s.split(" of ")
        return int(a), int(b)
    except ValueError:
        return 0, 0


def parse_mmss(s: str) -> int:
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except ValueError:
        return 0


def method_class(method: str) -> str:
    m = method.lower()
    if m.startswith(("ko/tko", "tko")):
        return "ko"
    if m.startswith("submission"):
        return "sub"
    if m.startswith("decision"):
        return "dec"
    return "other"


def clean_division(wc: str) -> str:
    wc = (wc.replace("UFC", "").replace("Title", "").replace("Bout", "")
            .replace("Interim", "").replace("Tournament", "")
            .replace("Ultimate Fighter", "").replace("Australia", "")
            .replace("UK", "").replace("vs.", "").replace("Latin America", "")
            .replace("China", "").replace("Brazil", ""))
    wc = " ".join(wc.split()).lstrip("0123456789 ")  # "25 Welterweight" -> "Welterweight"
    return wc if wc else "Unknown"


# ---------------------------------------------------------------- fitting --

def fit_logistic(X, y, iters=3000, lr=0.5, l2=1e-4):
    """Full-batch logistic regression without intercept (features are
    antisymmetric A-vs-B differences, so the model must be symmetric)."""
    k = len(X[0])
    w = [0.0] * k
    n = len(X)
    for _ in range(iters):
        grad = [l2 * wi for wi in w]
        for xi, yi in zip(X, y):
            z = sum(wj * xj for wj, xj in zip(w, xi))
            p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
            e = p - yi
            for j in range(k):
                grad[j] += e * xi[j] / n
        for j in range(k):
            w[j] -= lr * grad[j]
    return w


def log_loss(w, X, y):
    ll = 0.0
    for xi, yi in zip(X, y):
        z = sum(wj * xj for wj, xj in zip(w, xi))
        p = 1.0 / (1.0 + math.exp(-max(min(z, 30), -30)))
        p = min(max(p, 1e-12), 1 - 1e-12)
        ll -= yi * math.log(p) + (1 - yi) * math.log(1 - p)
    return ll / len(X)


def accuracy(w, X, y):
    hit = sum(1 for xi, yi in zip(X, y)
              if (sum(wj * xj for wj, xj in zip(w, xi)) > 0) == (yi == 1.0))
    return hit / len(X)


def build():
    events = {r["EVENT"].strip(): parse_date(r["DATE"]) for r in read_csv("ufc_event_details.csv")}

    fights = []
    for r in read_csv("ufc_fight_results.csv"):
        date = events.get(r["EVENT"].strip())
        bout = r["BOUT"]
        if date is None or " vs. " not in bout:
            continue
        a, b = [x.strip() for x in bout.split(" vs. ", 1)]
        outcome = r["OUTCOME"].strip()
        if outcome not in ("W/L", "L/W", "D/D", "NC/NC"):
            continue
        try:
            rnd = int(r["ROUND"])
        except ValueError:
            continue
        fights.append({
            "date": date, "a": a, "b": b, "outcome": outcome,
            "division": clean_division(r["WEIGHTCLASS"]),
            "method": method_class(r["METHOD"]),
            "round": rnd,
            "secs": (rnd - 1) * 300 + parse_mmss(r["TIME"]),
            "event": r["EVENT"].strip(), "bout": bout.strip(),
        })
    fights.sort(key=lambda f: f["date"])
    print(f"  {len(fights)} bouts across {len(events)} events")

    # tale-of-the-tape (needed early: DOB feeds the age feature)
    tott = {}
    for r in read_csv("ufc_fighter_tott.csv"):
        tott[r["FIGHTER"].strip()] = {
            "ht": parse_height(r["HEIGHT"]),
            "rc": parse_reach(r["REACH"]),
            "st": r["STANCE"] or None,
            "dob": parse_date(r["DOB"]),
        }

    def age_at(name, when):
        d = tott.get(name, {}).get("dob")
        return (when - d).days / 365.25 if d else None

    # --- Elo, chronological, collecting pre-fight features -----------------
    elo: dict[str, float] = defaultdict(lambda: ELO_START)
    n_fights: dict[str, int] = defaultdict(int)
    last_date: dict[str, datetime] = {}
    cal = []  # rated bouts: features known before the fight + outcome
    for f in fights:
        if f["outcome"] == "NC/NC":
            continue
        ra, rb = elo[f["a"]], elo[f["b"]]
        if f["outcome"] != "D/D" and n_fights[f["a"]] >= 3 and n_fights[f["b"]] >= 3:
            lay_a = (f["date"] - last_date[f["a"]]).days
            lay_b = (f["date"] - last_date[f["b"]]).days
            age_a, age_b = age_at(f["a"], f["date"]), age_at(f["b"], f["date"])
            cal.append({
                "date": f["date"],
                "x": [
                    (ra - rb) / 100.0,                                  # rating gap
                    math.log1p(lay_a / 365.0) - math.log1p(lay_b / 365.0),  # ring rust
                    ((age_a - age_b) / 10.0) if age_a and age_b else 0.0,   # age gap
                ],
                "y": 1.0 if f["outcome"] == "W/L" else 0.0,
            })
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        score_a = {"W/L": 1.0, "L/W": 0.0, "D/D": 0.5}[f["outcome"]]
        k = ELO_K * (FINISH_MULT if f["method"] in ("ko", "sub") else 1.0)
        delta = k * (score_a - exp_a)
        elo[f["a"]] = ra + delta
        elo[f["b"]] = rb - delta
        n_fights[f["a"]] += 1
        n_fights[f["b"]] += 1
        last_date[f["a"]] = f["date"]
        last_date[f["b"]] = f["date"]

    # legacy scale calibration (kept as the JS fallback path)
    best_scale, best_ll = 400.0, float("inf")
    for scale in range(200, 651, 25):
        ll = 0.0
        for c in cal:
            diff = c["x"][0] * 100.0
            p = 1.0 / (1.0 + 10.0 ** (-diff / scale))
            p = min(max(p, 1e-9), 1 - 1e-9)
            ll -= c["y"] * math.log(p) + (1 - c["y"]) * math.log(1 - p)
        if ll < best_ll:
            best_ll, best_scale = ll, float(scale)
    acc = sum(1 for c in cal if (c["x"][0] > 0) == (c["y"] == 1.0)) / max(len(cal), 1)
    print(f"  Elo scale {best_scale:.0f} (log-loss {best_ll/len(cal):.4f}, "
          f"accuracy {acc:.1%} on {len(cal)} rated bouts)")

    # --- ring-rust model: Elo gap + layoff + age, gated by a backtest -------
    split = datetime(2023, 1, 1)
    train = [c for c in cal if c["date"] < split]
    test = [c for c in cal if c["date"] >= split]
    Xtr, ytr = [c["x"] for c in train], [c["y"] for c in train]
    Xte, yte = [c["x"] for c in test], [c["y"] for c in test]

    w_base = fit_logistic([[x[0]] for x in Xtr], ytr)
    w_full = fit_logistic(Xtr, ytr)
    ll_base = log_loss(w_base, [[x[0]] for x in Xte], yte)
    ll_full = log_loss(w_full, Xte, yte)
    acc_base = accuracy(w_base, [[x[0]] for x in Xte], yte)
    acc_full = accuracy(w_full, Xte, yte)
    print(f"  backtest on {len(test)} bouts since {split.date()}:")
    print(f"    Elo only          log-loss {ll_base:.4f}  accuracy {acc_base:.1%}")
    print(f"    + layoff + age    log-loss {ll_full:.4f}  accuracy {acc_full:.1%}")

    form = None
    if ll_full < ll_base - 0.0005:
        # ships: refit the winning model on ALL rated bouts
        w_ship = fit_logistic([c["x"] for c in cal], [c["y"] for c in cal])
        form = {"w_elo": round(w_ship[0], 4), "w_layoff": round(w_ship[1], 4),
                "w_age": round(w_ship[2], 4)}
        print(f"    SHIPPED: {form}")
    else:
        print("    NOT SHIPPED: no held-out improvement, Elo-only stays")

    # --- per-fight totals from the round-by-round stats ---------------------
    # key: (event, bout, fighter) -> aggregated numbers
    totals = defaultdict(lambda: [0] * 7)  # sigL, sigA, tdL, tdA, subAtt, kd, rounds
    for r in read_csv("ufc_fight_stats.csv"):
        key = (r["EVENT"].strip(), r["BOUT"].strip(), r["FIGHTER"].strip())
        t = totals[key]
        sl, sa = parse_of(r["SIG.STR."])
        tl, ta = parse_of(r["TD"])
        t[0] += sl; t[1] += sa; t[2] += tl; t[3] += ta
        try:
            t[4] += int(r["SUB.ATT"])
        except ValueError:
            pass
        try:
            t[5] += int(r["KD"])
        except ValueError:
            pass
        t[6] += 1

    # --- career aggregates ---------------------------------------------------
    F = defaultdict(lambda: {
        "w": 0, "l": 0, "d": 0, "wm": [0, 0, 0], "lm": [0, 0, 0],
        "rh": [0, 0, 0, 0, 0], "secs": 0,
        "sigL": 0, "sigA": 0, "absL": 0, "tdL": 0, "tdA": 0,
        "sub": 0, "kd": 0, "div": "Unknown", "last": None, "n": 0,
    })
    M = {"ko": 0, "sub": 1, "dec": 2}
    for f in fights:
        if f["outcome"] == "NC/NC":
            continue
        for me, opp, res in ((f["a"], f["b"], f["outcome"]),
                             (f["b"], f["a"], f["outcome"][::-1])):
            st = F[me]
            st["n"] += 1
            st["secs"] += f["secs"]
            st["div"] = f["division"]   # most recent wins (chronological order)
            st["last"] = f["date"]
            mine = totals.get((f["event"], f["bout"], me))
            theirs = totals.get((f["event"], f["bout"], opp))
            if mine:
                st["sigL"] += mine[0]; st["sigA"] += mine[1]
                st["tdL"] += mine[2]; st["tdA"] += mine[3]
                st["sub"] += mine[4]; st["kd"] += mine[5]
            if theirs:
                st["absL"] += theirs[0]
            first = res == "W/L"
            if res == "D/D":
                st["d"] += 1
            elif first:
                st["w"] += 1
                if f["method"] in M:
                    st["wm"][M[f["method"]]] += 1
                if f["method"] in ("ko", "sub"):
                    st["rh"][min(f["round"], 5) - 1] += 1
            else:
                st["l"] += 1
                if f["method"] in M:
                    st["lm"][M[f["method"]]] += 1

    # --- roster filter + bundle ---------------------------------------------
    min_last = datetime.strptime(MIN_LAST_FIGHT, "%Y-%m-%d")
    fighters = {}
    div_wm = defaultdict(lambda: [0, 0, 0])
    for name, st in F.items():
        if st["n"] < MIN_FIGHTS or st["last"] is None or st["last"] < min_last:
            continue
        mins = st["secs"] / 60.0 or 1.0
        t = tott.get(name, {})
        fighters[name] = {
            "div": st["div"],
            "elo": round(elo[name], 1),
            "rec": [st["w"], st["l"], st["d"]],
            "wm": st["wm"], "lm": st["lm"], "rh": st["rh"],
            "slpm": round(st["sigL"] / mins, 2),
            "sacc": round(st["sigL"] / st["sigA"], 3) if st["sigA"] else 0,
            "sapm": round(st["absL"] / mins, 2),
            "td15": round(st["tdL"] / (mins / 15.0), 2),
            "tdacc": round(st["tdL"] / st["tdA"], 3) if st["tdA"] else 0,
            "sub15": round(st["sub"] / (mins / 15.0), 2),
            "kd15": round(st["kd"] / (mins / 15.0), 2),
            "mins": round(mins, 1),
            "ht": t.get("ht"), "rc": t.get("rc"),
            "st": t.get("st"),
            "dob": t["dob"].strftime("%Y-%m-%d") if t.get("dob") else None,
            "last": st["last"].strftime("%Y-%m-%d"),
        }
        for i in range(3):
            div_wm[st["div"]][i] += st["wm"][i]

    divisions = sorted({f["div"] for f in fighters.values() if f["div"] != "Unknown"})
    total_wm = [sum(v[i] for v in div_wm.values()) for i in range(3)]

    bundle = {
        "config": {
            "elo_scale": best_scale,
            "elo_start": ELO_START,
            "form": form,   # ring-rust logistic coefficients, or null if not shipped
            "built": datetime.now().strftime("%Y-%m-%d"),
            "accuracy_note": f"{acc:.1%} winner accuracy on {len(cal)} rated bouts",
        },
        "divisions": divisions,
        "global_wm": total_wm,
        "fighters": fighters,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    js = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("// Generated by ufc_build.py — fitted UFC model for the static site.\n")
        fh.write(f"const UFC_DATA = {js};\n")
    print(f"  {len(fighters)} fighters in {len(divisions)} divisions -> "
          f"{os.path.getsize(OUT)/1024:.0f} KB written to {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="refresh the CSV cache first")
    args = ap.parse_args()
    if args.download or not all(os.path.exists(os.path.join(CACHE, f)) for f in FILES):
        download()
    build()
