"""
api.py  —  FastAPI backend
==========================
Thin HTTP layer over the existing model. The fitted bundle is built once on
startup via main.build_current_model (which also caches to disk) and held in
memory; the xG-blended variant is fitted lazily on the first request that asks
for it, since it downloads StatsBomb event data.

Run with:
  uvicorn api:app --reload
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import predict as predict_mod
import simulate as simulate_mod
from main import build_current_model, resolve_team

MAX_SIMS = 100_000

# use_xg -> (model, elo_ratings, draw_params)
_bundles: dict[bool, tuple] = {}
_fit_lock = threading.Lock()


def _get_bundle(use_xg: bool) -> tuple:
    if use_xg not in _bundles:
        with _fit_lock:
            if use_xg not in _bundles:  # re-check after acquiring the lock
                _bundles[use_xg] = build_current_model(use_xg=use_xg)
    return _bundles[use_xg]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_bundle(use_xg=False)  # fit (or load from disk cache) before serving
    yield
    _bundles.clear()


app = FastAPI(title="World Cup Predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    sims: int = Field(default=10_000, ge=1,
                      description=f"Monte-Carlo samples (capped at {MAX_SIMS})")
    use_xg: bool = False
    home_advantage: bool = Field(
        default=False, description="first team plays at home (default: neutral)")
    method: Literal["dc", "skellam"] = "dc"


def _resolve(name: str, known) -> str:
    try:
        return resolve_team(name, known)
    except SystemExit as exc:  # resolve_team refuses to guess -> 404 with hints
        raise HTTPException(status_code=404, detail=str(exc)) from None


@app.get("/teams")
def teams() -> list[str]:
    model, _, _ = _get_bundle(use_xg=False)
    return list(model.teams)


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    model, ratings, draw_params = _get_bundle(use_xg=req.use_xg)
    home = _resolve(req.home_team, model.teams)
    away = _resolve(req.away_team, model.teams)
    neutral = not req.home_advantage
    n_sims = min(req.sims, MAX_SIMS)

    pred = predict_mod.predict_match(
        model, home, away, neutral=neutral, method=req.method,
        elo_ratings=ratings, draw_params=draw_params,
    )
    sim = simulate_mod.simulate_match(model, home, away, neutral=neutral, n=n_sims)

    return {
        "home": home,
        "away": away,
        "neutral": neutral,
        "method": req.method,
        "use_xg": req.use_xg,
        "probabilities": {
            "home_win": pred["p_home_win"],
            "draw": pred["p_draw"],
            "away_win": pred["p_away_win"],
        },
        "expected_goals": {
            "home": pred["lambda_home"],
            "away": pred["lambda_away"],
        },
        "data_trust": pred["data_trust"],
        "simulation": {
            "n": n_sims,
            "avg_home_goals": sim["avg_home_goals"],
            "avg_away_goals": sim["avg_away_goals"],
            "top_scores": sim["top_scores"],
        },
    }


from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="web", html=True), name="web")


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)