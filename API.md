# Football Predictor HTTP API

A small FastAPI backend that wraps the Dixon-Coles model behind two endpoints.
It is **self-hosted**: the public site (GitHub Pages) is static and computes
predictions in the browser, so to use the HTTP API you run it yourself.

## Quickstart

```bash
git clone https://github.com/xackerlud31337/worldcup-predictor.git
cd worldcup-predictor
pip install -r requirements.txt
uvicorn api:app
```

The server listens on `http://127.0.0.1:8000`. On the very first start it
downloads the historical results dataset (~4 MB) and fits the model (a few
seconds); both are cached under `data_cache/`, so later starts are instant.

Interactive documentation comes for free at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

The static website is also served at `/`, so the root URL shows the SportMath
pages.

---

## `GET /teams`

Returns every team the model knows, as a JSON array of canonical names.

```bash
curl http://127.0.0.1:8000/teams
```

```json
["Afghanistan", "Albania", "Algeria", "...", "Zimbabwe"]
```

Use these names in `/predict`. Spelling does not have to be exact (see
"Team name resolution" below), but the canonical list is the safe reference.

---

## `POST /predict`

Predicts a single match. Request body (JSON):

| Field            | Type    | Default    | Meaning |
|------------------|---------|------------|---------|
| `home_team`      | string  | required   | First team. "Home" only matters if `home_advantage` is true. |
| `away_team`      | string  | required   | Second team. |
| `home_advantage` | bool    | `false`    | `true` = first team plays at home; `false` = neutral venue (World Cup default). |
| `use_xg`         | bool    | `false`    | Use team strengths fitted with StatsBomb expected-goals data blended in. **First request with `true` is slow** (~1 min, downloads event data once, then cached). |
| `method`         | string  | `"dc"`     | `"dc"` = Dixon-Coles scoreline matrix (accurate). `"skellam"` = goal-difference approximation (faster, slightly cruder). |
| `sims`           | int     | `10000`    | Monte-Carlo samples for the scoreline distribution. Capped at 100,000. |

### Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Brazil", "away_team": "Argentina", "sims": 20000}'
```

```json
{
  "home": "Brazil",
  "away": "Argentina",
  "neutral": true,
  "method": "dc",
  "use_xg": false,
  "probabilities": {
    "home_win": 0.446,
    "draw": 0.263,
    "away_win": 0.291
  },
  "expected_goals": {
    "home": 1.52,
    "away": 1.21
  },
  "data_trust": 1.0,
  "simulation": {
    "n": 20000,
    "avg_home_goals": 1.52,
    "avg_away_goals": 1.21,
    "top_scores": { "1-1": 0.124, "1-0": 0.105, "2-1": 0.094 }
  }
}
```

### Response fields

- **`probabilities`**: win / draw / win chances for the two teams. These sum
  to 1 and come from the exact scoreline matrix (not sampling), blended toward
  an Elo prior when a team has thin recent data.
- **`expected_goals`**: the model's goal rates (lambda) per team, i.e. average
  goals each side would score if the match were replayed many times. Not a
  score prediction.
- **`data_trust`**: 0 to 1. Below 1 means at least one team has little recent
  data and the prediction leans partly on the Elo prior. Treat low-trust
  outputs with extra caution.
- **`simulation.top_scores`**: most frequent scorelines across `n` Monte-Carlo
  samples, as `"home-away": probability`. Sampling noise of roughly ±0.5
  percentage points at 10k sims; raise `sims` for smoother numbers.
- **`neutral`**: echo of the venue assumption actually used.

### Team name resolution

Team names are matched case-insensitively and small typos are auto-corrected
(`"Equador"` resolves to `"Ecuador"`). Historical names map to modern ones
(`"West Germany"` -> `"Germany"`). If a name cannot be resolved safely, the
API refuses to guess and returns **404** with the closest suggestions in
`detail`:

```json
{ "detail": "Unknown team 'Brasil'. Did you mean: Brazil?" }
```

### Errors

| Status | Cause |
|--------|-------|
| 404    | Unresolvable team name (suggestions in `detail`). |
| 422    | Invalid body (missing field, wrong type, `sims` < 1, bad `method`). Standard FastAPI validation errors. |

---

## Calling it from code

Python:

```python
import requests

r = requests.post("http://127.0.0.1:8000/predict", json={
    "home_team": "France",
    "away_team": "Paraguay",
})
r.raise_for_status()
p = r.json()["probabilities"]
print(f"France {p['home_win']:.1%} | draw {p['draw']:.1%} | Paraguay {p['away_win']:.1%}")
```

JavaScript (Node or browser):

```js
const resp = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ home_team: "France", away_team: "Paraguay" }),
});
const data = await resp.json();
console.log(data.probabilities);
```

## Good to know

- **No authentication and no rate limiting.** It is meant for local or
  trusted-network use; put it behind a reverse proxy if you expose it.
- **No CORS headers.** Calling it from a browser page served on a different
  origin will be blocked by the browser. Server-to-server calls (Python,
  curl, Node) are unaffected. Open an issue if you need CORS enabled.
- **Freshness**: the model fits on whatever results dataset is in
  `data_cache/`. Delete `data_cache/` (or run `python main.py --refit`) to
  pull the latest results before starting the server.
- The prediction math is identical to the public site, with one difference:
  the site computes `top_scores` exactly from the probability matrix, while
  the API samples them (`sims`), so the site's chart is deterministic and the
  API's has small sampling noise.
