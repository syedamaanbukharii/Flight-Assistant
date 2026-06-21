# Flight &amp; Travel Assistant

A conversational, AI-powered flight, hotel, and coupon assistant. Ask it
something in plain English — *"Find me an Emirates business-class flight from
Delhi to Dubai next Friday under ₹80,000"* — and a native Python planner agent
extracts the intent and parameters, calls the right tools (flight search, hotel
search, coupon lookup, preference memory, ranking), applies the best available
coupons, ranks the results **deterministically**, and replies in natural
language with booking links.

The LLM is used only to **plan** and to **write prose grounded in real tool
output** — never to invent flights, prices, or availability. Every numeric
result comes from a tool, and ranking is pure, explainable arithmetic.

---

## Highlights

- **Native tool-calling planner** — no LangChain / LangGraph / CrewAI / AutoGen.
  Just Python, a typed planner decision, and a small tool executor.
- **Graceful degradation** — runs with *no API keys at all*. Without a Groq key
  the planner falls back to a deterministic heuristic; without a RapidAPI key
  search endpoints return a clean `503 not_configured` while everything else
  keeps working.
- **Deterministic, explainable ranking** — weighted feature scoring with a
  human-readable rationale per result. No LLM in the ranking path.
- **Swappable providers** — flight and hotel back-ends sit behind abstract
  interfaces; the Sky Scrapper (RapidAPI) implementation can be replaced without
  touching the agent or API layers.
- **SQLite by default, PostgreSQL by URL change only** — SQLAlchemy 2.0 ORM.
- **Coupon engine** — seeded catalogue plus optional scraping of public coupon
  pages (httpx + BeautifulSoup, Playwright only when a source needs JS), with an
  APScheduler refresh job. Coupon parsing never uses the LLM.
- **Production hygiene** — structured JSON logging with request IDs, typed
  errors mapped to clean JSON responses, Pydantic v2 schemas everywhere, no
  secrets in code.

---

## Architecture

```
                      ┌──────────────────────────────────────────────┐
   user message  ───▶ │  PlannerAgent.handle()                       │
                      │   1. load session memory + preferences        │
                      │   2. plan  → LLM (JSON) or heuristic fallback │
                      │   3. run tools via ToolExecutor               │
                      │   4. rank results (deterministic)             │
                      │   5. synthesize reply (LLM prose or template) │
                      └───────┬───────────────────────┬──────────────┘
                              │                       │
              ┌───────────────▼─────┐      ┌──────────▼───────────┐
              │ ToolExecutor        │      │ MemoryStore          │
              │  • FlightService    │      │  (prefs + messages,  │
              │  • HotelService     │      │   SQLite-persisted)  │
              │  • CouponService    │      └──────────────────────┘
              │  • RankingService   │
              └───────┬─────────────┘
                      │
        ┌─────────────▼──────────────┐
        │ Providers (swappable)      │   FlightProvider / HotelProvider ABCs
        │  • SkyScrapper (RapidAPI)  │   + Unconfigured* fallbacks (raise 503)
        └────────────────────────────┘
```

**Layering:** `api` (FastAPI routes) → `agents` (planner, tool executor, memory)
→ `services` (LLM, flight, hotel, coupon, ranking) → `models` (Pydantic schemas
+ SQLAlchemy ORM) → `utils` (logging, errors, helpers, constants). The whole
object graph is assembled once at startup into an `AppContainer`
(`app/startup.py`) and exposed via `app.state.container`.

### Project structure

```
flight-assistant/
├── backend/
│   ├── app/            # FastAPI app, settings, lifespan/container, /verify
│   ├── agents/         # planner, tool_executor, memory
│   ├── services/       # llm, flight, hotel, coupon, ranking
│   ├── models/         # Pydantic schemas + SQLAlchemy models/Database
│   ├── utils/          # logger, errors, helpers, constants
│   ├── api/            # API routes
│   ├── data/           # coupons.seed.json (+ runtime SQLite db)
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py
│   └── Dockerfile
├── scripts/
│   └── refresh_coupons.py   # standalone coupon refresh
├── tests/                   # pytest suite (helpers, ranking, coupons, planner, api)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml           # pytest + ruff config
├── Makefile
├── docker-compose.yml
├── .env.example
└── README.md
```

> **Import root.** Application code under `backend/` uses absolute imports with
> **no `backend.` prefix** (e.g. `from app.config import Settings`). Run the app
> with `backend/` on the import path — either `uvicorn ... --app-dir backend`
> (as in the `Makefile` and Docker image) or by setting `PYTHONPATH=backend`.
> `pyproject.toml` already adds `backend` to `pythonpath` for pytest.

---

## Prerequisites

- **Python 3.12+**
- (Optional) a [Groq](https://console.groq.com) API key for LLM planning and
  natural-language replies.
- (Optional) a [RapidAPI Sky Scrapper](https://rapidapi.com/apiheya/api/sky-scrapper)
  key for live flight/hotel search.
- (Optional) Docker + Docker Compose to run the full stack in containers.

Everything runs without the optional keys; see
[Running without API keys](#running-without-api-keys).

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (runtime + dev/test)
pip install -r requirements-dev.txt
#   or, for runtime only:  pip install -r requirements.txt

# 3. Create your environment file and (optionally) add keys
cp .env.example .env
```

Open `.env` and fill in `GROQ_API_KEY` and/or `RAPIDAPI_KEY` if you have them.
All variables are documented inline in `.env.example`.

A `Makefile` wraps the common commands: `make help` lists them
(`install-dev`, `run`, `frontend`, `test`, `lint`, `format`, `coupons`,
`docker-up`, `docker-down`, `clean`).

---

## Running

### Backend (FastAPI)

From the **repository root**:

```bash
uvicorn app.main:app --reload --app-dir backend
# or:
make run
```

The API is then available at `http://localhost:8000` with interactive docs at
`http://localhost:8000/docs`.

On startup the app creates the database tables and loads the coupon seed file.

### Frontend (Streamlit)

In a second terminal (with the backend running):

```bash
streamlit run frontend/streamlit_app.py
# or:
make frontend
```

The UI is served at `http://localhost:8501`. It talks to the backend at
`BACKEND_URL` (default `http://localhost:8000`).

---

## API

| Method | Path              | Description                                                        |
| ------ | ----------------- | ------------------------------------------------------------------ |
| `GET`  | `/health`         | Liveness probe.                                                    |
| `GET`  | `/verify`         | Configuration/dependency report (`ok` / `warn` / `fail`).          |
| `POST` | `/chat`           | Conversational endpoint — the main entry point.                    |
| `POST` | `/search/flights` | Direct, structured flight search (requires a RapidAPI key).        |
| `POST` | `/search/hotels`  | Direct, structured hotel search (requires a RapidAPI key).         |

Every response carries an `X-Request-ID` header (echoed if you supply one), and
errors are returned as `{"error": {"code", "message"}, "request_id": ...}`.

### Example: chat

```bash
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Find me an Emirates business class flight from Delhi to Dubai next Friday under ₹80,000"}'
```

Other things you can ask:

- *"Cheapest flights from Mumbai to Singapore on 2025-03-10, refundable only."*
- *"Hotels near Marina Bay in Singapore within walking distance, rating 4+."*
- *"Apply the best coupon to those results."*  (follow-up; uses session memory)
- *"Actually, sort by fastest."*               (re-ranks the previous results)

Pass a `session_id` in the chat payload to keep preferences and prior results
across turns; one is generated for you if you omit it.

---

## Running without API keys

The system is designed to degrade gracefully:

| Missing key      | Effect                                                                                   |
| ---------------- | ---------------------------------------------------------------------------------------- |
| `GROQ_API_KEY`   | The planner uses a deterministic regex/heuristic planner and a templated reply writer.   |
| `RAPIDAPI_KEY`   | `/search/flights` and `/search/hotels` return `503 not_configured`; `/chat` still replies and notes that search is unavailable. Coupons and ranking still work. |

`GET /verify` reports `warn` for any unconfigured optional dependency and `ok`
once configured, so it's an easy operational readiness check.

---

## Coupons

- The seed catalogue lives in `backend/data/coupons.seed.json` and is loaded on
  startup.
- To scrape additional public coupon pages, set `COUPON_SOURCES` to a
  comma-separated list of URLs. Static pages are parsed with BeautifulSoup;
  set `COUPON_USE_PLAYWRIGHT=true` only if a source needs JavaScript rendering
  (and install browsers with `playwright install chromium`).
- A background `APScheduler` job refreshes sources every
  `COUPON_REFRESH_INTERVAL_MINUTES` when `ENABLE_SCHEDULER=true` and at least one
  source is configured.
- Refresh manually at any time:

  ```bash
  python scripts/refresh_coupons.py
  # or:
  make coupons
  ```

Savings are computed deterministically (percentage or flat, with `min_spend`,
`max_discount`, currency, scope, and provider rules) — never by the LLM.

---

## Testing

```bash
pytest
# or:
make test
```

The suite (50 tests) covers the pure helpers, the deterministic ranking maths,
coupon savings/selection and seed persistence, the offline planner pipeline, and
the HTTP API (including the keyless `503` and graceful-degradation paths). It
runs **fully offline** — no Groq or RapidAPI keys required — using fake
providers and the real coupon seed in a temporary SQLite database.

Lint and format with [ruff](https://docs.astral.sh/ruff/):

```bash
make lint     # ruff check .
make format   # ruff format .
```

---

## Docker

Build and run the full stack (backend + frontend):

```bash
docker compose up --build
# or:
make docker-up
```

- Backend → `http://localhost:8000`
- Frontend → `http://localhost:8501`

The compose file reads secrets from your `.env`, stores the SQLite database on a
named volume (`db-data`) so it survives restarts, and points the frontend at the
backend service over the internal network. Stop with `docker compose down` (or
`make docker-down`).

---

## Switching to PostgreSQL

No code changes are required — only the connection URL. Set `DATABASE_URL` in
`.env`:

```
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/flight_assistant
```

Install the driver (`pip install "psycopg[binary]"`) and start the app; the ORM
creates the schema on startup. SQLite-specific connection arguments are applied
only when the URL is a SQLite URL.

---

## Configuration reference

All settings are environment variables (see `.env.example` for the full list and
inline docs). The most relevant:

| Variable                  | Default                          | Purpose                                            |
| ------------------------- | -------------------------------- | -------------------------------------------------- |
| `GROQ_API_KEY`            | *(empty)*                        | Enables LLM planning + prose. Heuristic if unset.  |
| `GROQ_MODEL`              | `llama-3.3-70b-versatile`        | Groq model name.                                   |
| `RAPIDAPI_KEY`            | *(empty)*                        | Enables flight/hotel search.                       |
| `RAPIDAPI_HOST`           | `sky-scrapper.p.rapidapi.com`    | Sky Scrapper host.                                 |
| `DATABASE_URL`            | SQLite in `backend/data/`        | Any SQLAlchemy URL (e.g. PostgreSQL).              |
| `COUPON_SOURCES`          | *(empty)*                        | Comma-separated public coupon page URLs to scrape. |
| `ENABLE_SCHEDULER`        | `true`                           | Toggles the background coupon refresh job.         |
| `CORS_ALLOW_ORIGINS`      | `*`                              | Comma-separated CORS origins (restrict in prod).   |
| `MAX_RESULTS`             | `10`                             | Max results returned per search.                   |
| `BACKEND_URL`             | `http://localhost:8000`          | Used by the Streamlit frontend.                    |

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'app'`** — run from the repo root with
  `--app-dir backend` (or set `PYTHONPATH=backend`). See the import-root note
  above.
- **`/search/*` returns 503 `not_configured`** — expected without a
  `RAPIDAPI_KEY`. Add the key to `.env` and restart.
- **Replies feel templated / robotic** — that's the no-LLM fallback. Add a
  `GROQ_API_KEY` for natural-language synthesis.
- **Playwright errors when scraping** — either keep `COUPON_USE_PLAYWRIGHT=false`
  (static parsing) or run `playwright install chromium`.
- **Check readiness** — `curl http://localhost:8000/verify` shows exactly which
  dependencies are configured and reachable.

---

## Design notes &amp; possible extensions

- The LLM boundary is deliberately narrow: planning (structured JSON) and prose
  synthesis grounded in tool output. This keeps results trustworthy and the
  system usable offline.
- Ranking weights live in `services/ranking_service.py` as per-`SortBy` presets
  and are easy to tune; each result includes a rationale naming its top factors.
- Natural extensions: real round-trip/multi-city support, caching of provider
  responses, additional providers behind the existing interfaces, user accounts,
  and richer coupon sources.
# Flight-Assistant
