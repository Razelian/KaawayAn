# Dashboard ↔ events.db Sync Plan

Goal: `dashboard/dashboard.py` (Streamlit) renders the static HTML screens in
`dashboard/ui_assets/` with LIVE data from `database/events.db`, refreshing as
the pipeline writes. Implement in the order below; each phase is independently
verifiable. No BME/environment support yet (schema has no BME columns — see
Phase 4 for the reserved path).

Conventions for the implementing agent:
- Repo root is `KaawayAn/` (contains `config.py`). Run everything from there:
  `streamlit run dashboard/dashboard.py`. Never run from inside `dashboard/`.
- Read-only dashboard: it NEVER writes to `events.db`. Pipeline owns writes.
- DB helper lives in ONE place: `dashboard/db.py`. All queries go through it.
- Static screens are templates, not iframes: read each `code.html`, replace
  hardcoded demo values with placeholders, render via `st.html()` (Streamlit
  ≥ 1.32) or `components.html()`. Keep Tailwind CDN + font links as-is.

## Phase 0 — Concurrency prerequisite (do first, 5 lines)

`database.py` already sets `PRAGMA journal_mode=WAL` + `busy_timeout=5000`
(commit `database/database.py:13-14`). Verify these lines exist before
building the dashboard; without WAL, a Streamlit read during a pipeline
`commit()` raises `sqlite3.OperationalError: database is locked`.
No dashboard-side locking needed — WAL + short read-only transactions suffice.

## Phase 1 — `dashboard/db.py`: the only DB access layer

```python
"""Read-only queries for the dashboard. Open-close per call; never hold a connection."""
import sqlite3
from database.database import db_path  # single source of truth for the file location

def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # read-only intent
    con.row_factory = sqlite3.Row
    return con

def latest_events(limit: int = 100) -> list[dict]:
    with _connect() as con:
        return [dict(r) for r in con.execute(
            "SELECT event_id, node_id, predicted_drone, model_confidence, timestamp "
            "FROM detection_events ORDER BY event_id DESC LIMIT ?", (limit,))]

def event_count() -> int: ...
def drone_count_since(minutes: int) -> int: ...   # WHERE predicted_drone=1 AND timestamp > ...
def confidence_series(limit: int = 300) -> list[tuple[int, float]]: ...  # for the SVG chart
def max_event_id() -> int | None: ...             # change-detection anchor (Phase 2)
```

Notes:
- `mode=ro` makes accidental writes impossible from the dashboard process.
- `timestamp` column is TEXT holding an int (ns since boot) — see Phase 3 note.
- Wrap every call in try/except `sqlite3.OperationalError` → return last-good
  data; the DB file may briefly not exist on first ever run (0-byte `events.db`).

## Phase 2 — `dashboard/dashboard.py`: shell + live refresh (no threading)

Do NOT use threads. Streamlit's execution model (full script rerun per
interaction) fights manual threading; the correct primitive is
`st.fragment(run_every=...)` (Streamlit ≥ 1.33) for the live region:

```python
import streamlit as st
from dashboard.db import latest_events, max_event_id

st.set_page_config(page_title="KaawayAn Console", layout="wide")

@st.fragment(run_every=2)   # re-runs ONLY this function every 2 s
def live_detection_table():
    rows = latest_events(50)
    ...render rows into the detection_history template...

def main():
    page = st.sidebar.radio(...)  # Overview / Detections / Environment / About
    if page == "Detections":
        live_detection_table()
    ...
```

Why fragment over `st.rerun()` loop or threads: fragment reruns skip the
expensive parts (template file reads, static sidebar), cost one indexed
`ORDER BY event_id DESC LIMIT 50` query (~µs on this table size), and never
hold a DB handle across reruns. Polling every 2 s vs ~1 row/s pipeline rate
means at most one rerun shows no new data — acceptable, no websocket needed.

Change-detection optimization (skip re-render when idle):
- Cache `max_event_id()` in `st.session_state`; if unchanged, return early.
- This makes overnight idle dashboards cost ~1 tiny query per 2 s, zero DOM churn.

## Phase 3 — Template binding (per screen)

For each `ui_assets/<screen>/code.html`:
1. Read the file once per process (`@st.cache_resource` on a loader).
2. Replace demo values with Python format placeholders. Screens contain
   hardcoded demo rows/badges (e.g. `DEMO DATA` pill, fake node names,
   static SVG chart paths). Mark each replacement with an HTML comment so the
   next agent can find them: `<!-- LIVE: event_rows -->`.
3. Render with `st.html(rendered)` inside the fragment.

Screen-by-screen binding map:

| Screen dir | Live source | Notes |
|---|---|---|
| `monitoring_overview` | `event_count()`, `drone_count_since(60)`, latest row per node | Node online/offline badge: online = row within last 30 s (pipeline ~1 row/s/node). Header clock: render client-side, not from DB. |
| `detection_history` | `latest_events(50)` → `<tbody>` rows; `confidence_series(300)` → SVG path | Table columns map 1:1 to schema: `event_id, node_id, predicted_drone→badge, model_confidence→%, timestamp`. Timestamp is ns-since-boot TEXT: display as relative age (`now - ts` is meaningless across reboot — show window sequence + wall arrival time instead; see caveat). |
| `environmental_conditions` | RESERVED — schema has no BME columns | Render static "No environmental data yet" card. Do NOT invent columns. See Phase 4. |
| `bioacoustic_telemetry` | DESIGN.md only (no code.html yet) | Skip until a template exists. |
| `about_team` | static | Render as-is, no DB binding. |

Timestamp caveat (important): pipeline stores ESP32 monotonic ns (`t_us`,
resets on reboot) as TEXT. It is NOT wall time and NOT comparable across
reboots. For the dashboard, derive display time as "pipeline write order"
(`event_id` sequence) plus the dashboard host's own clock at read time.
Do not attempt `datetime.fromtimestamp(ts)` — it produces garbage dates.

## Phase 4 — BME / environment later (reserved, do not implement now)

When firmware + pipeline persist environment data:

1. Migration (pipeline side, NOT dashboard): add columns to
   `detection_events` — `temp_c REAL, humidity_pct REAL, pressure_hpa REAL` —
   or a new `environment_samples(node_id, temp_c, humidity_pct, pressure_hpa,
   timestamp)` table written per packet (10× the event rate; separate table
   keeps the event query fast). Dashboard reads whichever exists; probe with
   `PRAGMA table_info` at startup and degrade to the Phase-3 placeholder card
   if absent.
2. `environmental_conditions/code.html`: bind latest-per-node BME row to the
   gauge cards; bind a time-bucketed aggregate (`AVG(temp_c) GROUP BY
   strftime('%H', ...)`) to the trend chart. Raw per-packet rows are too dense
   to render — always aggregate.
3. The packet path already carries BME (`receiver.py` parses `temp/hum/press`
   per packet; `handler.py` forwards `bme_values`). Only persistence is missing.

## Phase 5 — Verify (acceptance checklist for the implementing agent)

```sh
# terminal 1: pipeline testbench writing rows (~1/s)
python staging/test_inference.py --n=2
# terminal 2: dashboard
streamlit run dashboard/dashboard.py
```

- [ ] Dashboard loads with zero pipeline running (empty table, no traceback).
- [ ] With testbench running: new rows appear within ~3 s of DB write.
- [ ] `sqlite3 database/events.db "SELECT COUNT(*) FROM detection_events;"`
      grows while dashboard is open (proves no read-lock contention).
- [ ] Overnight-idle: no error growth in Streamlit logs; fragment skips
      re-render when `max_event_id` unchanged.
- [ ] Kill pipeline mid-run: dashboard keeps showing last data + stale badge
      (no crash, no empty wipe).

## Explicit non-goals

- No dashboard writes to the DB (no ack/mute/threshold controls in v1).
- No `streamlit run` inside `main.py` — pipeline and dashboard are separate
  processes sharing only the SQLite file. Documented together in `build.md`.
- No audio playback of `staging/dumps/` windows (add only if asked).
- `pip install streamlit` is a NEW dependency — add to `build.md` §2 step 3.
