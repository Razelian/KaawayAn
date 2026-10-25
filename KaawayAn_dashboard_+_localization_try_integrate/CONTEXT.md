# KaawayAn Dashboard: Context & Contracts

## 1. Scope
- Base station runtime application: multi-node serial ingest, decoupled ML classifier worker thread, SQLite database persistence, and Streamlit visualization dashboard.
- Localization math, protocol parsing, bearing calculation, and multi-node fusion live in the root `kaawayan/` package. Dashboard and ingest code import this package; they do not duplicate protocol or fusion math.
- Binding wire protocol, coordinate system conventions, and fusion contracts are defined by `kaawayan/protocol.py` and `kaawayan/fusion.py`.

## 2. Database Contract (`database/events.db`)
The database uses SQLite with WAL mode (`PRAGMA journal_mode=WAL;`) and `PRAGMA busy_timeout=5000;`. The dashboard process connects in read-only mode (`file:events.db?mode=ro`).

Additive migrations preserve compatibility with historical databases. Tables:

- `detection_events` (Classifier only):
  - `event_id` (INTEGER PRIMARY KEY AUTOINCREMENT)
  - `node_id` (INTEGER)
  - `predicted_drone` (INTEGER CHECK IN (0, 1))
  - `model_confidence` (REAL CHECK BETWEEN 0.0 AND 1.0)
  - `timestamp` (TEXT)
  - `t_wall` (REAL epoch seconds)
- `localizations`:
  - `loc_id` (INTEGER PRIMARY KEY AUTOINCREMENT)
  - `t_wall` (REAL epoch seconds)
  - `x_m` (REAL, nullable when status != "ok")
  - `y_m` (REAL, nullable when status != "ok")
  - `crossing_deg` (REAL, nullable)
  - `n_nodes` (INTEGER)
  - `status` (TEXT: "ok" | "uncertain" | "degraded" | "none")
  - `reason` (TEXT)
  - `drone_recent` (INTEGER 0/1)
  - `dead_nodes` (TEXT, JSON array of node IDs)
- `bearings`:
  - `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
  - `t_wall` (REAL epoch seconds)
  - `node_id` (INTEGER)
  - `window_idx` (INTEGER)
  - `theta_deg` (REAL)
  - `confidence` (REAL)
  - `peak_median_ratio` (REAL)
  - `gate_ok` (INTEGER 0/1)
- `node_telemetry`:
  - `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
  - `node_id` (INTEGER)
  - `timestamp` (TEXT)
  - `temp_c` (REAL)
  - `humidity_pct` (REAL)
  - `overrun_count` (INTEGER)
  - `usb_drop_count` (INTEGER)
  - `bme_status` (INTEGER: 0 ok, 1 NACK, 2 wrong chip ID, 3 timeout)
  - `t_wall` (REAL epoch seconds)

### Environmental Rule
Only ENV frames with `bme_status == 0` may update the speed-of-sound temperature used for acoustic bearing calculation.

### Gate Failure Rule
When no node bearing in a paired window group passes gating thresholds, a localization row is recorded with status `"none"` so that the dashboard clears stale fixes immediately.

## 3. Installation and first run

Install from the repository root with `python -m pip install -e ".[live,dash,test]"`.
The `database/` directory contains no required populated database: opening
`Database()` creates the selected SQLite file and applies additive migrations.
The dashboard opens its selected database read-only and displays the empty
state until ingest creates it.

`config/site.toml` contains physical survey placeholders. Demo mode may load
them to exercise the simulation, but live startup must refuse to open serial
ports until every configured USB serial identity has been surveyed. Coordinates,
headings, microphone spacing, and `tau0_s` must never be replaced with invented
deployment measurements.

## 4. Acceptance boundary

The acceptance simulation emits a report explicitly labeled `SIMULATED`; it
does not establish hardware readiness. Physical serial capture, ramp verification,
timing, channel order, BME health, calibration, localization, and outdoor
classifier validation remain `UNTESTED ON HARDWARE` until a real-device
acceptance report is generated and reviewed.
