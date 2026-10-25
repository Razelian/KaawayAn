"""Read-only queries for the dashboard. Open-close per call; never hold a connection."""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from database.database import DEFAULT_DB_PATH, get_db_path

# Module-level variable for test overrides / backward compatibility
db_path = DEFAULT_DB_PATH


def _get_active_path() -> Path:
    if db_path != DEFAULT_DB_PATH:
        return Path(db_path)
    return get_db_path()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    p = _get_active_path()
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)  # read-only intent
    con.execute("PRAGMA busy_timeout=5000;")
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


def _safe(fn, fallback):
    """Run fn(); on missing/locked DB return fallback instead of raising."""
    try:
        return fn()
    except (sqlite3.Error, OSError):
        return fallback


def latest_events(limit: int = 100) -> list[dict]:
    """Retrieve latest detection events, resolving theta_deg from nearby bearings if null."""
    def _q():
        with _connect() as con:
            cur = con.execute(
                """
                SELECT de.event_id, de.node_id, de.predicted_drone, de.model_confidence, de.timestamp,
                       de.t_wall, de.temp_c, de.humidity_pct,
                       COALESCE(
                           de.theta_deg,
                           (SELECT b.theta_deg FROM bearings b
                            WHERE b.node_id = de.node_id AND abs(b.t_wall - de.t_wall) <= 1.0
                            ORDER BY b.id DESC LIMIT 1)
                       ) AS theta_deg,
                       COALESCE(
                           de.bearing_confidence,
                           (SELECT b.confidence FROM bearings b
                            WHERE b.node_id = de.node_id AND abs(b.t_wall - de.t_wall) <= 1.0
                            ORDER BY b.id DESC LIMIT 1)
                       ) AS bearing_confidence,
                       COALESCE(
                           de.peak_median_ratio,
                           (SELECT b.peak_median_ratio FROM bearings b
                            WHERE b.node_id = de.node_id AND abs(b.t_wall - de.t_wall) <= 1.0
                            ORDER BY b.id DESC LIMIT 1)
                       ) AS peak_median_ratio
                FROM detection_events de
                ORDER BY de.event_id DESC LIMIT ?
                """, (limit,))
            return [dict(r) for r in cur]
    return _safe(_q, [])


def latest_drone_event() -> Optional[dict]:
    """Get the latest event where predicted_drone = 1."""
    def _q():
        with _connect() as con:
            row = con.execute(
                """
                SELECT de.event_id, de.node_id, de.predicted_drone, de.model_confidence, de.timestamp,
                       de.t_wall, de.temp_c, de.humidity_pct,
                       COALESCE(
                           de.theta_deg,
                           (SELECT b.theta_deg FROM bearings b
                            WHERE b.node_id = de.node_id AND abs(b.t_wall - de.t_wall) <= 1.0
                            ORDER BY b.id DESC LIMIT 1)
                       ) AS theta_deg
                FROM detection_events de
                WHERE de.predicted_drone = 1
                ORDER BY de.event_id DESC LIMIT 1
                """
            ).fetchone()
            return dict(row) if row else None
    return _safe(_q, None)


def event_count() -> int:
    def _q():
        with _connect() as con:
            row = con.execute("SELECT COUNT(*) FROM detection_events").fetchone()
            return int(row[0]) if row else 0
    return _safe(_q, 0)


def drone_count() -> int:
    def _q():
        with _connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM detection_events WHERE predicted_drone = 1").fetchone()
            return int(row[0]) if row else 0
    return _safe(_q, 0)


def confidence_series(limit: int = 300) -> list[tuple[int, float]]:
    """(event_id, confidence) oldest-first for the SVG chart."""
    def _q():
        with _connect() as con:
            rows = con.execute(
                "SELECT event_id, model_confidence FROM detection_events "
                "ORDER BY event_id DESC LIMIT ?", (limit,)).fetchall()
            return [(r[0], float(r[1])) for r in reversed(rows)]
    return _safe(_q, [])


def distinct_nodes() -> list[int]:
    def _q():
        with _connect() as con:
            nodes = set()
            for tbl, col in [("bearings", "node_id"), ("node_telemetry", "node_id"), ("detection_events", "node_id")]:
                try:
                    for r in con.execute(f"SELECT DISTINCT {col} FROM {tbl} LIMIT 20"):
                        if r[0] is not None:
                            nodes.add(int(r[0]))
                except sqlite3.OperationalError:
                    pass
            return sorted(nodes)
    return _safe(_q, [])


def max_event_id() -> int | None:
    """Change-detection anchor: skip re-render when unchanged."""
    def _q():
        with _connect() as con:
            row = con.execute("SELECT MAX(event_id) FROM detection_events").fetchone()
            return int(row[0]) if row and row[0] is not None else None
    return _safe(_q, None)


def db_write_time() -> float | None:
    """Host mtime of active database file."""
    try:
        return _get_active_path().stat().st_mtime
    except OSError:
        return None


def seconds_since_write(now: float | None = None) -> float | None:
    mtime = db_write_time()
    if mtime is None:
        return None
    return max(0.0, (now if now is not None else time.time()) - mtime)


def latest_bearing(node_id: Optional[int] = None) -> Optional[dict]:
    """Get the latest bearing record with a non-null theta_deg."""
    def _q():
        with _connect() as con:
            try:
                if node_id is not None:
                    row = con.execute(
                        """
                        SELECT id, t_wall, node_id, window_idx, theta_deg, confidence,
                               peak_median_ratio, gate_ok
                        FROM bearings
                        WHERE node_id = ? AND theta_deg IS NOT NULL
                        ORDER BY id DESC LIMIT 1
                        """, (node_id,)).fetchone()
                else:
                    row = con.execute(
                        """
                        SELECT id, t_wall, node_id, window_idx, theta_deg, confidence,
                               peak_median_ratio, gate_ok
                        FROM bearings
                        WHERE theta_deg IS NOT NULL
                        ORDER BY id DESC LIMIT 1
                        """).fetchone()
                if row:
                    d = dict(row)
                    d["bearing_confidence"] = d.get("confidence")
                    return d
            except sqlite3.OperationalError:
                pass

            if node_id is not None:
                row = con.execute(
                    """
                    SELECT event_id, node_id, timestamp, theta_deg, bearing_confidence,
                           peak_median_ratio, predicted_drone, model_confidence, t_wall
                    FROM detection_events
                    WHERE node_id = ? AND theta_deg IS NOT NULL
                    ORDER BY event_id DESC LIMIT 1
                    """, (node_id,)).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT event_id, node_id, timestamp, theta_deg, bearing_confidence,
                           peak_median_ratio, predicted_drone, model_confidence, t_wall
                    FROM detection_events
                    WHERE theta_deg IS NOT NULL
                    ORDER BY event_id DESC LIMIT 1
                    """).fetchone()
            return dict(row) if row else None
    return _safe(_q, None)


def latest_bearings_by_node(node_ids: Optional[list[int]] = None) -> dict[int, dict]:
    """Get latest bearing for each node using index seeks rather than full scans."""
    def _q():
        with _connect() as con:
            res: dict[int, dict] = {}
            target_nodes = set(node_ids or [])
            if not target_nodes:
                target_nodes = set(distinct_nodes() or [1, 2])

            for nid in target_nodes:
                try:
                    row = con.execute(
                        """
                        SELECT id, t_wall, node_id, window_idx, theta_deg, confidence,
                               peak_median_ratio, gate_ok
                        FROM bearings
                        WHERE node_id = ? AND theta_deg IS NOT NULL
                        ORDER BY id DESC LIMIT 1
                        """, (nid,)
                    ).fetchone()
                    if row:
                        d = dict(row)
                        d["bearing_confidence"] = d.get("confidence")
                        res[nid] = d
                        continue
                except sqlite3.OperationalError:
                    pass

                try:
                    row = con.execute(
                        """
                        SELECT event_id, node_id, timestamp, theta_deg, bearing_confidence,
                               peak_median_ratio, predicted_drone, model_confidence, t_wall
                        FROM detection_events
                        WHERE node_id = ? AND theta_deg IS NOT NULL
                        ORDER BY event_id DESC LIMIT 1
                        """, (nid,)
                    ).fetchone()
                    if row:
                        res[nid] = dict(row)
                except sqlite3.OperationalError:
                    pass
            return res
    return _safe(_q, {})


def node_last_seen(node_ids: Optional[list[int]] = None) -> dict[int, float]:
    """Latest t_wall observed per node across telemetry, bearings, and detections.

    Uses indexed seeks (ORDER BY ... DESC LIMIT 1) to avoid expensive GROUP BY scans.
    """
    def _q():
        with _connect() as con:
            seen: dict[int, float] = {}
            target_nodes = set(node_ids or [])
            if not target_nodes:
                target_nodes = set(distinct_nodes() or [1, 2])

            for nid in target_nodes:
                best_t: Optional[float] = None
                # Check bearings
                try:
                    r = con.execute(
                        "SELECT t_wall FROM bearings WHERE node_id = ? AND t_wall IS NOT NULL ORDER BY id DESC LIMIT 1",
                        (nid,)
                    ).fetchone()
                    if r and r[0] is not None:
                        best_t = max(best_t or 0.0, float(r[0]))
                except sqlite3.OperationalError:
                    pass

                # Check node_telemetry
                try:
                    r = con.execute(
                        "SELECT t_wall FROM node_telemetry WHERE node_id = ? AND t_wall IS NOT NULL ORDER BY id DESC LIMIT 1",
                        (nid,)
                    ).fetchone()
                    if r and r[0] is not None:
                        best_t = max(best_t or 0.0, float(r[0]))
                except sqlite3.OperationalError:
                    pass

                # Check detection_events
                try:
                    r = con.execute(
                        "SELECT t_wall FROM detection_events WHERE node_id = ? AND t_wall IS NOT NULL ORDER BY event_id DESC LIMIT 1",
                        (nid,)
                    ).fetchone()
                    if r and r[0] is not None:
                        best_t = max(best_t or 0.0, float(r[0]))
                except sqlite3.OperationalError:
                    pass

                if best_t is not None:
                    seen[nid] = best_t
            return seen
    return _safe(_q, {})


def latest_localization() -> Optional[dict]:
    """Get the latest 2D position fix from the localizations table."""
    def _q():
        with _connect() as con:
            row = con.execute(
                """
                SELECT *
                FROM localizations
                ORDER BY loc_id DESC LIMIT 1
                """).fetchone()
            return dict(row) if row else None
    return _safe(_q, None)


def latest_environment(node_id: Optional[int] = None) -> Optional[dict]:
    """Get latest temperature and humidity from node_telemetry with bme_status == 0 only."""
    def _q():
        with _connect() as con:
            # Query node_telemetry strictly where bme_status == 0 (healthy sensor reading)
            query = "SELECT temp_c, humidity_pct, overrun_count, usb_drop_count, t_wall FROM node_telemetry WHERE bme_status = 0"
            params: list[Any] = []
            if node_id is not None:
                query += " AND node_id = ?"
                params.append(node_id)
            query += " ORDER BY id DESC LIMIT 1"

            try:
                row = con.execute(query, params).fetchone()
                if row and row["temp_c"] is not None:
                    return dict(row)
            except sqlite3.OperationalError:
                pass
            return None
    return _safe(_q, None)


def environment_series(node_id: int, limit: int = 60) -> list[dict]:
    """Get recent valid telemetry rows for plotting."""
    def _q():
        with _connect() as con:
            rows = con.execute(
                """
                SELECT id, temp_c, humidity_pct, t_wall
                FROM node_telemetry
                WHERE node_id = ? AND bme_status = 0
                ORDER BY id DESC LIMIT ?
                """, (node_id, limit)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
    return _safe(_q, [])


def get_system_status(key: str) -> Optional[dict]:
    """Read a system_status row by key."""
    def _q():
        with _connect() as con:
            try:
                row = con.execute(
                    "SELECT key, value, t_wall FROM system_status WHERE key = ?", (key,)
                ).fetchone()
                if row:
                    return {"key": row[0], "value": row[1], "t_wall": float(row[2])}
            except sqlite3.OperationalError:
                pass
            return None
    return _safe(_q, None)


def all_system_status() -> dict[str, dict]:
    """Read all system_status rows."""
    def _q():
        with _connect() as con:
            res = {}
            try:
                rows = con.execute("SELECT key, value, t_wall FROM system_status").fetchall()
                for r in rows:
                    res[r[0]] = {"value": r[1], "t_wall": float(r[2])}
            except sqlite3.OperationalError:
                pass
            return res
    return _safe(_q, {})
