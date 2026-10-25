import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_DB_PATH = Path(__file__).parent / "events.db"
DEMO_DB_PATH = Path(__file__).parent / "demo.db"


def get_db_path(demo: bool = False) -> Path:
    env_path = os.environ.get("KAAWAYAN_DB_PATH")
    if env_path:
        return Path(env_path)
    if demo:
        return DEMO_DB_PATH
    # In read-only / dashboard mode, prefer demo.db if demo.db is newer and active
    if DEMO_DB_PATH.exists():
        try:
            if not DEFAULT_DB_PATH.exists():
                return DEMO_DB_PATH
            if DEMO_DB_PATH.stat().st_mtime > DEFAULT_DB_PATH.stat().st_mtime:
                return DEMO_DB_PATH
        except OSError:
            pass
    return DEFAULT_DB_PATH


# Default module-level export for backward compatibility
db_path = DEFAULT_DB_PATH


def _column_names(con: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


def _add_column_if_missing(
    con: sqlite3.Connection, table: str, col: str, decl: str, existing: set[str]
) -> None:
    if col not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        existing.add(col)


class Database:
    def __init__(self, override_path: Optional[Path] = None):
        self.path = override_path or db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()

        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA busy_timeout=5000;")

        try:
            with self._lock:
                self._migrate()
                self.connection.commit()
        except Exception:
            self.connection.close()
            raise

    def _write_with_retry(self, fn, max_retries: int = 5, initial_delay: float = 0.02):
        """Execute a write function, catching sqlite3.OperationalError ('database is locked')

        and retrying with exponential backoff before raising.
        """
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2.0
                    continue
                raise

    def _migrate(self) -> None:
        con = self.connection
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER,
                predicted_drone INTEGER NOT NULL
                    CHECK (predicted_drone IN (0, 1)),
                model_confidence REAL NOT NULL
                    CHECK (model_confidence BETWEEN 0.0 AND 1.0),
                timestamp TEXT NOT NULL
            );
            """
        )
        det_cols = _column_names(con, "detection_events")
        for col, decl in (
            ("theta_deg", "REAL"),
            ("bearing_confidence", "REAL"),
            ("peak_median_ratio", "REAL"),
            ("temp_c", "REAL"),
            ("humidity_pct", "REAL"),
            ("t_wall", "REAL"),
            ("rms_dbfs", "REAL"),
        ):
            _add_column_if_missing(con, "detection_events", col, decl, det_cols)

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS localizations (
                loc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                x_m REAL,
                y_m REAL,
                crossing_deg REAL,
                n_nodes INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason TEXT
            );
            """
        )
        loc_cols = _column_names(con, "localizations")
        for col, decl in (
            ("t_wall", "REAL"),
            ("drone_recent", "INTEGER"),
            ("dead_nodes", "TEXT"),
        ):
            _add_column_if_missing(con, "localizations", col, decl, loc_cols)

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS node_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                temp_c REAL,
                humidity_pct REAL,
                overrun_count INTEGER,
                usb_drop_count INTEGER,
                bme_status INTEGER
            );
            """
        )
        tel_cols = _column_names(con, "node_telemetry")
        _add_column_if_missing(con, "node_telemetry", "t_wall", "REAL", tel_cols)

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS bearings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                t_wall REAL NOT NULL,
                node_id INTEGER NOT NULL,
                window_idx INTEGER,
                theta_deg REAL,
                confidence REAL,
                peak_median_ratio REAL,
                gate_ok INTEGER
            );
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS system_status (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                t_wall REAL NOT NULL
            );
            """
        )

        # Additive indexes required for scale and sub-50ms queries
        con.execute("CREATE INDEX IF NOT EXISTS idx_bearings_t_wall ON bearings(t_wall);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_bearings_node_id ON bearings(node_id, id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_localizations_t_wall ON localizations(t_wall);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_detection_events_t_wall ON detection_events(t_wall);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_detection_events_node_event ON detection_events(node_id, event_id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_node_telemetry_node_id ON node_telemetry(node_id, id);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_node_telemetry_t_wall ON node_telemetry(t_wall);")

    def set_system_status(self, key: str, value: str, t_wall: Optional[float] = None) -> None:
        if t_wall is None:
            t_wall = time.time()

        def _do():
            with self._lock:
                self.connection.execute(
                    """
                    INSERT INTO system_status (key, value, t_wall)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value, t_wall = excluded.t_wall
                    """,
                    (str(key), str(value), float(t_wall)),
                )
                self.connection.commit()

        self._write_with_retry(_do)

    def get_system_status(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self.connection.execute(
                "SELECT key, value, t_wall FROM system_status WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return {"key": row[0], "value": row[1], "t_wall": float(row[2])}
            return None

    def get_all_system_status(self) -> dict[str, dict]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT key, value, t_wall FROM system_status"
            ).fetchall()
            return {r[0]: {"value": r[1], "t_wall": float(r[2])} for r in rows}

    def get_nearest_bearing_theta(
        self, node_id: int, t_wall: float, max_dt: float = 1.0
    ) -> Optional[float]:
        """Find the nearest bearing for a node within max_dt seconds."""
        with self._lock:
            row = self.connection.execute(
                """
                SELECT theta_deg, abs(t_wall - ?) as dt
                FROM bearings
                WHERE node_id = ? AND theta_deg IS NOT NULL AND abs(t_wall - ?) <= ?
                ORDER BY dt ASC LIMIT 1
                """,
                (float(t_wall), int(node_id), float(t_wall), float(max_dt)),
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return None

    def prune_older_than(self, cutoff_t: float, batch_size: int = 500) -> dict[str, int]:
        """Prune rows older than cutoff_t in small batches so writes are not blocked."""
        deleted: dict[str, int] = {}
        for table, id_col in [
            ("bearings", "id"),
            ("localizations", "loc_id"),
            ("detection_events", "event_id"),
            ("node_telemetry", "id"),
        ]:
            total = 0
            while True:
                def _do_batch():
                    with self._lock:
                        cur = self.connection.execute(
                            f"DELETE FROM {table} WHERE {id_col} IN ("
                            f"SELECT {id_col} FROM {table} WHERE t_wall < ? LIMIT ?"
                            f")",
                            (float(cutoff_t), int(batch_size)),
                        )
                        self.connection.commit()
                        return cur.rowcount

                n = self._write_with_retry(_do_batch)
                total += n
                if n < batch_size:
                    break
            deleted[table] = total
        return deleted

    def log_detection(
        self,
        node_id: int,
        predicted_drone: int,
        model_confidence: float,
        timestamp: str | int | float,
        theta_deg: Optional[float] = None,
        bearing_confidence: Optional[float] = None,
        peak_median_ratio: Optional[float] = None,
        temp_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
        t_wall: Optional[float] = None,
        rms_dbfs: Optional[float] = None,
    ) -> int:
        if theta_deg is None and t_wall is not None:
            theta_deg = self.get_nearest_bearing_theta(node_id, t_wall, max_dt=1.0)

        def _do():
            with self._lock:
                cur = self.connection.execute(
                    """
                    INSERT INTO detection_events (
                        node_id, predicted_drone, model_confidence, timestamp,
                        theta_deg, bearing_confidence, peak_median_ratio,
                        temp_c, humidity_pct, t_wall, rms_dbfs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        int(predicted_drone),
                        float(model_confidence),
                        str(timestamp),
                        float(theta_deg) if theta_deg is not None else None,
                        float(bearing_confidence) if bearing_confidence is not None else None,
                        float(peak_median_ratio) if peak_median_ratio is not None else None,
                        float(temp_c) if temp_c is not None else None,
                        float(humidity_pct) if humidity_pct is not None else None,
                        float(t_wall) if t_wall is not None else None,
                        float(rms_dbfs) if rms_dbfs is not None else None,
                    ),
                )
                self.connection.commit()
                return cur.lastrowid

        return self._write_with_retry(_do)

    def log_classifier_prediction(
        self,
        node_id: int,
        predicted_drone: int,
        model_confidence: float,
        timestamp: int,
    ) -> None:
        self.log_detection(
            node_id=node_id,
            predicted_drone=predicted_drone,
            model_confidence=model_confidence,
            timestamp=str(timestamp),
            t_wall=float(timestamp) if timestamp else None,
        )

    def log_localization(
        self,
        timestamp: str | int | float,
        x_m: Optional[float],
        y_m: Optional[float],
        crossing_deg: Optional[float],
        n_nodes: int,
        status: str,
        reason: str = "",
        t_wall: Optional[float] = None,
        drone_recent: Optional[bool] = None,
        dead_nodes: Optional[Any] = None,
    ) -> int:
        dead_json = None
        if dead_nodes is not None:
            dead_json = json.dumps(dead_nodes) if not isinstance(dead_nodes, str) else dead_nodes

        def _do():
            with self._lock:
                cur = self.connection.execute(
                    """
                    INSERT INTO localizations (
                        timestamp, x_m, y_m, crossing_deg, n_nodes, status, reason,
                        t_wall, drone_recent, dead_nodes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(timestamp),
                        float(x_m) if x_m is not None else None,
                        float(y_m) if y_m is not None else None,
                        float(crossing_deg) if crossing_deg is not None else None,
                        n_nodes,
                        status,
                        reason,
                        float(t_wall) if t_wall is not None else None,
                        (1 if drone_recent else 0) if drone_recent is not None else None,
                        dead_json,
                    ),
                )
                self.connection.commit()
                return cur.lastrowid

        return self._write_with_retry(_do)

    def log_env(
        self,
        node_id: int,
        timestamp: str | int | float,
        temp_c: float,
        humidity_pct: float,
        overrun_count: int = 0,
        usb_drop_count: int = 0,
        bme_status: int = 0,
        t_wall: Optional[float] = None,
    ) -> None:
        def _do():
            with self._lock:
                self.connection.execute(
                    """
                    INSERT INTO node_telemetry (
                        node_id, timestamp, temp_c, humidity_pct,
                        overrun_count, usb_drop_count, bme_status, t_wall
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        str(timestamp),
                        float(temp_c),
                        float(humidity_pct),
                        overrun_count,
                        usb_drop_count,
                        bme_status,
                        float(t_wall) if t_wall is not None else None,
                    ),
                )
                self.connection.commit()

        self._write_with_retry(_do)

    def log_bearing(
        self,
        t_wall: float,
        node_id: int,
        window_idx: Optional[int],
        theta_deg: Optional[float],
        confidence: Optional[float],
        peak_median_ratio: Optional[float],
        gate_ok: Optional[bool],
    ) -> None:
        def _do():
            with self._lock:
                self.connection.execute(
                    """
                    INSERT INTO bearings (
                        t_wall, node_id, window_idx, theta_deg, confidence,
                        peak_median_ratio, gate_ok
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        float(t_wall),
                        int(node_id),
                        window_idx,
                        theta_deg,
                        confidence,
                        peak_median_ratio,
                        (1 if gate_ok else 0) if gate_ok is not None else None,
                    ),
                )
                self.connection.commit()

        self._write_with_retry(_do)

    def close(self):
        with self._lock:
            self.connection.close()
