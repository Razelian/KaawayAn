"""Phase 4 tests for KaawayAn Dashboard integration.

Tested without hardware or torch.
All external contracts from LOCALIZATION/CONTEXT.md obeyed.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from config import Config
from dashboard import db as dash_db
from database.database import Database
from demo_source import build_demo_sources, harmonic_signal
from kaawayan.config import load_config
from kaawayan.fusion import poses_from_config
from kaawayan.live import LivePipeline
from kaawayan.protocol import FrameType, encode_frame
from live_adapter import ClassifierWorker, _NodeAudio, default_cfg_path, handle_frame, run_pipeline


@pytest.fixture
def site_cfg():
    cfg, cfg_hash = load_config(default_cfg_path())
    poses = poses_from_config(cfg)
    return cfg, poses, cfg_hash


def test_clean_simulated_source_accuracy(site_cfg, tmp_path):
    """Clean simulated source (2 nodes, harmonic 180 Hz, 20 dB SNR, 8 s):

    median position error <= 0.05 m.
    """
    cfg, poses, cfg_hash = site_cfg
    db_path = tmp_path / "test_clean.db"
    db = Database(override_path=db_path)

    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(cfg_hash)

    target_xy = (1.5, 3.0)
    sources = build_demo_sources(
        cfg,
        poses,
        duration_s=8.0,
        source_xy=target_xy,
        snr_db=20.0,
        f0_hz=180.0,
        seed=42,
    )

    t_sim = [1000.0]

    def fake_clock():
        t_sim[0] += 0.010
        return t_sim[0]

    try:
        run_pipeline(
            cfg,
            pipeline,
            sources,
            db,
            duration_s=None,
            clock=fake_clock,
            sleep=lambda _: None,
        )

        rows = db.connection.execute(
            "SELECT x_m, y_m, status FROM localizations WHERE status = 'ok'"
        ).fetchall()

        assert len(rows) > 0, "Expected 'ok' fixes from clean 20 dB SNR harmonic source"

        errors = [
            math.hypot(float(r[0]) - target_xy[0], float(r[1]) - target_xy[1])
            for r in rows
        ]
        med_err = float(np.median(errors))
        assert med_err <= 0.05, f"Median error {med_err:.4f} m exceeds 0.05 m bound"
    finally:
        db.close()


def test_noise_only_zero_ok_fixes(site_cfg, tmp_path):
    """Noise-only 8 s: zero 'ok' localization rows."""
    cfg, poses, cfg_hash = site_cfg
    db_path = tmp_path / "test_noise.db"
    db = Database(override_path=db_path)

    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(cfg_hash)

    sources = build_demo_sources(
        cfg,
        poses,
        duration_s=8.0,
        noise_only=True,
        seed=123,
    )

    t_sim = [2000.0]

    def fake_clock():
        t_sim[0] += 0.010
        return t_sim[0]

    try:
        run_pipeline(
            cfg,
            pipeline,
            sources,
            db,
            duration_s=None,
            clock=fake_clock,
            sleep=lambda _: None,
        )

        ok_count = db.connection.execute(
            "SELECT COUNT(*) FROM localizations WHERE status = 'ok'"
        ).fetchone()[0]

        assert ok_count == 0, f"Expected 0 'ok' rows on noise-only audio, got {ok_count}"

        # Confirm gate failure rows were written as 'none'
        none_count = db.connection.execute(
            "SELECT COUNT(*) FROM localizations WHERE status = 'none'"
        ).fetchone()[0]
        assert none_count > 0, "Expected 'none' rows recorded on gate failures"
    finally:
        db.close()


def test_classifier_window_length_and_non_blocking_queue(tmp_path):
    """The classifier window is exactly 16000 samples, and a full queue never blocks."""
    db_path = tmp_path / "test_cls.db"
    db = Database(override_path=db_path)

    lengths = []

    def stub_predict(samples: np.ndarray) -> float:
        lengths.append(len(samples))
        return 0.75

    worker = ClassifierWorker(
        predict=stub_predict,
        database=db,
        threshold=0.60,
        maxsize=2,
    )
    worker.start()

    try:
        # 1 second of 48 kHz mono -> resampled to 16 kHz
        test_audio_48k = np.random.randn(48000).astype(np.float64) * 5000.0
        worker.submit(node_id=1, t_wall=time.time(), left=test_audio_48k)

        # Allow worker thread to process
        deadline = time.time() + 2.0
        while not lengths and time.time() < deadline:
            time.sleep(0.01)

        assert lengths, "Worker did not process submitted audio"
        assert lengths[0] == 16000, f"Expected exactly 16000 samples, got {lengths[0]}"
        assert worker.last_len == 16000

        # Test queue never blocks when full
        def slow_predict(_s):
            time.sleep(0.5)
            return 0.0

        slow_worker = ClassifierWorker(
            predict=slow_predict,
            database=db,
            threshold=0.60,
            maxsize=2,
        )
        slow_worker.start()
        try:
            t0 = time.time()
            for i in range(10):
                slow_worker.submit(node_id=1, t_wall=time.time(), left=test_audio_48k)
            elapsed = time.time() - t0
            assert elapsed < 0.1, f"submit() blocked for {elapsed:.3f}s on full queue"
            assert slow_worker.dropped > 0, "Expected dropped submissions when queue full"
        finally:
            slow_worker.stop()
    finally:
        worker.stop()
        db.close()


def test_invalid_bme_status_rejects_temperature(site_cfg, tmp_path):
    """An ENV frame with bme_status != 0 does not change the temperature used for bearing."""
    cfg, poses, cfg_hash = site_cfg
    db_path = tmp_path / "test_bme.db"
    db = Database(override_path=db_path)
    pipeline = LivePipeline(cfg, poses)

    # Initial buffer temperature default is 20.0 C
    nid = 1
    buf = pipeline._bufs[nid]
    assert buf.temp_c == 20.0

    # Create dummy audio bufs
    win_n = int(round(float(cfg["window_s"]) * Config.AUDIO_FS))
    hop_n = int(round(float(cfg["hop_s"]) * Config.AUDIO_FS))
    audio_bufs = {nid: _NodeAudio(win_n, hop_n)}

    # Frame 1: bme_status = 1 (I2C NACK), reported temp = 45.0 C (4500)
    # Payload format: <hHIIB (temp_x100, rh_x100, overrun, usb_drop, bme_status)
    import struct
    env_payload_bad = struct.pack("<hHIIB", 4500, 5000, 0, 0, 1)
    frame_bad = {
        "type": FrameType.ENV,
        "node_id": nid,
        "seq": 1,
        "t_us": 1_000_000,
        "payload": env_payload_bad,
    }

    handle_frame(frame_bad, 100.0, pipeline, db, classifier=None, audio_bufs=audio_bufs)
    assert buf.temp_c == 20.0, f"Temperature was updated to {buf.temp_c} despite bme_status != 0"

    # Frame 2: bme_status = 0 (OK), reported temp = 28.5 C (2850)
    env_payload_ok = struct.pack("<hHIIB", 2850, 5000, 0, 0, 0)
    frame_ok = {
        "type": FrameType.ENV,
        "node_id": nid,
        "seq": 2,
        "t_us": 2_000_000,
        "payload": env_payload_ok,
    }

    handle_frame(frame_ok, 101.0, pipeline, db, classifier=None, audio_bufs=audio_bufs)
    assert abs(buf.temp_c - 28.5) < 1e-4, f"Expected 28.5 C, got {buf.temp_c}"

    db.close()


def test_old_schema_migration_and_dashboard_queries(tmp_path):
    """An old-schema events.db migrates cleanly, and the dashboard query layer still reads it."""
    old_db_path = tmp_path / "old_events.db"

    # 1. Create database with strictly the old pre-migration schema
    con = sqlite3.connect(old_db_path)
    con.execute(
        """
        CREATE TABLE detection_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER,
            predicted_drone INTEGER NOT NULL CHECK (predicted_drone IN (0, 1)),
            model_confidence REAL NOT NULL CHECK (model_confidence BETWEEN 0.0 AND 1.0),
            timestamp TEXT NOT NULL
        );
        """
    )
    con.execute(
        """
        CREATE TABLE localizations (
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
    con.execute(
        "INSERT INTO detection_events (node_id, predicted_drone, model_confidence, timestamp) "
        "VALUES (1, 1, 0.92, '50.0')"
    )
    con.execute(
        "INSERT INTO localizations (timestamp, x_m, y_m, crossing_deg, n_nodes, status, reason) "
        "VALUES ('50.0', 1.5, 3.0, 53.0, 2, 'ok', 'clean')"
    )
    con.commit()
    con.close()

    # 2. Open via Database class -> trigger additive migrations
    db = Database(override_path=old_db_path)

    # Verify new columns and tables exist
    det_cols = {r[1] for r in db.connection.execute("PRAGMA table_info(detection_events)").fetchall()}
    assert "t_wall" in det_cols

    loc_cols = {r[1] for r in db.connection.execute("PRAGMA table_info(localizations)").fetchall()}
    assert "t_wall" in loc_cols
    assert "drone_recent" in loc_cols
    assert "dead_nodes" in loc_cols

    tables = {r[0] for r in db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "bearings" in tables
    assert "node_telemetry" in tables

    # 3. Verify dashboard query layer reads the migrated DB
    old_target = dash_db.db_path
    dash_db.db_path = old_db_path
    try:
        events = dash_db.latest_events()
        assert len(events) == 1
        assert events[0]["event_id"] == 1
        assert events[0]["predicted_drone"] == 1

        loc = dash_db.latest_localization()
        assert loc is not None
        assert abs(loc["x_m"] - 1.5) < 1e-4
        assert loc["status"] == "ok"

        count = dash_db.event_count()
        assert count == 1
        drones = dash_db.drone_count()
        assert drones == 1
    finally:
        dash_db.db_path = old_target
        db.close()


def test_demo_mode_end_to_end_writes_temp_db(site_cfg, tmp_path):
    """Demo mode end-to-end writes rows to a temp DB."""
    cfg, poses, cfg_hash = site_cfg
    db_path = tmp_path / "test_demo_e2e.db"
    db = Database(override_path=db_path)
    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(cfg_hash)

    sources = build_demo_sources(cfg, poses, duration_s=2.0, loop=False)

    t_sim = [500.0]

    def fake_clock():
        t_sim[0] += 0.010
        return t_sim[0]

    try:
        run_pipeline(
            cfg,
            pipeline,
            sources,
            db,
            duration_s=None,
            clock=fake_clock,
            sleep=lambda _: None,
        )

        n_loc = db.connection.execute("SELECT COUNT(*) FROM localizations").fetchone()[0]
        n_brg = db.connection.execute("SELECT COUNT(*) FROM bearings").fetchone()[0]
        n_env = db.connection.execute("SELECT COUNT(*) FROM node_telemetry").fetchone()[0]

        assert n_loc > 0, "No localization rows written in demo mode"
        assert n_brg > 0, "No bearing rows written in demo mode"
        assert n_env > 0, "No telemetry rows written in demo mode"
    finally:
        db.close()
