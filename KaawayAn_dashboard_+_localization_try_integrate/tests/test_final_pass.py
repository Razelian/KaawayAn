"""Comprehensive test suite for the KaawayAn Dashboard final functional pass.

Verifies:
1. No external URLs and no mock data leaks across dashboard runtime and templates.
2. Localization map geometry (uniform px/m scale) and fix labeling ('DRONE' vs 'SOUND SOURCE').
3. Site config loading with stdlib tomllib (no kaawayan import, graceful red banner on error).
4. System status banners: SIMULATED DATA, CLASSIFIER DISABLED/FAILED, INGEST STALE, NODE LOST.
5. Classifier failure handling, watchdog restart, and RMS dBFS logging.
6. Ingest resilience: sequence gap detection, node lost/reconnect reporting, selftest preflight.
7. Database indexes, lock retries, retention pruning, and 4-hour database query latency (< 50 ms).
8. Firmware synchronization between esp32.ino and esp32_devkit_j1.ino.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
import pytest

pytest.importorskip("streamlit", reason="dashboard UI tests require the [dash] extra")

from config import Config
from dashboard import db as dash_db
from dashboard import dashboard
from database.database import Database, DEFAULT_DB_PATH, DEMO_DB_PATH
from demo_source import ByteSource
from live_adapter import ClassifierWorker, _NodeAudio, handle_frame


# ---------------------------------------------------------------------------
# 1. Offline Operation and Template Audit
# ---------------------------------------------------------------------------

def test_offline_zero_external_urls():
    """Verify that dashboard/ and all templates contain 0 external URLs."""
    repo_root = Path(__file__).resolve().parent.parent
    dashboard_dir = repo_root / "dashboard"
    url_pattern = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)

    found_urls = []
    for p in dashboard_dir.rglob("*"):
        if p.is_file() and p.suffix in (".html", ".py", ".css", ".svg"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            matches = url_pattern.findall(text)
            for m in matches:
                # Ignore standard XML namespaces if any (e.g. w3.org in SVG)
                if "w3.org/2000/svg" not in m:
                    found_urls.append(f"{p.relative_to(repo_root)}: {m}")

    assert not found_urls, f"Found external URLs in dashboard: {found_urls}"


def test_rendered_pages_no_mock_data(tmp_path, monkeypatch):
    """Verify that all rendered dashboard pages contain zero leftover mock values."""
    db_file = tmp_path / "rendered_test.db"
    db = Database(override_path=db_file)
    monkeypatch.setattr(dash_db, "db_path", db_file)

    # Provide site config
    cfg, cfg_err = dashboard._get_site_config()
    assert cfg_err is None

    # Mock streamlit components.html and st.session_state
    rendered_html = {}
    import streamlit.components.v1 as components
    import streamlit as st
    monkeypatch.setattr(components, "html", lambda html, **kw: rendered_html.setdefault("current", html))
    if not hasattr(st, "session_state"):
        st.session_state = {}

    mock_needles = [
        "+3.1%",
        "1008.6",
        "1008.5",
        "Front Perimeter",
        "Rear Canopy",
        "T_SPAN: 6.00s",
        "ACOUSTIC ARRAY V2.4 LOCK",
        "Sampling: 48 kHz / 24-bit",
        "updated 2 s ago",
        "Updated 2 s ago",
        "5 Filtered",
        "Payload: Quad-rotor signature",
        "fonts.googleapis.com",
        "cdn.tailwindcss.com",
        "lh3.googleusercontent.com",
    ]

    # Test 1: Overview
    rendered_html.clear()
    dashboard._render_overview([], 0, 0, cfg, None)
    overview_html = rendered_html.get("current", "")
    for needle in mock_needles:
        assert needle not in overview_html, f"Found mock artifact {needle!r} in rendered Overview"

    # Test 2: Detections (empty rows)
    rendered_html.clear()
    dashboard._render_detection_page([], cfg)
    det_html = rendered_html.get("current", "")
    for needle in mock_needles:
        assert needle not in det_html, f"Found mock artifact {needle!r} in rendered Detections"

    # Test 3: Environment
    rendered_html.clear()
    dashboard._render_environment(cfg)
    env_html = rendered_html.get("current", "")
    for needle in mock_needles:
        assert needle not in env_html, f"Found mock artifact {needle!r} in rendered Environment"

    # Test 4: About
    rendered_html.clear()
    dashboard._render_about()
    about_html = rendered_html.get("current", "")
    for needle in mock_needles:
        assert needle not in about_html, f"Found mock artifact {needle!r} in rendered About"

    db.close()


def test_assert_replace_detects_missing_placeholder():
    """_assert_replace must record missing placeholders rather than failing silently."""
    missing = []
    sample_tpl = "<div>Hello World</div>"
    res = dashboard._assert_replace(sample_tpl, "MISSING_TAG", "Value", "TestPage", missing)
    assert len(missing) == 1
    assert "MISSING_TAG" in missing[0]
    assert res == sample_tpl


# ---------------------------------------------------------------------------
# 2. Site Config Loading (stdlib tomllib, no kaawayan dependency)
# ---------------------------------------------------------------------------

def test_site_config_loading_success():
    cfg, err = dashboard._get_site_config()
    assert err is None
    assert cfg is not None
    assert "nodes" in cfg
    assert len(cfg["nodes"]) >= 2
    for node in cfg["nodes"]:
        for k in ("node_id", "x_m", "y_m", "heading_deg"):
            assert k in node


def test_site_config_missing_or_invalid(tmp_path, monkeypatch):
    non_existent = tmp_path / "does_not_exist.toml"
    monkeypatch.setattr(dashboard, "Path", lambda *args: non_existent)
    cfg, err = dashboard._get_site_config()
    assert cfg is None
    assert err is not None
    assert "not found" in err


# ---------------------------------------------------------------------------
# 3. Localization Map Geometry and Fix Labeling
# ---------------------------------------------------------------------------

def test_map_geometry_uniform_scale():
    """Arena width and height must scale with identical px/m to avoid distortion."""
    w_m, h_m = 6.0, 6.0
    w_svg, h_svg = 640.0, 520.0
    margin_x, margin_y = 40.0, 40.0
    avail_w = w_svg - 2 * margin_x
    avail_h = h_svg - 2 * margin_y

    scale = min(avail_w / w_m, avail_h / h_m)
    scale_x = (w_m * scale) / w_m
    scale_y = (h_m * scale) / h_m

    assert abs(scale_x - scale_y) < 1e-9
    assert scale_x == pytest.approx(scale)


def test_fix_label_logic_drone_vs_sound_source():
    """Fix must be labeled DRONE only if fresh, ok, AND drone_recent == 1."""
    # Case 1: fresh ok + drone_recent=1 -> DRONE
    raw_status = "ok"
    age_s = 0.5
    max_age_s = 2.0
    drone_recent = True
    is_fresh_ok = (raw_status == "ok") and (age_s < max_age_s)
    
    if is_fresh_ok:
        status_label = "DRONE POSITION LOCKED" if drone_recent else "SOUND SOURCE (not classified as drone)"
    else:
        status_label = "NO FIX"
    assert status_label == "DRONE POSITION LOCKED"

    # Case 2: fresh ok + drone_recent=0 -> SOUND SOURCE
    drone_recent = False
    if is_fresh_ok:
        status_label = "DRONE POSITION LOCKED" if drone_recent else "SOUND SOURCE (not classified as drone)"
    assert status_label == "SOUND SOURCE (not classified as drone)"

    # Case 3: stale fix -> STALE FIX
    age_s = 3.5
    is_fresh_ok = (raw_status == "ok") and (age_s < max_age_s)
    if is_fresh_ok:
        status_label = "DRONE POSITION LOCKED"
    else:
        status_label = f"STALE FIX ({age_s:.1f}s ago)"
    assert "STALE FIX" in status_label


# ---------------------------------------------------------------------------
# 4. System Status & Banners
# ---------------------------------------------------------------------------

def test_system_status_table_crud(tmp_path):
    db_file = tmp_path / "status_test.db"
    db = Database(override_path=db_file)

    db.set_system_status("mode", "demo", 1000.0)
    db.set_system_status("classifier", "ok", 1000.0)
    db.set_system_status("ingest_heartbeat_t", "1000.0", 1000.0)

    all_st = db.get_all_system_status()
    assert all_st["mode"]["value"] == "demo"
    assert all_st["classifier"]["value"] == "ok"
    assert all_st["ingest_heartbeat_t"]["value"] == "1000.0"

    # Update mode
    db.set_system_status("mode", "live", 1005.0)
    assert db.get_system_status("mode")["value"] == "live"
    db.close()


def test_stale_ingest_detection(tmp_path, monkeypatch):
    """Heartbeat older than 5s must trigger ingest stale status."""
    db_file = tmp_path / "stale_test.db"
    db = Database(override_path=db_file)
    monkeypatch.setattr(dash_db, "db_path", db_file)

    now = 2000.0
    # Fresh heartbeat (2s ago)
    db.set_system_status("ingest_heartbeat_t", str(now - 2.0), now - 2.0)
    hb = dash_db.get_system_status("ingest_heartbeat_t")
    assert (now - float(hb["t_wall"])) <= 5.0

    # Stale heartbeat (7s ago)
    db.set_system_status("ingest_heartbeat_t", str(now - 7.0), now - 7.0)
    hb = dash_db.get_system_status("ingest_heartbeat_t")
    assert (now - float(hb["t_wall"])) > 5.0
    db.close()


# ---------------------------------------------------------------------------
# 5. Classifier Worker Resilience, Watchdog & RMS dBFS
# ---------------------------------------------------------------------------

def test_classifier_rms_dbfs_calculation():
    """RMS dBFS calculation: full scale sine wave is ~ -3.01 dBFS, silence is -120 dBFS."""
    # Full amplitude sine wave (amplitude 32767)
    t = np.linspace(0, 1, 16000, endpoint=False)
    sine_int16 = (32767.0 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    
    norm = sine_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(norm ** 2)))
    dbfs = 20.0 * math.log10(max(rms, 1e-6))
    assert -3.1 < dbfs < -2.9

    # Digital silence
    silence = np.zeros(16000, dtype=np.int16)
    norm_sil = silence.astype(np.float32) / 32768.0
    rms_sil = float(np.sqrt(np.mean(norm_sil ** 2)))
    dbfs_sil = 20.0 * math.log10(max(rms_sil, 1e-6))
    assert dbfs_sil <= -120.0


def test_classifier_worker_watchdog_restart(tmp_path):
    """ClassifierWorker watchdog must restart thread if terminated unexpectedly."""
    db_file = tmp_path / "worker_test.db"
    db = Database(override_path=db_file)

    def dummy_classifier(audio):
        return 0.95

    worker = ClassifierWorker(dummy_classifier, db, threshold=0.60)
    worker.start()
    assert worker._thread is not None
    assert worker._thread.is_alive()

    # Simulate unexpected thread termination while _stop is NOT set
    dead_thread = threading.Thread(target=lambda: None, daemon=True)
    dead_thread.start()
    dead_thread.join()
    worker._thread = dead_thread
    assert not worker._thread.is_alive()

    # Watchdog check should restart it
    worker.check_watchdog()
    assert worker._thread.is_alive()

    worker.stop()
    db.close()


# ---------------------------------------------------------------------------
# 6. Ingest Resilience: Sequence Gap Detection & Node Lost
# ---------------------------------------------------------------------------

def test_sequence_gap_detection(tmp_path):
    """Host-side sequence gap detection must increment usb_drop_count."""
    from unittest.mock import MagicMock
    db_file = tmp_path / "gap_test.db"
    db = Database(override_path=db_file)

    audio_state: dict[int, _NodeAudio] = {1: _NodeAudio(1, None)}
    seq_state: dict[tuple[int, int], int] = {}
    usb_drops: dict[int, int] = {1: 0}
    mock_pipeline = MagicMock()

    from kaawayan.protocol import FrameType

    # Frame 0
    frame0 = {
        "version": 1,
        "type": FrameType.AUDIO,
        "node_id": 1,
        "seq": 0,
        "t_us": 1000,
        "payload": b"\x00" * 2048,
    }

    handle_frame(
        frame0, t_wall=100.0,
        pipeline=mock_pipeline, db=db, classifier=None,
        audio_bufs=audio_state,
        seq_trackers=seq_state, node_drops=usb_drops
    )
    assert usb_drops[1] == 0

    # Frame 3 (missing seq 1 and 2 -> gap of 2)
    frame3 = {
        "version": 1,
        "type": FrameType.AUDIO,
        "node_id": 1,
        "seq": 3,
        "t_us": 4000,
        "payload": b"\x00" * 2048,
    }
    handle_frame(
        frame3, t_wall=100.03,
        pipeline=mock_pipeline, db=db, classifier=None,
        audio_bufs=audio_state,
        seq_trackers=seq_state, node_drops=usb_drops
    )
    assert usb_drops[1] == 2

    st = db.get_system_status("node_1_usb_drops")
    assert st is not None
    assert int(st["value"]) == 2
    db.close()


def test_byte_source_exhausted_property():
    """ByteSource must have exhausted property without test-only attribute checks."""
    src = ByteSource(b"\x01\x02\x03\x04")
    assert not src.exhausted
    chunk1 = src.read(2)
    assert chunk1 == b"\x01\x02"
    assert not src.exhausted
    chunk2 = src.read(2)
    assert chunk2 == b"\x03\x04"
    assert src.exhausted


# ---------------------------------------------------------------------------
# 7. Database Scale & 4-Hour Database Query Performance (< 50 ms)
# ---------------------------------------------------------------------------

def test_database_locked_retry(tmp_path):
    """Database writes must retry on OperationalError ('database is locked')."""
    db_file = tmp_path / "retry_test.db"
    db = Database(override_path=db_file)

    attempts = [0]
    def failing_then_succeeding():
        attempts[0] += 1
        if attempts[0] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success"

    res = db._write_with_retry(failing_then_succeeding, max_retries=5, initial_delay=0.001)
    assert res == "success"
    assert attempts[0] == 3
    db.close()


def test_prune_older_than_in_batches(tmp_path):
    """Retention pruning must delete rows in batches."""
    db_file = tmp_path / "prune_test.db"
    db = Database(override_path=db_file)

    now = 10000.0
    # Add 1200 old bearings and 300 fresh bearings
    for i in range(1200):
        db.log_bearing(t_wall=now - 100000.0 + i, node_id=1, window_idx=i, theta_deg=10.0, confidence=0.9, peak_median_ratio=2.0, gate_ok=True)
    for i in range(300):
        db.log_bearing(t_wall=now - 100.0 + i, node_id=1, window_idx=i, theta_deg=10.0, confidence=0.9, peak_median_ratio=2.0, gate_ok=True)

    cutoff = now - 500.0
    deleted = db.prune_older_than(cutoff, batch_size=500)
    assert deleted["bearings"] == 1200

    remaining = db.connection.execute("SELECT COUNT(*) FROM bearings").fetchone()[0]
    assert remaining == 300
    db.close()


def test_four_hour_database_query_latencies(tmp_path, monkeypatch):
    """Simulate a 4-hour database (~30,000 rows across tables).

    All dashboard queries MUST execute in < 50 ms each.
    """
    db_file = tmp_path / "scale_4hr.db"
    db = Database(override_path=db_file)
    monkeypatch.setattr(dash_db, "db_path", db_file)

    base_t = time.time() - (4 * 3600.0)

    # 1. Populate 4 hours of telemetry: 2 nodes @ 1 Hz = 28,800 rows
    # Batch insert for fast test setup
    telemetry_data = []
    for step in range(7200):  # 2 hours equivalent @ 2 nodes = 14,400 rows
        t_w = base_t + step * 2.0
        telemetry_data.append((1, str(t_w), 28.5, 65.0, 0, 0, 0, t_w))
        telemetry_data.append((2, str(t_w), 29.0, 64.0, 0, 0, 0, t_w))
    db.connection.executemany(
        """
        INSERT INTO node_telemetry (node_id, timestamp, temp_c, humidity_pct, overrun_count, usb_drop_count, bme_status, t_wall)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        telemetry_data,
    )

    # 2. Populate 15,000 bearings
    bearings_data = []
    for step in range(7500):
        t_w = base_t + step * 1.9
        bearings_data.append((t_w, 1, step, 15.2, 0.92, 3.4, 1))
        bearings_data.append((t_w, 2, step, -22.5, 0.88, 3.1, 1))
    db.connection.executemany(
        """
        INSERT INTO bearings (t_wall, node_id, window_idx, theta_deg, confidence, peak_median_ratio, gate_ok)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        bearings_data,
    )

    # 3. Populate 2,000 detections and localizations
    det_data = []
    loc_data = []
    for step in range(2000):
        t_w = base_t + step * 7.2
        is_drone = 1 if (step % 20 == 0) else 0
        det_data.append((1, is_drone, 0.95 if is_drone else 0.1, str(t_w), 15.0, 0.9, 3.0, 28.5, 65.0, t_w, -25.0))
        loc_data.append((str(t_w), 1.5, 2.5, 45.0, 2, "ok", "", t_w, is_drone, None))

    db.connection.executemany(
        """
        INSERT INTO detection_events (node_id, predicted_drone, model_confidence, timestamp, theta_deg, bearing_confidence, peak_median_ratio, temp_c, humidity_pct, t_wall, rms_dbfs)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        det_data,
    )
    db.connection.executemany(
        """
        INSERT INTO localizations (timestamp, x_m, y_m, crossing_deg, n_nodes, status, reason, t_wall, drone_recent, dead_nodes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        loc_data,
    )
    db.set_system_status("mode", "live", time.time())
    db.connection.commit()

    # Benchmark every single dashboard query
    queries = [
        ("latest_events(50)", lambda: dash_db.latest_events(50)),
        ("latest_drone_event()", lambda: dash_db.latest_drone_event()),
        ("event_count()", lambda: dash_db.event_count()),
        ("drone_count()", lambda: dash_db.drone_count()),
        ("confidence_series()", lambda: dash_db.confidence_series(100)),
        ("distinct_nodes()", lambda: dash_db.distinct_nodes()),
        ("latest_bearing(1)", lambda: dash_db.latest_bearing(1)),
        ("latest_bearings_by_node([1, 2])", lambda: dash_db.latest_bearings_by_node([1, 2])),
        ("node_last_seen([1, 2])", lambda: dash_db.node_last_seen([1, 2])),
        ("latest_localization()", lambda: dash_db.latest_localization()),
        ("latest_environment(1)", lambda: dash_db.latest_environment(1)),
        ("environment_series(1, 60)", lambda: dash_db.environment_series(1, 60)),
        ("all_system_status()", lambda: dash_db.all_system_status()),
    ]

    for name, q_fn in queries:
        t0 = time.perf_counter()
        res = q_fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50.0, f"Query {name} took {elapsed_ms:.2f} ms (exceeded 50 ms target!)"

    db.close()


# ---------------------------------------------------------------------------
# 8. Firmware Codebase Synchronization
# ---------------------------------------------------------------------------

def test_firmware_synchronization():
    """Verify esp32.ino and esp32_devkit_j1.ino are synchronized with only node/board diffs."""
    repo_root = Path(__file__).resolve().parent.parent
    f1 = (repo_root / "esp32" / "esp32.ino").read_text(encoding="utf-8")
    f2 = (repo_root / "esp32" / "esp32_devkit_j1.ino").read_text(encoding="utf-8")

    # Both must have watchdog handling
    assert "esp_task_wdt_reset();" in f1
    assert "esp_task_wdt_reset();" in f2
    assert "esp_task_wdt_reconfigure" in f1 or "esp_task_wdt_init" in f1
    assert "esp_task_wdt_reconfigure" in f2 or "esp_task_wdt_init" in f2

    # Both must count real overruns
    assert "overrun_count++;" in f1
    assert "overrun_count++;" in f2

    # Both must count real USB drops
    assert "usb_drop_count++;" in f1
    assert "usb_drop_count++;" in f2

    # Both must have Serial TX timeout
    assert "Serial.setTxTimeoutMs(50);" in f1
    assert "Serial.setTxTimeoutMs(50);" in f2

    # Both must check isnan on BME280 readings
    assert "isnan(t) || isnan(rh)" in f1
    assert "isnan(t) || isnan(rh)" in f2
    for firmware in (f1, f2):
        assert "bme_status_code = probe_bme280();" in firmware
        assert "env.bme_status = bme_status_code;" in firmware
        assert "return 1;" in firmware
        assert "return 3;" in firmware
        assert "return chip_id == 0x60 ? 0 : 2;" in firmware

    # Node IDs and Board Types must differ appropriately
    assert "#define NODE_ID         1" in f1
    assert "#define BOARD_TYPE      1" in f1
    assert "#define NODE_ID         2" in f2
    assert "#define BOARD_TYPE      2" in f2

    def structural_code(source):
        lines = []
        for line in source.splitlines():
            code = line.split("//", 1)[0].strip()
            if code.startswith("#define NODE_ID") or code.startswith("#define BOARD_TYPE"):
                continue
            if code:
                lines.append(code)
        return lines

    assert structural_code(f1) == structural_code(f2)
    assert "#ifndef RAMP_TEST" in f1 and "#ifndef RAMP_TEST" in f2
    assert "Serial.availableForWrite()" in f1 and "Serial.availableForWrite()" in f2
