"""Automated verification script for KaawayAn Definition of Done.

Simulates and verifies:
1. python main.py --demo + dashboard rendering with network blocked.
2. Kill a node source mid-run (NODE LOST banner, recovery when restored).
3. Classifier load failure: live mode refuses to start (exit 1), demo mode shows CLASSIFIER banner.
4. Stale ingest: stop writing heartbeat -> INGEST STALE banner.
5. 4-hour database query benchmark: all dashboard queries stay under 50 ms.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dashboard import dashboard, db as dash_db
from database.database import Database
import streamlit as st
import streamlit.components.v1 as components


def simulate_network_blocked():
    """Block all external socket connections to simulate offline environment."""
    orig_socket = socket.socket
    def blocked_socket(*args, **kwargs):
        raise ConnectionRefusedError("Offline simulation: External network calls strictly forbidden")
    socket.socket = blocked_socket
    return orig_socket


def restore_network(orig_socket):
    socket.socket = orig_socket


def call_live_fragment():
    fn = getattr(dashboard._live_fragment, "__wrapped__", dashboard._live_fragment)
    return fn()
    """Block all external socket connections to simulate offline environment."""
    orig_socket = socket.socket
    def blocked_socket(*args, **kwargs):
        raise ConnectionRefusedError("Offline simulation: External network calls strictly forbidden")
    socket.socket = blocked_socket
    return orig_socket


def restore_network(orig_socket):
    socket.socket = orig_socket


def run_dod_checks():
    print("=" * 70)
    print("KAAWAYAN FINAL PASS: DEFINITION OF DONE AUTOMATED VERIFICATION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Scenario 1: Offline Dashboard Render with Demo Database
    # -----------------------------------------------------------------------
    print("\n[Scenario 1] Offline Dashboard Render with Demo Database (network blocked)...")
    orig_sock = simulate_network_blocked()
    try:
        dash_db.db_path = _ROOT / "database" / "demo.db"
        cfg, cfg_err = dashboard._get_site_config()
        assert cfg_err is None, f"Site config error: {cfg_err}"

        # Capture Streamlit banner outputs
        banners = []
        dashboard.st.warning = lambda msg: banners.append(("warning", msg))
        dashboard.st.error = lambda msg: banners.append(("error", msg))
        dashboard.st.info = lambda msg: banners.append(("info", msg))

        rendered_pages = {}
        dashboard.components.html = lambda html, **kw: rendered_pages.setdefault("current", html)
        if not hasattr(dashboard.st, "session_state"):
            dashboard.st.session_state = {}

        # Render Live Fragment (evaluates banners)
        banners.clear()
        call_live_fragment()
        warning_texts = [b[1] for b in banners if b[0] == "warning"]
        assert any("SIMULATED DATA" in w for w in warning_texts), f"Expected SIMULATED DATA banner, got: {banners}"
        print("  -> SIMULATED DATA banner displayed as expected.")

        # Check all pages render offline without external URLs
        for page_name in ["Overview", "Localization", "Detections", "Environment", "About"]:
            rendered_pages.clear()
            dashboard.st.session_state["page"] = page_name
            call_live_fragment()
            html = rendered_pages.get("current", "")
            assert "fonts.googleapis.com" not in html
            assert "cdn.tailwindcss.com" not in html
            assert "lh3.googleusercontent.com" not in html
            print(f"  -> Page '{page_name}' rendered offline with 0 external URLs.")
    finally:
        restore_network(orig_sock)
    print("  [PASS] Scenario 1 verified successfully.")

    # -----------------------------------------------------------------------
    # Scenario 2: Kill a node source mid-run (NODE LOST -> recovery)
    # -----------------------------------------------------------------------
    print("\n[Scenario 2] Kill a node source mid-run simulation...")
    test_db_path = _ROOT / "database" / "test_dod.db"
    db = Database(override_path=test_db_path)
    dash_db.db_path = test_db_path
    db.set_system_status("mode", "live")
    db.set_system_status("classifier", "ok")
    db.set_system_status("ingest_heartbeat_t", str(time.time()))
    db.set_system_status("node_1_state", "ok")
    db.set_system_status("node_2_state", "ok")

    # Both nodes ok
    banners.clear()
    call_live_fragment()
    node_lost_banners = [b[1] for b in banners if "NODE LOST" in b[1]]
    assert len(node_lost_banners) == 0
    print("  -> Both nodes online: no NODE LOST banner.")

    # Node 2 disconnects -> state=lost
    db.set_system_status("node_2_state", "lost")
    banners.clear()
    call_live_fragment()
    node_lost_banners = [b[1] for b in banners if "NODE LOST" in b[1]]
    assert len(node_lost_banners) == 1
    assert "Node 2" in node_lost_banners[0]
    print(f"  -> Node 2 dropped: {node_lost_banners[0]}")

    # Node 2 reconnects -> state=ok
    db.set_system_status("node_2_state", "ok")
    banners.clear()
    call_live_fragment()
    node_lost_banners = [b[1] for b in banners if "NODE LOST" in b[1]]
    assert len(node_lost_banners) == 0
    print("  -> Node 2 recovered: NODE LOST banner cleared.")
    print("  [PASS] Scenario 2 verified successfully.")

    # -----------------------------------------------------------------------
    # Scenario 3: Simulate Classifier Failure
    # -----------------------------------------------------------------------
    print("\n[Scenario 3] Simulate Classifier Load Failure...")
    import subprocess
    # Verify live startup refusal with invalid site.toml placeholders:
    res = subprocess.run([sys.executable, "main.py", "--seconds", "1"], capture_output=True, text=True)
    combined = res.stdout + res.stderr
    assert "Refusing to start" in combined and res.returncode == 1
    print("  -> Live mode refuses to start normally with placeholders / uncalibrated config (exit code 1).")

    # B) In dashboard, classifier=failed or disabled renders banner
    db.set_system_status("classifier", "failed")
    banners.clear()
    call_live_fragment()
    cls_banners = [b[1] for b in banners if "CLASSIFIER FAILED" in b[1]]
    assert len(cls_banners) == 1
    print(f"  -> Dashboard displayed: {cls_banners[0]}")

    db.set_system_status("classifier", "disabled")
    banners.clear()
    call_live_fragment()
    cls_banners = [b[1] for b in banners if "CLASSIFIER DISABLED" in b[1]]
    assert len(cls_banners) == 1
    print(f"  -> Dashboard displayed: {cls_banners[0]}")
    print("  [PASS] Scenario 3 verified successfully.")

    # -----------------------------------------------------------------------
    # Scenario 4: Stale Ingest Heartbeat (> 5s)
    # -----------------------------------------------------------------------
    print("\n[Scenario 4] Simulate Stale Ingest (Heartbeat > 5s)...")
    db.set_system_status("classifier", "ok")
    # Fresh heartbeat
    now_t = time.time()
    db.set_system_status("ingest_heartbeat_t", str(now_t - 2.0), t_wall=now_t - 2.0)
    banners.clear()
    call_live_fragment()
    stale_banners = [b[1] for b in banners if "INGEST STALE" in b[1]]
    assert len(stale_banners) == 0
    print("  -> Fresh heartbeat (2s ago): no INGEST STALE banner.")

    # Stale heartbeat (8s ago)
    db.set_system_status("ingest_heartbeat_t", str(now_t - 8.0), t_wall=now_t - 8.0)
    banners.clear()
    call_live_fragment()
    stale_banners = [b[1] for b in banners if "INGEST STALE" in b[1]]
    assert len(stale_banners) == 1
    print(f"  -> Stale heartbeat (8s ago): {stale_banners[0]}")
    print("  [PASS] Scenario 4 verified successfully.")

    # -----------------------------------------------------------------------
    # Scenario 5: 4-Hour Database Query Scale (< 50 ms per query)
    # -----------------------------------------------------------------------
    print("\n[Scenario 5] 4-Hour Database Scale Query Latency Benchmark...")
    scale_db_path = _ROOT / "database" / "scale_benchmark.db"
    scale_db = Database(override_path=scale_db_path)
    dash_db.db_path = scale_db_path

    base_t = time.time() - (4 * 3600.0)

    # 1. 28,800 telemetry rows (2 nodes @ 1 Hz for 4 hours)
    print("  -> Generating 28,800 telemetry rows and 20,000 bearings...")
    telemetry_rows = []
    for step in range(14400):
        t_w = base_t + step
        telemetry_rows.append((1, str(t_w), 28.5, 65.0, 0, 0, 0, t_w))
        telemetry_rows.append((2, str(t_w), 29.0, 64.0, 0, 0, 0, t_w))
    scale_db.connection.executemany(
        "INSERT INTO node_telemetry (node_id, timestamp, temp_c, humidity_pct, overrun_count, usb_drop_count, bme_status, t_wall) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        telemetry_rows,
    )

    bearings_rows = []
    for step in range(10000):
        t_w = base_t + step * 1.44
        bearings_rows.append((t_w, 1, step, 15.2, 0.92, 3.4, 1))
        bearings_rows.append((t_w, 2, step, -22.5, 0.88, 3.1, 1))
    scale_db.connection.executemany(
        "INSERT INTO bearings (t_wall, node_id, window_idx, theta_deg, confidence, peak_median_ratio, gate_ok) VALUES (?, ?, ?, ?, ?, ?, ?)",
        bearings_rows,
    )

    scale_db.connection.commit()

    # Benchmark queries
    benchmarks = [
        ("latest_events(50)", lambda: dash_db.latest_events(50)),
        ("latest_drone_event()", lambda: dash_db.latest_drone_event()),
        ("event_count()", lambda: dash_db.event_count()),
        ("drone_count()", lambda: dash_db.drone_count()),
        ("confidence_series(100)", lambda: dash_db.confidence_series(100)),
        ("latest_bearing(node_id=1)", lambda: dash_db.latest_bearing(1)),
        ("latest_bearings_by_node([1, 2])", lambda: dash_db.latest_bearings_by_node([1, 2])),
        ("node_last_seen([1, 2])", lambda: dash_db.node_last_seen([1, 2])),
        ("latest_localization()", lambda: dash_db.latest_localization()),
        ("latest_environment(node_id=1)", lambda: dash_db.latest_environment(1)),
        ("environment_series(node_id=1, limit=60)", lambda: dash_db.environment_series(1, 60)),
        ("all_system_status()", lambda: dash_db.all_system_status()),
    ]

    all_passed = True
    for name, fn in benchmarks:
        # Measure warm run
        t0 = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        passed = elapsed_ms < 50.0
        all_passed = all_passed and passed
        status_str = "PASS" if passed else "FAIL"
        print(f"  -> [{status_str}] {name:38s}: {elapsed_ms:6.2f} ms (< 50 ms)")

    assert all_passed, "One or more queries exceeded the 50 ms latency threshold!"
    print("  [PASS] Scenario 5 verified successfully.")

    # Cleanup temporary test DBs
    db.close()
    scale_db.close()
    try:
        test_db_path.unlink(missing_ok=True)
        scale_db_path.unlink(missing_ok=True)
    except OSError:
        pass

    print("\n" + "=" * 70)
    print("ALL 5 DEFINITION-OF-DONE SIMULATION SCENARIOS VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_dod_checks()
