"""Live KaawayAn ingest: SerialSource -> LivePipeline -> SQLite.

Usage:
    python main.py [--config config\\site.toml] [--no-classifier] [--demo] [--record] [--selftest]
"""

from __future__ import annotations

import argparse
import logging
import signal
import shutil
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, Callable
from logging.handlers import RotatingFileHandler

import numpy as np

from config import Config
from database.database import DEFAULT_DB_PATH, DEMO_DB_PATH, Database
from demo_source import build_demo_sources
from kaawayan.config import ConfigError, load_config, validate_live_config
from kaawayan.fusion import poses_from_config
from kaawayan.live import LivePipeline
from live_adapter import ClassifierWorker, default_cfg_path, run_pipeline

log = logging.getLogger("kaawayan.dash")


def _stub_predict(_samples) -> float:
    return 0.0


def _load_predictor(no_classifier: bool, demo: bool = False) -> tuple[Callable[[np.ndarray], float], str]:
    if no_classifier:
        print("[classifier] disabled (--no-classifier); using stub predict")
        return _stub_predict, "disabled"
    try:
        from model.model import DroneClassifier

        model = DroneClassifier()
        print("[classifier] DroneClassifier loaded successfully")
        return model.predict, "ok"
    except Exception as exc:
        if not demo:
            msg = (
                f"[classifier] FATAL: Could not load torch/model in live mode: {exc!r}\n"
                "Live mode requires the real neural network classifier.\n"
                "To explicitly run with a stub classifier, pass --no-classifier."
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(msg) from exc
        print(
            f"[classifier] WARNING: model load failed in demo mode ({exc!r}).\n"
            "Using stub classifier for simulation."
        )
        return _stub_predict, "failed"


def _open_serial_sources(cfg: dict) -> dict:
    from kaawayan.record import SerialSource

    sources = {}
    opened = []
    try:
        for node in cfg.get("nodes", []):
            nid = node["node_id"]
            ident = node.get("usb_serial") or ""
            print(f"[ingest] opening node {nid} via {ident!r}")
            sources[nid] = SerialSource(ident)
            opened.append(nid)
    except Exception:
        for src in sources.values():
            try:
                src.close()
            except Exception:
                pass
        raise
    return sources


def run_selftest(
    cfg_path: Path,
    cfg: dict,
    db: Database,
    no_classifier: bool = False,
    demo: bool = False,
) -> int:
    """Preflight checklist for live deployment."""
    print("\n=======================================================")
    print("KaawayAn Live Preflight Checklist (--selftest)")
    print("=======================================================")

    results: list[tuple[str, bool, str]] = []
    try:
        thresholds = tomllib.loads(
            (Path(__file__).resolve().parent / "config" / "acceptance.toml").read_text(encoding="utf-8")
        )["thresholds"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        thresholds = {}
        results.append(("Acceptance thresholds", False, f"Could not load config/acceptance.toml: {exc}"))

    # 1. Check site.toml for placeholders
    try:
        raw_toml = cfg_path.read_text(encoding="utf-8")
        if "REPLACE_WITH_" in raw_toml and not demo:
            results.append(("Config Placeholders", False, "Found REPLACE_WITH_* placeholders in site.toml"))
        elif demo and "REPLACE_WITH_" in raw_toml:
            results.append(("Config Placeholders", True, "Demo mode does not open hardware ports; placeholders remain unmodified"))
        else:
            results.append(("Config Placeholders", True, "No REPLACE_WITH_* placeholders in site.toml"))
    except Exception as e:
        results.append(("Config Placeholders", False, str(e)))

    # 2. Check configured serial ports exist
    try:
        if demo:
            results.append(("Serial Ports", True, "Demo mode; physical USB discovery is not claimed"))
        else:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            available_ports = [p.device for p in ports] + [getattr(p, "serial_number", None) for p in ports]
            available_ports = [p for p in available_ports if p]
            missing_nodes = []
            for n in cfg.get("nodes", []):
                ident = n.get("usb_serial", "")
                if ident and ident not in available_ports:
                    missing_nodes.append(f"Node {n['node_id']} ({ident})")
            if missing_nodes:
                results.append(("Serial Ports", False, f"Configured ports not found: {', '.join(missing_nodes)}"))
            else:
                results.append(("Serial Ports", True, "All configured serial ports detected"))
    except Exception as e:
        results.append(("Serial Ports", False, f"Could not list serial ports: {e}"))

    # 3. Check dummy 16000-sample window runs through classifier
    try:
        predict, status = _load_predictor(no_classifier=no_classifier, demo=demo)
        dummy = np.zeros(16000, dtype=np.float32)
        prob = float(predict(dummy))
        if 0.0 <= prob <= 1.0:
            label = "SIMULATED classifier fallback" if demo and status == "failed" else f"status={status}"
            results.append(("Classifier Inference", True, f"16000-sample window evaluated ({label}, prob={prob:.4f})"))
        else:
            results.append(("Classifier Inference", False, f"Classifier returned out-of-range prob {prob}"))
    except Exception as e:
        results.append(("Classifier Inference", False, f"Classifier failed: {e}"))

    # 4. Check DB is writable
    try:
        t_now = time.time()
        db.set_system_status("selftest_t", f"{t_now:.6f}", t_now)
        st = db.get_system_status("selftest_t")
        if st and "selftest_t" in st["key"]:
            results.append(("Database Writable", True, f"Verified write/readback at {db.path}"))
        else:
            results.append(("Database Writable", False, "Readback failed after write"))
    except Exception as e:
        results.append(("Database Writable", False, f"Database write error: {e}"))

    # 5. Ensure sufficient room for retained database data and optional recordings.
    try:
        free_mb = shutil.disk_usage(cfg_path.resolve().parent).free / (1024 * 1024)
        min_free_mb = float(thresholds["minimum_free_disk_mb"])
        results.append(
            ("Free Disk Space", free_mb >= min_free_mb, f"{free_mb:.0f} MB free; minimum {min_free_mb:.0f} MB")
        )
    except Exception as e:
        results.append(("Free Disk Space", False, f"Could not check free disk space: {e}"))

    # 6. Check both nodes deliver valid frames within 10s.
    try:
        from kaawayan.protocol import EnvPayload, FrameType, StartupPayload
        from kaawayan.record import FrameAssembler
        if demo:
            sources = build_demo_sources(cfg, poses_from_config(cfg), duration_s=0.1)
            received_nodes = set()
            assemblers = {nid: FrameAssembler() for nid in sources}
            for _ in range(20):
                for nid, src in sources.items():
                    for frame, _ in assemblers[nid].feed(src.read(4096), time.time()):
                        if frame["type"] in (FrameType.AUDIO, FrameType.ENV):
                            received_nodes.add(nid)
                if len(received_nodes) == len(sources):
                    break
            for src in sources.values():
                src.close()
            if len(received_nodes) == len(sources):
                results.append(("Frame Delivery (simulation)", True, f"Valid synthetic frames from nodes {sorted(received_nodes)}; hardware not tested"))
            else:
                results.append(("Frame Delivery (simulation)", False, f"Synthetic frames received only from {sorted(received_nodes)}"))
        else:
            sources = _open_serial_sources(cfg)
            started = time.time()
            assemblers = {nid: FrameAssembler() for nid in sources}
            node_ids = list(sources)
            startup: dict[int, StartupPayload] = {}
            valid_env: set[int] = set()
            audio_counts = {nid: 0 for nid in node_ids}
            last_audio_seq: dict[int, int] = {}
            sequence_gaps = {nid: 0 for nid in node_ids}
            while time.time() - started < 10.0:
                for nid, src in sources.items():
                    chunk = src.read(4096)
                    if not chunk:
                        continue
                    for frame, _ in assemblers[nid].feed(chunk, time.time()):
                        if frame["type"] == FrameType.STARTUP:
                            startup[nid] = StartupPayload.from_bytes(frame["payload"])
                        elif frame["type"] == FrameType.AUDIO:
                            audio_counts[nid] += 1
                            seq = int(frame["seq"])
                            if nid in last_audio_seq:
                                sequence_gaps[nid] += max(0, (seq - last_audio_seq[nid] - 1) & 0xFFFFFFFF)
                            last_audio_seq[nid] = seq
                        elif frame["type"] == FrameType.ENV:
                            env = EnvPayload.from_bytes(frame["payload"])
                            temp = env.temp_c_x100 / 100.0
                            humidity = env.rh_x100 / 100.0
                            if (
                                env.bme_status == 0
                                and float(thresholds["temperature_c_min"]) <= temp <= float(thresholds["temperature_c_max"])
                                and float(thresholds["humidity_pct_min"]) <= humidity <= float(thresholds["humidity_pct_max"])
                            ):
                                valid_env.add(nid)
                time.sleep(0.002)
            elapsed = max(time.time() - started, 0.001)
            for nid, src in sources.items():
                try:
                    src.close()
                except Exception:
                    log.exception("Failed to close selftest serial source for node %s", nid)

            startup_ok = set(startup) == set(node_ids)
            incompatible = [
                nid for nid, info in startup.items()
                if info.fw_major != 1 or info.sample_rate_hz != Config.AUDIO_FS
                or info.samples_per_ch != 512 or info.channels != 2
            ]
            results.append(
                (
                    "Firmware version and STARTUP format",
                    startup_ok and not incompatible,
                    f"versions={{{', '.join(f'{nid}: {v.fw_major}.{v.fw_minor}.{v.fw_patch}' for nid, v in startup.items())}}}; incompatible={incompatible}",
                )
            )
            results.append(
                ("Valid ENV frames", valid_env == set(node_ids), f"Valid BME280 ENV nodes: {sorted(valid_env)}")
            )
            rates = {nid: audio_counts[nid] / elapsed * 512 for nid in node_ids}
            tolerance_pct = float(thresholds["audio_rate_tolerance_pct"])
            rate_ok = all(
                abs(rate - Config.AUDIO_FS) / Config.AUDIO_FS * 100 <= tolerance_pct
                for rate in rates.values()
            )
            gaps_ok = all(count == 0 for count in sequence_gaps.values())
            crc_ok = all(assembler.crc_failures == 0 for assembler in assemblers.values())
            results.append(
                (
                    "48kHz audio rate, sequence gaps, and CRC",
                    rate_ok and gaps_ok and crc_ok,
                    f"sample_rates={rates}; sequence_gaps={sequence_gaps}; crc_errors="
                    f"{ {nid: assembler.crc_failures for nid, assembler in assemblers.items()} }",
                )
            )
    except Exception as e:
        results.append(("Frame Delivery (10s)", False, f"Could not receive frames: {e}"))

    # Print summary
    has_failure = False
    for title, ok, detail in results:
        status_tag = "[PASS]" if ok else "[FAIL]"
        if not ok:
            has_failure = True
        print(f"{status_tag} {title}: {detail}")

    print("=======================================================")
    if has_failure:
        print("OVERALL RESULT: FAIL — Preflight checks failed.")
        print("=======================================================\n")
        return 1
    else:
        result = (
            "OVERALL RESULT: SIMULATION PREFLIGHT PASS — physical hardware remains untested."
            if demo
            else "OVERALL RESULT: PASS — preflight checks passed; complete hardware acceptance before production."
        )
        print(result)
        print("=======================================================\n")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KaawayAn live ingest")
    parser.add_argument("--config", default=str(default_cfg_path()))
    parser.add_argument("--no-classifier", action="store_true")
    parser.add_argument("--demo", action="store_true", help="synthetic two-node v1 frames")
    parser.add_argument("--record", action="store_true", help="also write a trial folder")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N wall seconds")
    parser.add_argument("--db", default=None, help="override database file path")
    parser.add_argument("--selftest", action="store_true", help="run preflight checklist")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_dir / "station.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )

    cfg_path = Path(args.config)
    try:
        cfg, cfg_hash = load_config(cfg_path)
    except (OSError, ConfigError, ValueError) as exc:
        print(f"[config] FATAL: {exc}", file=sys.stderr)
        return 2
    poses = poses_from_config(cfg)

    # Route database to demo.db by default in demo mode to keep demo data isolated
    if args.db:
        db_path = Path(args.db)
    elif args.demo:
        db_path = DEMO_DB_PATH
    else:
        db_path = DEFAULT_DB_PATH

    db = Database(override_path=db_path)

    # Preflight self-test mode
    if args.selftest:
        try:
            return run_selftest(
                cfg_path,
                cfg,
                db,
                no_classifier=args.no_classifier,
                demo=args.demo,
            )
        finally:
            db.close()

    # Refuse to start live mode if placeholders remain in site.toml
    if not args.demo:
        try:
            validate_live_config(cfg)
        except ConfigError as exc:
            print(
                f"[config] FATAL: {exc}\n"
                "Refusing to start live mode until node USB serials are surveyed.\n"
                "Run with --demo for simulated exhibit data or --selftest for checklist.",
                file=sys.stderr,
            )
            db.close()
            return 1

    # Record mode in system_status
    mode_str = "demo" if args.demo else "live"
    db.set_system_status("mode", mode_str, time.time())

    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(cfg_hash)

    try:
        predict, cls_status = _load_predictor(args.no_classifier, demo=args.demo)
    except Exception as exc:
        db.set_system_status("classifier", "failed", time.time())
        db.close()
        return 1

    worker = ClassifierWorker(predict, db, Config.drone_detection_threshold, status=cls_status)
    worker.start()

    recorder = None
    if args.record:
        from datetime import datetime
        from kaawayan.record import TrialRecorder

        trial_id = datetime.now().strftime("trial_%Y%m%d_%H%M%S")
        out = Path("trials") / trial_id
        recorder = TrialRecorder(out, [n["node_id"] for n in cfg["nodes"]])
        print(f"[record] writing {out}")

    def serial_source_factory(nid: int):
        from kaawayan.record import SerialSource
        for n in cfg.get("nodes", []):
            if n["node_id"] == nid:
                ident = n.get("usb_serial", "")
                return SerialSource(ident)
        raise ValueError(f"No config for node {nid}")

    sources = None
    stop_event = threading.Event()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
    try:
        if args.demo:
            print(f"[demo] synthetic two-node frames through live ingest -> {db_path.name}")
            sources = build_demo_sources(cfg, poses, duration_s=30.0, loop=True)
            source_factory = None
        else:
            sources = _open_serial_sources(cfg)
            source_factory = serial_source_factory

        print("Pipeline running. Ctrl+C to stop.")
        run_pipeline(
            cfg,
            pipeline,
            sources,
            db,
            classifier=worker,
            recorder=recorder,
            source_factory=source_factory,
            duration_s=args.seconds,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as exc:
        print(f"[error] {exc}")
        return 1
    finally:
        worker.stop()
        if recorder is not None:
            recorder.close(
                trial_id=getattr(recorder, "_out", Path("trials")).name,
                kind="demo" if args.demo else "live",
                cfg_path=cfg_path,
            )
        if sources:
            for src in sources.values():
                try:
                    src.close()
                except Exception:
                    pass
        db.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
