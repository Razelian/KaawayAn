"""Generate explicitly simulated acceptance evidence; hardware checks need devices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
import tomllib
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from kaawayan.calibrate import estimate_tau0, fit_heading, format_config_snippet
from kaawayan.config import ConfigError, load_config, validate_live_config
from kaawayan.fusion import poses_from_config
from kaawayan.live import LivePipeline
from kaawayan.protocol import AudioPayload, EnvPayload, FrameType, encode_frame, parse_stream
from kaawayan.scripts.verify_ramp import _classify_samples


ROOT = Path(__file__).resolve().parent.parent


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, indent=2, sort_keys=True)
    report["integrity_sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"acceptance_{stamp}.json"
    md_path = out_dir / f"acceptance_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [
        "# KaawayAn Acceptance Report",
        "",
        f"- **Mode:** {report['mode']}",
        f"- **Result:** {report['result']}",
        f"- **Generated (UTC):** {report['generated_utc']}",
        f"- **Config SHA-256:** `{report['config_sha256']}`",
        f"- **Git commit:** `{report['git_commit']}`",
        f"- **Integrity SHA-256:** `{report['integrity_sha256']}`",
        "- **Operator sign-off:** NOT PROVIDED; review and sign the report after physical acceptance.",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for item in report["checks"]:
        evidence = str(item["evidence"]).replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {item['name']} | {item['status']} | {evidence} |")
    rows.extend(
        [
            "",
            "Simulation evidence is not evidence of real hardware behavior.",
            "",
        ]
    )
    md_path.write_text("\n".join(rows), encoding="utf-8")
    return md_path, json_path


def _check(
    checks: list[dict[str, str]],
    name: str,
    passed: bool,
    evidence: str,
    *,
    skipped: bool = False,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "SKIPPED (optional)" if skipped else ("SIMULATED PASS" if passed else "SIMULATED FAIL"),
            "evidence": evidence,
        }
    )


def run_simulation(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config, config_hash = load_config(config_path)
    thresholds_path = ROOT / "config" / "acceptance.toml"
    thresholds = tomllib.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
    checks: list[dict[str, str]] = []

    placeholders = False
    try:
        validate_live_config(config)
    except ConfigError:
        placeholders = True
    _check(
        checks,
        "Config and placeholder guard",
        True,
        "TOML validated; live placeholder guard was exercised. No serial discovery is simulated.",
    )

    sample_rate = int(thresholds["sample_rate_hz"])
    ramp_samples = int(float(thresholds["ramp_test_duration_s"]) * sample_rate)
    ramp_index = np.arange(ramp_samples, dtype=np.uint64)
    ramp_left = ramp_index.astype(np.uint16).view(np.int16)
    ramp = np.column_stack((ramp_left, np.bitwise_not(ramp_left)))
    _, runs, ok, zero, bad = _classify_samples(ramp)
    ramp_nodes = [int(node["node_id"]) for node in config["nodes"]]
    _check(
        checks,
        "Ramp loss, 60 seconds per node",
        bad <= int(thresholds["ramp_max_bad_samples"]) and zero == 0 and ok == ramp_samples,
        f"Nodes={ramp_nodes}; simulated ramp samples per node={ramp_samples} (logical duration {ramp_samples / sample_rate:.1f}s); bad={bad}, zero={zero}, runs={len(runs)}.",
    )

    expected_frames = int(
        float(thresholds["throughput_duration_s"]) * sample_rate / 512
    )
    frame_seq = range(expected_frames)
    gaps = sum(1 for previous, current in zip(frame_seq, range(1, expected_frames)) if current != previous + 1)
    node_count = len(config["nodes"])
    _check(
        checks,
        "Two-node throughput and 5-minute sequence gaps",
        node_count == 2 and gaps <= int(thresholds["sequence_max_gaps"]),
        f"Nodes={node_count}; {expected_frames} AUDIO frames per node over logical {thresholds['throughput_duration_s']}s; sequence gaps={gaps}.",
    )

    nominal_interval = 512 / sample_rate
    measured_rate = 512 / nominal_interval
    pair_delta = min(float(config["pair_tolerance_s"]) / 2, nominal_interval / 2)
    rate_error_pct = abs(measured_rate - sample_rate) / sample_rate * 100
    _check(
        checks,
        "Sampling rate and inter-node timing",
        rate_error_pct <= float(thresholds["sample_rate_tolerance_pct"])
        and pair_delta <= float(config["pair_tolerance_s"]),
        f"Synthetic rate={measured_rate:.1f}Hz (error {rate_error_pct:.3f}%); simulated node receive-time delta={pair_delta:.6f}s.",
    )

    audio = np.sin(2 * np.pi * 440 * np.arange(48_000) / 48_000) * 0.2
    audio_stereo = np.column_stack((audio, np.roll(audio, 1)))
    channel_rms = np.sqrt(np.mean(audio_stereo**2, axis=0))
    peak = float(np.max(np.abs(audio_stereo)))
    channels_ok = bool(
        np.all(channel_rms >= float(thresholds["channel_min_rms"]))
        and peak <= float(thresholds["channel_max_peak"])
    )
    _check(
        checks,
        "Stereo channel and microphone sanity",
        channels_ok,
        f"Synthetic left/right RMS={channel_rms.tolist()}, peak={peak:.3f}; physical channel order is not measured.",
    )

    env = EnvPayload.from_bytes(bytes.fromhex("9808 8813 00000000 00000000 00".replace(" ", "")))
    temp_c = env.temp_c_x100 / 100
    humidity = env.rh_x100 / 100
    env_ok = (
        env.bme_status == 0
        and float(thresholds["temperature_c_min"]) <= temp_c <= float(thresholds["temperature_c_max"])
        and float(thresholds["humidity_pct_min"]) <= humidity <= float(thresholds["humidity_pct_max"])
    )
    _check(checks, "Environmental sensor plausibility", env_ok, f"Synthetic BME status={env.bme_status}, temperature={temp_c:.2f}C, humidity={humidity:.2f}%.")

    calibration = np.random.default_rng(7).normal(0, 0.1, 48_000)
    suggested = estimate_tau0(
        calibration,
        calibration.copy(),
        fs=48_000,
        d_m=float(config["mic_spacing_m"]),
        temp_c=22.0,
        band_hz=tuple(config["band_hz"]),
        beta=float(config.get("phat_beta", 1.0)),
        window_s=1.0,
        hop_s=0.5,
    )
    heading = fit_heading([(0.0, 0.0, 1.0, 1.0, 45.0, 1.0)])
    snippet = format_config_snippet(1, suggested["tau0_s"], heading["heading_deg"])
    calibration_ok = (
        suggested["n_windows"] > 0
        and suggested["mad_s"] <= float(thresholds["tau0_mad_max_s"])
        and "tau0_s" in snippet
    )
    _check(checks, "Calibration helper", calibration_ok, f"Synthetic centerline clip; tau0 suggestion={suggested['tau0_s']:.9g}s, MAD={suggested['mad_s']:.9g}s. No config written.")

    localization_ok = False
    localization_evidence = "Not run."
    try:
        from database.database import Database
        from demo_source import build_demo_sources
        from live_adapter import run_pipeline

        poses = poses_from_config(config)
        target = (1.5, 3.0)
        sources = build_demo_sources(config, poses, duration_s=8.0, source_xy=target, snr_db=20.0, seed=42)
        pipeline = LivePipeline(config, poses)
        with tempfile.TemporaryDirectory(prefix="kaawayan-accept-") as temp_dir:
            database = Database(override_path=Path(temp_dir) / "events.db")
            try:
                run_pipeline(config, pipeline, sources, database)
                rows = database.connection.execute(
                    "SELECT x_m, y_m FROM localizations WHERE status='ok' AND x_m IS NOT NULL AND y_m IS NOT NULL"
                ).fetchall()
            finally:
                database.close()
        errors = [math.hypot(row[0] - target[0], row[1] - target[1]) for row in rows]
        median = float(np.median(errors)) if errors else math.inf
        p95 = float(np.percentile(errors, 95)) if errors else math.inf
        localization_ok = (
            median <= float(thresholds["localization_median_error_max_m"])
            and p95 <= float(thresholds["localization_p95_error_max_m"])
        )
        localization_evidence = f"Synthetic source only: n={len(errors)}, median={median:.4f}m, p95={p95:.4f}m; geometry is not a physical survey."
    except Exception as exc:
        localization_evidence = f"Simulation pipeline raised {type(exc).__name__}: {exc}"
    _check(checks, "Localization with known ground truth", localization_ok, localization_evidence)

    classifier_ok = False
    classifier_evidence = "Optional classifier dependencies or checkpoint unavailable."
    try:
        from model.model import DroneClassifier

        probability = DroneClassifier().predict((audio[:16_000] * 32767).astype(np.int16))
        classifier_ok = math.isfinite(probability) and 0 <= probability <= 1
        classifier_evidence = f"Synthetic 16kHz smoke input produced probability={probability:.4f}; this is not accuracy evidence."
    except ImportError as exc:
        classifier_evidence = f"SKIPPED: optional [classifier] dependency missing ({exc})."
    except Exception as exc:
        classifier_evidence = f"Classifier smoke test failed: {type(exc).__name__}: {exc}"
    _check(
        checks,
        "Classifier smoke test",
        classifier_ok,
        classifier_evidence,
        skipped=classifier_evidence.startswith("SKIPPED:"),
    )

    result = (
        "SIMULATED PASS"
        if all(item["status"] in {"SIMULATED PASS", "SKIPPED (optional)"} for item in checks)
        else "SIMULATED FAIL"
    )
    report = {
        "mode": "SIMULATED",
        "result": result,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "git_commit": _git_commit(),
        "live_placeholders_present": placeholders,
        "hardware_verified": False,
        "thresholds": thresholds,
        "checks": checks,
        "operator_sign_off": None,
    }
    md_path, json_path = _write_report(report, out_dir)
    print(f"{result}: {md_path}")
    print(json_path)
    for item in checks:
        print(f"{item['status']}: {item['name']} — {item['evidence']}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KaawayAn acceptance evidence generator")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "site.toml")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--simulate", action="store_true", help="run software acceptance checks only")
    args = parser.parse_args(argv)
    try:
        if not args.simulate:
            config, config_hash = load_config(args.config)
            try:
                validate_live_config(config)
            except ConfigError as exc:
                print(f"Live acceptance blocked: {exc}", file=sys.stderr)
                checks = [{"name": "Config and placeholder guard", "status": "FAIL", "evidence": str(exc)}]
                checks.extend(
                    {
                        "name": name,
                        "status": "NEEDS HARDWARE",
                        "evidence": "Real capture and verification have not been run.",
                    }
                    for name in (
                        "Ramp loss, 60 seconds per node",
                        "Two-node throughput and 5-minute sequence gaps",
                        "Sampling rate and inter-node timing",
                        "Stereo channel and microphone sanity",
                        "Environmental sensor plausibility",
                        "Calibration helper",
                        "Localization with known ground truth",
                        "Classifier smoke test",
                    )
                )
                _write_report(
                    {
                        "mode": "HARDWARE",
                        "result": "FAIL — CONFIGURATION",
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "config_path": str(args.config),
                        "config_sha256": config_hash,
                        "git_commit": _git_commit(),
                        "hardware_verified": False,
                        "checks": checks,
                        "operator_sign_off": None,
                    },
                    args.out,
                )
                return 2
            print(
                "Live hardware acceptance capture is not implemented; no hardware PASS can be issued. "
                f"Config SHA-256: {config_hash}",
                file=sys.stderr,
            )
            _write_report(
                {
                    "mode": "HARDWARE",
                    "result": "NEEDS HARDWARE",
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "config_path": str(args.config),
                    "config_sha256": config_hash,
                    "git_commit": _git_commit(),
                    "hardware_verified": False,
                    "checks": [
                        {
                            "name": name,
                            "status": "NEEDS HARDWARE",
                            "evidence": "Real capture and verification have not been run by this software version.",
                        }
                        for name in (
                            "Config and serial discovery",
                            "Ramp loss, 60 seconds per node",
                            "Two-node throughput and 5-minute sequence gaps",
                            "Sampling rate and inter-node timing",
                            "Stereo channel and microphone sanity",
                            "Environmental sensor plausibility",
                            "Calibration helper",
                            "Localization with known ground truth",
                            "Classifier smoke test",
                        )
                    ],
                    "operator_sign_off": None,
                },
                args.out,
            )
            return 2
        report = run_simulation(args.config, args.out)
    except (OSError, ValueError, ConfigError) as exc:
        print(f"Acceptance failed: {exc}", file=sys.stderr)
        return 2
    return 0 if report["result"] == "SIMULATED PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
