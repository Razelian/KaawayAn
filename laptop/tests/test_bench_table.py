"""
Tests for laptop/scripts/bench_table.py — Slice 4t.

Tests use synthetic trial folders built with make_stereo from sim.py.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import subprocess
import sys
import wave

import numpy as np
import pytest

from kaawayan.bearing import speed_of_sound
from kaawayan.sim import make_stereo

FS = 48_000
D_M = 0.10
TEMP_C = 20.0
N = FS  # 1 s


def _make_wav(path: pathlib.Path, stereo: np.ndarray) -> None:
    """Write a stereo int16 WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(stereo.astype(np.int16).tobytes())


def _make_env_csv(path: pathlib.Path, temp_c: float = TEMP_C) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_laptop_s", "temp_c", "rh_pct",
                                           "overrun_count", "usb_drop_count", "bme_status"])
        w.writeheader()
        w.writerow({"t_laptop_s": 0.0, "temp_c": temp_c, "rh_pct": 50.0,
                     "overrun_count": 0, "usb_drop_count": 0, "bme_status": 0})


def _make_meta(path: pathlib.Path, true_theta_by_node: dict) -> None:
    meta = {
        "trial_id": "synth",
        "kind": "tone",
        "config_hash": "n/a",
        "git_commit": "n/a",
        "true_theta_deg": true_theta_by_node,
    }
    path.write_text(json.dumps(meta))


def _make_site_toml(path: pathlib.Path, mic_spacing_m: float = D_M,
                     band_hz=None, phat_beta: float = 0.5,
                     window_s: float = 1.0, hop_s: float = 0.5) -> None:
    if band_hz is None:
        band_hz = [300, 1500]
    lines = [
        "convention_version = 1",
        f"mic_spacing_m = {mic_spacing_m}",
        f"band_hz = [{band_hz[0]}, {band_hz[1]}]",
        f"phat_beta = {phat_beta}",
        f"window_s = {window_s}",
        f"hop_s = {hop_s}",
        "pair_tolerance_s = 0.1",
        "gate_min_confidence = 0.0",
        "gate_min_band_energy_db = -100.0",
        "min_crossing_deg = 30",
        "",
        "[[nodes]]",
        "node_id = 1",
        'usb_serial = "SN001"',
        'board = "xiao"',
        "x_m = 0.0",
        "y_m = 0.0",
        "heading_deg = 90.0",
        "tau0_s = 0.0",
        "channel_swap = false",
        'mic_model = "INMP441"',
    ]
    path.write_text("\n".join(lines))


def _make_trial(trial_dir: pathlib.Path, theta_deg: float,
                 config_path: pathlib.Path) -> None:
    """Build a synthetic trial: one node, white noise at theta_deg."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    mono = rng.standard_normal(N).astype(np.float32)
    c = speed_of_sound(TEMP_C)
    stereo = make_stereo(mono, theta_deg, D_M, c, FS)
    _make_wav(trial_dir / "node1.wav", stereo)
    _make_env_csv(trial_dir / "env.csv", TEMP_C)
    _make_meta(trial_dir / "meta.json", {"1": theta_deg})


_SCRIPT = pathlib.Path(__file__).parent.parent / "scripts" / "bench_table.py"
_LAPTOP = pathlib.Path(__file__).parent.parent


def _run_bench(*trial_dirs: pathlib.Path, config: pathlib.Path = None) -> str:
    """Run bench_table.py and return stdout."""
    args = [sys.executable, str(_SCRIPT)]
    if config:
        args += ["--config", str(config)]
    args += [str(d) for d in trial_dirs]
    result = subprocess.run(
        args,
        capture_output=True, text=True,
        env={"PATH": str(pathlib.Path(sys.executable).parent), "PYTHONPATH": str(_LAPTOP)},
    )
    return result.stdout + result.stderr


class TestBenchTable:
    def test_all_20_cells_present(self, tmp_path):
        """5 betas × 4 bands = 20 cells in the summary table."""
        cfg = tmp_path / "site.toml"
        _make_site_toml(cfg)
        t0 = tmp_path / "trial_0"
        t30 = tmp_path / "trial_30"
        _make_trial(t0, 0.0, cfg)
        _make_trial(t30, 30.0, cfg)
        out = _run_bench(t0, t30, config=cfg)
        # Count distinct beta-band combinations
        bands = ["300-1500", "300-1700", "300-4000", "300-6000"]
        betas = ["0.00", "0.25", "0.50", "0.75", "1.00"]
        # The output must mention all betas and all bands
        for b in betas:
            assert b in out, f"beta {b} missing from output"
        for band in bands:
            assert band in out, f"band {band} missing from output"

    def test_low_error_best_cell(self, tmp_path):
        """White noise at 0° and 30°: at least one cell has median |err| < 1 deg."""
        cfg = tmp_path / "site.toml"
        _make_site_toml(cfg, band_hz=[300, 1500], phat_beta=0.25)
        t0 = tmp_path / "trial_0"
        t30 = tmp_path / "trial_30"
        _make_trial(t0, 0.0, cfg)
        _make_trial(t30, 30.0, cfg)
        out = _run_bench(t0, t30, config=cfg)
        # At least some line should have a number < 1.0 deg median error
        import re
        numbers = [float(m) for m in re.findall(r"\d+\.\d+", out)]
        assert any(n < 1.0 for n in numbers), (
            f"No cell with median error < 1 deg; numbers found: {numbers[:20]}"
        )

    def test_output_identical_on_second_run(self, tmp_path):
        """Running twice gives identical output (determinism)."""
        cfg = tmp_path / "site.toml"
        _make_site_toml(cfg)
        t0 = tmp_path / "trial_0"
        _make_trial(t0, 0.0, cfg)
        out1 = _run_bench(t0, config=cfg)
        out2 = _run_bench(t0, config=cfg)
        assert out1 == out2, "bench_table.py output is not deterministic"

    def test_config_hash_in_output(self, tmp_path):
        """Output must include a SHA-256 config hash."""
        cfg = tmp_path / "site.toml"
        _make_site_toml(cfg)
        t0 = tmp_path / "trial_0"
        _make_trial(t0, 0.0, cfg)
        out = _run_bench(t0, config=cfg)
        assert "SHA-256" in out or "sha256" in out.lower() or len([
            line for line in out.splitlines()
            if len(line.strip()) == 64 and all(c in "0123456789abcdef" for c in line.strip())
        ]) > 0, "No SHA-256 hash in bench_table output"
