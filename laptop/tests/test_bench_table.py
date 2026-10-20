"""
Tests for laptop/scripts/bench_table.py — Slice 4t (R1 revision).

All behavioural tests call compute_table() directly.
One CLI smoke test uses subprocess.
"""

from __future__ import annotations

import hashlib
import json
import math
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
DURATION_S = 3.0  # 3 s gives 5 windows with window=1 s, hop=0.5 s

_LAPTOP = pathlib.Path(__file__).parent.parent
_SCRIPT = _LAPTOP / "scripts" / "bench_table.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: pathlib.Path, stereo: np.ndarray) -> None:
    """Write a stereo int16 WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(FS)
        wf.writeframes(stereo.tobytes())


def _make_meta(path: pathlib.Path, true_theta_by_node: dict) -> None:
    meta = {
        "trial_id": "synth",
        "kind": "tone",
        "config_hash": "n/a",
        "git_commit": "n/a",
        "true_theta_deg": {str(k): float(v) for k, v in true_theta_by_node.items()},
    }
    path.write_text(json.dumps(meta))


def _make_site_toml(
    path: pathlib.Path,
    *,
    mic_spacing_m: float = D_M,
    band_hz=None,
    phat_beta: float = 0.25,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    tau0_s: float = 0.0,
    channel_swap: bool = False,
) -> None:
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
        f"tau0_s = {tau0_s}",
        f"channel_swap = {'true' if channel_swap else 'false'}",
        'mic_model = "INMP441"',
    ]
    path.write_text("\n".join(lines))


def _make_trial(
    trial_dir: pathlib.Path,
    theta_deg: float,
    *,
    snr_db: float = 30.0,
    seed: int = 42,
    tau_offset_samples: float = 0.0,  # extra artificial offset added BEFORE writing
) -> None:
    """Build a synthetic trial: one node (node1), white noise at theta_deg.

    Uses 3 s of white noise ~4000 LSB RMS so iter_windows gets 5 windows.
    """
    trial_dir.mkdir(parents=True, exist_ok=True)
    n = int(DURATION_S * FS)
    rng = np.random.default_rng(seed)
    # ~4000 LSB RMS
    mono = rng.standard_normal(n).astype(np.float64) * 4000.0

    c = speed_of_sound(TEMP_C)
    left, right = make_stereo(mono, FS, theta_deg, D_M, c, snr_db=snr_db, rng=rng)

    # Optionally inject an extra fractional-sample offset (FFT phase shift on right)
    if tau_offset_samples != 0.0:
        tau_extra_s = tau_offset_samples / FS
        freqs = np.fft.rfftfreq(n, d=1.0 / FS)
        R = np.fft.rfft(right)
        right = np.fft.irfft(R * np.exp(-1j * 2.0 * np.pi * freqs * tau_extra_s), n=n)

    stereo = np.stack([left, right], axis=1)
    stereo = np.clip(stereo, -32768, 32767).astype(np.int16)
    _make_wav(trial_dir / "node1.wav", stereo)
    _make_meta(trial_dir / "meta.json", {1: theta_deg})


def _load_cfg(cfg_path: pathlib.Path) -> dict:
    from kaawayan.config import load_config
    cfg, _ = load_config(cfg_path)
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_angle_setup(tmp_path):
    """Returns (trial_0_dir, trial_30_dir, cfg_path, cfg) for standard tests."""
    cfg_path = tmp_path / "site.toml"
    _make_site_toml(cfg_path)
    t0 = tmp_path / "trial_0"
    t30 = tmp_path / "trial_30"
    _make_trial(t0, 0.0)
    _make_trial(t30, 30.0)
    cfg = _load_cfg(cfg_path)
    return t0, t30, cfg_path, cfg


# ---------------------------------------------------------------------------
# Test: all 20 (beta, band) cells present, each with entries for both angles
# ---------------------------------------------------------------------------

class TestComputeTableStructure:
    def test_all_20_cells_present(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        from laptop.scripts.bench_table import compute_table, BETAS, BANDS
        result = compute_table([t0, t30], cfg)
        cells = result["cells"]
        summary = result["summary"]
        assert len(cells) == 20, f"Expected 20 cells, got {len(cells)}"
        assert len(summary) == 20
        for beta in BETAS:
            for band in BANDS:
                key = (beta, band)
                assert key in cells, f"Missing cell {key}"
                assert key in summary
                # Both angles present
                assert 0.0 in cells[key], f"0.0 missing from {key}"
                assert 30.0 in cells[key], f"30.0 missing from {key}"

    def test_cell_fields(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        from laptop.scripts.bench_table import compute_table
        result = compute_table([t0, t30], cfg)
        st = result["cells"][(0.25, (300, 1500))][0.0]
        for field in ("median_abs_err", "p95_abs_err", "signed_median",
                      "mean_conf", "pct_gated", "n"):
            assert field in st, f"Field {field!r} missing"
        assert st["n"] >= 1


# ---------------------------------------------------------------------------
# Test: accuracy at (0.25, (300,1500)) for 0 deg and 30 deg
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_median_error_under_1deg_at_0(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        from laptop.scripts.bench_table import compute_table
        result = compute_table([t0, t30], cfg)
        med = result["cells"][(0.25, (300, 1500))][0.0]["median_abs_err"]
        assert med < 1.0, f"Median |err| at 0 deg = {med:.3f} >= 1 deg"

    def test_median_error_under_1deg_at_30(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        from laptop.scripts.bench_table import compute_table
        result = compute_table([t0, t30], cfg)
        med = result["cells"][(0.25, (300, 1500))][30.0]["median_abs_err"]
        assert med < 1.0, f"Median |err| at 30 deg = {med:.3f} >= 1 deg"

    def test_signed_bias_within_1deg_at_30(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        from laptop.scripts.bench_table import compute_table
        result = compute_table([t0, t30], cfg)
        bias = result["cells"][(0.25, (300, 1500))][30.0]["signed_median"]
        assert abs(bias) <= 1.0, f"Bias at 30 deg = {bias:.3f} > 1 deg"


# ---------------------------------------------------------------------------
# Test: calibration (tau0_s correction)
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_raw_offset_causes_bias(self, tmp_path):
        """0.3-sample injected offset → signed bias > 1 deg with --raw."""
        from laptop.scripts.bench_table import compute_table
        cfg_path = tmp_path / "site.toml"
        _make_site_toml(cfg_path, tau0_s=0.0)  # tau0 = 0 in config
        trial = tmp_path / "trial_offset"
        _make_trial(trial, 0.0, tau_offset_samples=0.3)
        cfg = _load_cfg(cfg_path)

        result_raw = compute_table([trial], cfg, raw=True)
        bias_raw = result_raw["cells"][(0.25, (300, 1500))][0.0]["signed_median"]
        assert abs(bias_raw) > 1.0, (
            f"Expected |bias| > 1 deg with raw offset, got {bias_raw:.3f}"
        )

    def test_tau0_corrects_bias(self, tmp_path):
        """With tau0_s set, the bias is under 0.5 deg."""
        from laptop.scripts.bench_table import compute_table
        # The injected offset is 0.3 samples on the right channel.
        # tau0_s to correct it: tau0 = +0.3/FS (right delayed = positive tau)
        tau0_correction = 0.3 / FS
        cfg_path = tmp_path / "site.toml"
        _make_site_toml(cfg_path, tau0_s=tau0_correction)
        trial = tmp_path / "trial_offset"
        _make_trial(trial, 0.0, tau_offset_samples=0.3)
        cfg = _load_cfg(cfg_path)

        result = compute_table([trial], cfg, raw=False)
        bias = result["cells"][(0.25, (300, 1500))][0.0]["signed_median"]
        assert abs(bias) < 0.5, (
            f"Expected |bias| < 0.5 deg with tau0 correction, got {bias:.3f}"
        )


# ---------------------------------------------------------------------------
# Test: determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_two_calls_equal(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        from laptop.scripts.bench_table import compute_table
        r1 = compute_table([t0, t30], cfg)
        r2 = compute_table([t0, t30], cfg)
        # Compare summary dicts
        assert set(r1["summary"].keys()) == set(r2["summary"].keys())
        for key in r1["summary"]:
            v1 = r1["summary"][key]
            v2 = r2["summary"][key]
            assert math.isnan(v1) == math.isnan(v2)
            if not math.isnan(v1):
                assert abs(v1 - v2) < 1e-12
        # Compare every cell
        for key in r1["cells"]:
            for theta in r1["cells"][key]:
                for field in ("median_abs_err", "signed_median", "n"):
                    a = r1["cells"][key][theta][field]
                    b = r2["cells"][key][theta][field]
                    if isinstance(a, float) and math.isnan(a):
                        assert math.isnan(b)
                    else:
                        assert a == b


# ---------------------------------------------------------------------------
# Test: CLI smoke test — stdout contains the SHA-256 of the config file
# ---------------------------------------------------------------------------

class TestCLI:
    def test_stdout_contains_config_sha256(self, two_angle_setup):
        t0, t30, cfg_path, cfg = two_angle_setup
        sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        result = subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--config", str(cfg_path),
             str(t0), str(t30)],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(_LAPTOP)},
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert sha in result.stdout, (
            f"SHA-256 {sha!r} not found in stdout:\n{result.stdout[:500]}"
        )
