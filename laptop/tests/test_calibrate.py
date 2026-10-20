"""
Tests for kaawayan.calibrate — Slice 5 (Step 3).

Tests:
- estimate_tau0: 0.3-sample injected offset → tau0 within 0.03 sample;
  with that tau0 corrected, |theta| < 0.3 deg.
- fit_heading: true 63.4 deg, 5 source positions, 1 deg noise → within 0.5 deg.
- fit_heading wraparound: true 179 deg with noisy values on both sides of 180.
- format_config_snippet: returns TOML lines, never writes files.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from kaawayan.bearing import speed_of_sound, estimate_bearing
from kaawayan.calibrate import estimate_tau0, fit_heading, format_config_snippet
from kaawayan.sim import make_stereo

FS = 48_000
D_M = 0.10
TEMP_C = 20.0
BAND_HZ = (300, 1500)
BETA = 0.25
WINDOW_S = 1.0
HOP_S = 0.5
DURATION_S = 5.0  # enough windows


def _make_stereo_1d(theta_deg: float, snr_db: float = 40.0,
                    seed: int = 0) -> tuple:
    """Return (left_1d, right_1d) float64 arrays."""
    n = int(DURATION_S * FS)
    rng = np.random.default_rng(seed)
    mono = rng.standard_normal(n).astype(np.float64) * 4000.0
    c = speed_of_sound(TEMP_C)
    left, right = make_stereo(mono, FS, theta_deg, D_M, c,
                              snr_db=snr_db, rng=rng)
    return left, right


def _inject_offset(right: np.ndarray, offset_samples: float) -> np.ndarray:
    """Add a fractional-sample phase delay to the right channel."""
    n = len(right)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    R = np.fft.rfft(right)
    tau_extra = offset_samples / FS
    return np.fft.irfft(R * np.exp(-1j * 2.0 * np.pi * freqs * tau_extra), n=n)


# ---------------------------------------------------------------------------
# estimate_tau0 tests
# ---------------------------------------------------------------------------

class TestEstimateTau0:
    def test_offset_detected_within_0_03_sample(self):
        """0.3-sample injected offset → tau0_s within 0.03 sample of truth."""
        left, right = _make_stereo_1d(0.0)
        right_shifted = _inject_offset(right, 0.3)

        result = estimate_tau0(
            left, right_shifted, FS, D_M, TEMP_C,
            band_hz=BAND_HZ, beta=BETA,
            window_s=WINDOW_S, hop_s=HOP_S,
        )
        tau0_samples = result["tau0_s"] * FS
        assert result["n_windows"] >= 5
        assert abs(tau0_samples - 0.3) < 0.03, (
            f"tau0 = {tau0_samples:.4f} samples, expected ~0.3"
        )

    def test_corrected_bearing_under_0_3deg(self):
        """With tau0 applied via estimate_bearing, |theta| < 0.3 deg at 0 true angle."""
        left, right = _make_stereo_1d(0.0)
        right_shifted = _inject_offset(right, 0.3)

        cal = estimate_tau0(
            left, right_shifted, FS, D_M, TEMP_C,
            band_hz=BAND_HZ, beta=BETA,
            window_s=WINDOW_S, hop_s=HOP_S,
        )
        tau0_s = cal["tau0_s"]

        # Now estimate bearing with the calibrated tau0
        errs = []
        from kaawayan.windows import iter_windows
        stereo = np.stack([left, right_shifted], axis=1)
        c = speed_of_sound(TEMP_C)
        for _s, win in iter_windows(stereo, FS, WINDOW_S, HOP_S):
            res = estimate_bearing(
                win[:, 0].astype(np.float64),
                win[:, 1].astype(np.float64),
                FS, D_M, TEMP_C,
                band_hz=BAND_HZ, beta=BETA,
                tau0_s=tau0_s,
            )
            errs.append(res["theta_deg"])

        med_theta = float(np.median(np.abs(errs)))
        assert med_theta < 0.3, (
            f"Median |theta| after tau0 correction = {med_theta:.3f} deg"
        )

    def test_no_windows_returns_zero(self):
        """With too short audio (<1 window), returns n_windows=0 and tau0=0."""
        short = np.zeros(100)
        result = estimate_tau0(short, short, FS, D_M, TEMP_C,
                               band_hz=BAND_HZ, beta=BETA,
                               window_s=WINDOW_S, hop_s=HOP_S)
        assert result["n_windows"] == 0
        assert result["tau0_s"] == 0.0


# ---------------------------------------------------------------------------
# fit_heading tests
# ---------------------------------------------------------------------------

class TestFitHeading:
    def _make_measurements(self, true_heading: float,
                           source_positions: list[tuple[float, float]],
                           noise_deg: float = 0.0, seed: int = 1) -> list:
        """Build measurement tuples for a node at (0,0) with given heading."""
        rng = np.random.default_rng(seed)
        node_x, node_y = 0.0, 0.0
        measurements = []
        for src_x, src_y in source_positions:
            # True azimuth from node to source
            true_az = math.degrees(math.atan2(src_y - node_y, src_x - node_x))
            # true bearing = true_az - heading
            true_theta = true_az - true_heading
            # wrap to [-180, 180]
            true_theta = ((true_theta + 180.0) % 360.0) - 180.0
            # Add noise
            measured_theta = true_theta + rng.normal(0.0, noise_deg)
            measurements.append((node_x, node_y, src_x, src_y, measured_theta, 1.0))
        return measurements

    def test_heading_fit_within_0_5deg(self):
        """True heading 63.4 deg, 5 positions, 1 deg noise → within 0.5 deg."""
        true_heading = 63.4
        sources = [
            (3.0, 1.0),
            (2.0, 4.0),
            (1.5, 2.5),
            (4.0, 0.5),
            (2.5, 3.5),
        ]
        measurements = self._make_measurements(true_heading, sources, noise_deg=1.0)
        result = fit_heading(measurements)
        err = abs(result["heading_deg"] - true_heading)
        # Wrap error
        err = ((err + 180.0) % 360.0) - 180.0
        assert abs(err) < 0.5, (
            f"Heading fit error = {err:.3f} deg (got {result['heading_deg']:.3f},"
            f" expected {true_heading})"
        )

    def test_heading_wraparound_179(self):
        """True heading 179 deg; noisy values cross ±180 boundary."""
        true_heading = 179.0
        sources = [
            (0.1, 3.0),
            (-0.1, 3.0),
            (0.05, 2.0),
            (-0.05, 2.5),
            (0.0, 4.0),
        ]
        measurements = self._make_measurements(true_heading, sources, noise_deg=2.0, seed=7)
        result = fit_heading(measurements)
        # Circular difference
        err = result["heading_deg"] - true_heading
        err = ((err + 180.0) % 360.0) - 180.0
        assert abs(err) < 3.0, (
            f"Heading wraparound error = {err:.3f} deg (got {result['heading_deg']:.3f})"
        )

    def test_empty_measurements(self):
        result = fit_heading([])
        assert result["heading_deg"] == pytest.approx(0.0)

    def test_single_measurement(self):
        """Single measurement → heading = true_az - theta, std = 0."""
        # Node at origin, source at (1, 0): true_az = 0 deg
        # heading = 0 - 0 = 0
        result = fit_heading([(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)])
        assert result["heading_deg"] == pytest.approx(0.0, abs=0.01)
        assert result["residual_std_deg"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# format_config_snippet tests
# ---------------------------------------------------------------------------

class TestFormatConfigSnippet:
    def test_returns_string(self):
        s = format_config_snippet(1)
        assert isinstance(s, str)

    def test_contains_node_id(self):
        s = format_config_snippet(2, tau0_s=0.001, heading_deg=63.4)
        assert "2" in s

    def test_tau0_in_snippet(self):
        tau0 = 0.000123456789
        s = format_config_snippet(1, tau0_s=tau0)
        assert "tau0_s" in s
        # 9 decimal places: 0.000123456789 rounds to 0.000123457
        assert "0.00012345" in s

    def test_heading_in_snippet(self):
        s = format_config_snippet(1, heading_deg=63.4)
        assert "heading_deg" in s
        assert "63.4" in s

    def test_never_writes_file(self, tmp_path):
        """format_config_snippet must not create any files."""
        before = list(tmp_path.iterdir())
        _ = format_config_snippet(1, tau0_s=0.001, heading_deg=90.0)
        after = list(tmp_path.iterdir())
        assert before == after
