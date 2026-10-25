"""
Tests for bearing.py and sim.py — Slice 2 + Slice 2b.

Convention (CONTEXT §5):
  tau = t(M2) - t(M1); positive when sound reaches M1 (left) first.
  theta = asin(c * tau / d); positive counterclockwise (toward M1).
  Pinning: delaying the RIGHT channel by +tau gives POSITIVE theta.

Test parameters
---------------
  fs      = 48 000 Hz
  d       = 0.10 m
  T       = 20 °C
  band_hz = (300, 1500)
  SNR     = 20 dB
  signal  = 1 s white noise
  max_lag = 1.5 * d / c  (slightly above the geometric maximum)
"""

import math
import numpy as np
import pytest

from kaawayan.bearing import (
    speed_of_sound,
    gcc_phat,
    tau_to_theta,
    estimate_bearing,
)
from kaawayan.sim import make_stereo

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
FS = 48_000
D = 0.10           # mic spacing, metres
T_C = 20.0         # Celsius
BAND = (300, 1500)
SNR_DB = 20.0
N = FS * 1         # 1 s of samples
C = speed_of_sound(T_C)
MAX_LAG = 1.5 * D / C   # just above the physical maximum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _white(rng, n: int) -> np.ndarray:
    """Unit-variance white noise."""
    return rng.standard_normal(n)


def _harmonic_source(f0: float, n_harmonics: int, n: int, fs: float, rng) -> np.ndarray:
    """Harmonic source: sum of harmonics at f0*k with amplitude 1/k and random phases."""
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        sig += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + phase)
    return sig


# ---------------------------------------------------------------------------
# speed_of_sound
# ---------------------------------------------------------------------------
class TestSpeedOfSound:
    def test_zero_celsius(self):
        assert math.isclose(speed_of_sound(0.0), 331.3, rel_tol=1e-6)

    def test_20_celsius(self):
        expected = 331.3 * math.sqrt(1 + 20.0 / 273.15)
        assert math.isclose(speed_of_sound(20.0), expected, rel_tol=1e-9)

    def test_increases_with_temperature(self):
        assert speed_of_sound(30.0) > speed_of_sound(20.0)


# ---------------------------------------------------------------------------
# tau_to_theta
# ---------------------------------------------------------------------------
class TestTauToTheta:
    def test_zero_tau_gives_zero_degrees(self):
        assert tau_to_theta(0.0, D, C) == pytest.approx(0.0, abs=1e-9)

    def test_positive_tau_gives_positive_theta(self):
        tau = 0.5 * D / C   # 30 deg
        theta = tau_to_theta(tau, D, C)
        assert theta == pytest.approx(30.0, abs=1e-6)

    def test_tau0_subtracted(self):
        tau = 0.5 * D / C
        theta_with_offset = tau_to_theta(tau, D, C, tau0_s=tau)
        assert theta_with_offset == pytest.approx(0.0, abs=1e-9)

    def test_clipped_at_90(self):
        big_tau = 10.0 * D / C
        assert abs(tau_to_theta(big_tau, D, C)) <= 90.0


# ---------------------------------------------------------------------------
# gcc_phat: sign convention (original slice 2)
# ---------------------------------------------------------------------------
class TestGccPhatSign:
    """Pinning test: delay the RIGHT channel by +tau → POSITIVE theta."""

    def test_positive_delay_on_right_gives_positive_tau(self):
        rng = _rng(0)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=40.0, rng=rng)
        tau, conf, _ = gcc_phat(left, right, FS, BAND, MAX_LAG)
        assert tau > 0.0, "positive theta must give positive tau"
        assert conf > 0.0

    def test_negative_theta_gives_negative_tau(self):
        rng = _rng(1)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=-30.0, d_m=D, c_mps=C,
                                  snr_db=40.0, rng=rng)
        tau, _, _ = gcc_phat(left, right, FS, BAND, MAX_LAG)
        assert tau < 0.0


# ---------------------------------------------------------------------------
# gcc_phat: sign convention — beta parametrised (slice 2b)
# ---------------------------------------------------------------------------
class TestGccPhatSignBeta:
    """Sign convention must hold for beta in {1.0, 0.5, 0.0} (CONTEXT §5)."""

    @pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
    def test_positive_theta_gives_positive_tau_at_beta(self, beta):
        rng = _rng(0)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=40.0, rng=rng)
        tau, conf, _ = gcc_phat(left, right, FS, BAND, MAX_LAG, beta=beta)
        assert tau > 0.0, f"beta={beta}: positive theta must give positive tau"
        assert conf > 0.0

    @pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
    def test_negative_theta_gives_negative_tau_at_beta(self, beta):
        rng = _rng(1)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=-30.0, d_m=D, c_mps=C,
                                  snr_db=40.0, rng=rng)
        tau, _, _ = gcc_phat(left, right, FS, BAND, MAX_LAG, beta=beta)
        assert tau < 0.0, f"beta={beta}: negative theta must give negative tau"


# ---------------------------------------------------------------------------
# gcc_phat: max-lag clamp
# ---------------------------------------------------------------------------
class TestGccPhatMaxLag:
    def test_returned_lag_never_exceeds_max_lag(self):
        """Even with low SNR or adversarial signal the lag is clamped."""
        rng = _rng(99)
        # Independent noise → GCC peak may land anywhere; clamp must hold.
        left = _white(rng, N)
        right = _white(rng, N)
        tight_lag = 0.5 * D / C
        tau, _, _ = gcc_phat(left, right, FS, BAND, tight_lag)
        assert abs(tau) <= tight_lag + 1e-12


# ---------------------------------------------------------------------------
# estimate_bearing: accuracy over several angles
# ---------------------------------------------------------------------------
class TestEstimateBearingAccuracy:
    @pytest.mark.parametrize("theta_true", [-60, -30, 0, 30, 60])
    def test_accuracy_within_half_degree(self, theta_true):
        rng = _rng(theta_true + 100)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=theta_true, d_m=D, c_mps=C,
                                  snr_db=SNR_DB, rng=rng)
        result = estimate_bearing(left, right, FS, D, T_C, BAND)
        err = abs(result["theta_deg"] - theta_true)
        # At ±60° (endfire) cos(60°)=0.5 so each sample spans ~8° of bearing;
        # 1.0° tolerance is still well inside the T3 spec (≤8° at ±60°).
        tol = 1.0 if abs(theta_true) == 60 else 0.5
        assert err < tol, (
            f"theta_true={theta_true}, recovered={result['theta_deg']:.3f}, err={err:.3f}"
        )

    def test_17p3_degrees_recovered(self):
        """17.3 deg is a non-round angle; tests sub-sample interpolation."""
        rng = _rng(200)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=17.3, d_m=D, c_mps=C,
                                  snr_db=SNR_DB, rng=rng)
        result = estimate_bearing(left, right, FS, D, T_C, BAND)
        err = abs(result["theta_deg"] - 17.3)
        assert err < 0.5, f"17.3 deg: recovered={result['theta_deg']:.3f}, err={err:.3f}"


# ---------------------------------------------------------------------------
# estimate_bearing: channel_swap flips sign
# ---------------------------------------------------------------------------
class TestChannelSwap:
    def test_swap_flips_theta(self):
        rng = _rng(7)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=SNR_DB, rng=rng)
        normal = estimate_bearing(left, right, FS, D, T_C, BAND)
        swapped = estimate_bearing(left, right, FS, D, T_C, BAND, channel_swap=True)
        # Sign must flip; magnitude roughly preserved (within 2 deg due to noise).
        assert normal["theta_deg"] * swapped["theta_deg"] < 0, (
            f"normal={normal['theta_deg']:.3f}, swapped={swapped['theta_deg']:.3f}"
        )
        assert abs(abs(normal["theta_deg"]) - abs(swapped["theta_deg"])) < 2.0


# ---------------------------------------------------------------------------
# estimate_bearing: tau0 removes a fixed offset
# ---------------------------------------------------------------------------
class TestTau0Removal:
    def test_tau0_corrects_fixed_delay_offset(self):
        """A fixed delay added to the right channel is removed by tau0."""
        rng = _rng(11)
        sig = _white(rng, N)
        # Synthesise at 0 deg so true tau = 0.
        left, right = make_stereo(sig, FS, theta_deg=0.0, d_m=D, c_mps=C,
                                  snr_db=SNR_DB, rng=rng)
        # First: measure the bias when tau0=0 → should already be ~0.
        r0 = estimate_bearing(left, right, FS, D, T_C, BAND)

        # Introduce a small artificial hardware bias on the right channel (fixed shift).
        bias_s = 0.2 * D / C   # ~10 deg of bias at 20 °C
        n_bias = int(round(bias_s * FS))
        right_biased = np.roll(right, n_bias)   # cyclic shift ≈ delay for long signals
        right_biased[:n_bias] = 0.0

        # Without correction the bearing is biased.
        r_biased = estimate_bearing(left, right_biased, FS, D, T_C, BAND)
        # With tau0 = measured tau the bearing is restored to ~0.
        tau0 = r_biased["tau_s"]
        r_corrected = estimate_bearing(left, right_biased, FS, D, T_C, BAND, tau0_s=tau0)
        assert abs(r_corrected["theta_deg"]) < 1.5, (
            f"After tau0 removal: {r_corrected['theta_deg']:.3f} deg (expected ~0)"
        )


# ---------------------------------------------------------------------------
# Confidence: noise-only well below signal confidence (original slice 2)
# ---------------------------------------------------------------------------
class TestConfidence:
    def test_noise_confidence_below_half_signal_confidence(self):
        rng = _rng(42)
        sig = _white(rng, N)
        left_sig, right_sig = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                          snr_db=SNR_DB, rng=rng)
        r_sig = estimate_bearing(left_sig, right_sig, FS, D, T_C, BAND)

        # Independent noise channels — no common signal.
        left_noise = rng.standard_normal(N)
        right_noise = rng.standard_normal(N)
        _, conf_noise, _ = gcc_phat(left_noise, right_noise, FS, BAND, MAX_LAG)

        assert conf_noise < 0.5 * r_sig["confidence"], (
            f"noise conf={conf_noise:.4f}, signal conf={r_sig['confidence']:.4f}"
        )


# ---------------------------------------------------------------------------
# Echo robustness: error stays under 3 deg with (5 ms, 0.5) echo
# ---------------------------------------------------------------------------
class TestEchoRobustness:
    def test_echo_error_under_3_deg(self):
        rng = _rng(55)
        sig = _white(rng, N)
        echo = (0.005, 0.5)   # 5 ms delay, gain 0.5
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=SNR_DB, rng=rng, echo=echo)
        result = estimate_bearing(left, right, FS, D, T_C, BAND)
        err = abs(result["theta_deg"] - 30.0)
        print(f"\n[echo test] theta_recovered={result['theta_deg']:.3f} deg, err={err:.3f} deg")
        assert err < 3.0, f"Echo error too large: {err:.3f} deg"


# ===========================================================================
# Slice 2b — new tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Absolute confidence tests (CONTEXT rule 9)
# ---------------------------------------------------------------------------
class TestAbsoluteConfidence:
    """Rule 9: absolute tests on known inputs — not just relative comparisons."""

    @pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
    def test_high_snr_white_source_confidence_above_0p95(self, beta):
        """Pure delay at 200 dB SNR ≈ noiseless → confidence must be >= 0.95."""
        rng = _rng(42)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=200.0, rng=rng)
        _, conf, _ = gcc_phat(left, right, FS, BAND, MAX_LAG, beta=beta)
        assert conf >= 0.95, (
            f"beta={beta}: noiseless confidence={conf:.4f}, expected >= 0.95"
        )

    @pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
    def test_independent_noise_confidence_below_0p1(self, beta):
        """Independent noise on L and R (no source) → confidence must be < 0.1."""
        rng = _rng(99)
        left = rng.standard_normal(N)
        right = rng.standard_normal(N)
        _, conf, _ = gcc_phat(left, right, FS, BAND, MAX_LAG, beta=beta)
        assert conf < 0.1, (
            f"beta={beta}: noise-only confidence={conf:.4f}, expected < 0.1"
        )

    @pytest.mark.parametrize("beta", [1.0, 0.5, 0.0])
    def test_0dB_snr_white_source_confidence_in_range(self, beta):
        """White source at 0 dB SNR → confidence in (0.2, 0.9)."""
        rng = _rng(7)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=0.0, rng=rng)
        _, conf, _ = gcc_phat(left, right, FS, BAND, MAX_LAG, beta=beta)
        assert 0.2 < conf < 0.9, (
            f"beta={beta}: 0dB SNR confidence={conf:.4f}, expected in (0.2, 0.9)"
        )


# ---------------------------------------------------------------------------
# peak_median_ratio absolute tests (CONTEXT rule 9, fixed: over ALL lags)
# ---------------------------------------------------------------------------
class TestPeakMedianRatioAbsolute:
    """Rule 9: PMR over ALL lags — noise should be low, signal should be high."""

    def test_noise_only_pmr_below_3(self):
        """Independent noise → PMR over all lags < 3."""
        rng = _rng(123)
        left = rng.standard_normal(N)
        right = rng.standard_normal(N)
        _, _, pmr = gcc_phat(left, right, FS, BAND, MAX_LAG)
        assert pmr < 3.0, f"noise PMR={pmr:.3f}, expected < 3"

    def test_white_source_0dB_pmr_above_10(self):
        """White source at 0 dB SNR → PMR over all lags > 10."""
        rng = _rng(7)
        sig = _white(rng, N)
        left, right = make_stereo(sig, FS, theta_deg=30.0, d_m=D, c_mps=C,
                                  snr_db=0.0, rng=rng)
        _, _, pmr = gcc_phat(left, right, FS, BAND, MAX_LAG)
        assert pmr > 10.0, f"signal PMR={pmr:.3f}, expected > 10"


# ---------------------------------------------------------------------------
# Harmonic source tests (CONTEXT §3 known risk 2)
# ---------------------------------------------------------------------------
class TestHarmonicSource:
    """
    Harmonic source: f0=180 Hz, 12 harmonics, amplitude 1/k, random phases,
    1 s at 48 kHz, theta=+30, d=0.10 m, band (300, 1500), snr 10 dB, 20 seeds.

    beta=0.5: median |error| < 2 deg
    beta=0.0: median |error| < 1 deg
    beta=1.0: allowed to fail bearing accuracy; assert median confidence < 0.1
              (gate must catch it)
    """

    F0 = 180.0
    N_HARM = 12
    THETA_TRUE = 30.0
    HARM_BAND = (300, 1500)
    SNR_DB_HARM = 10.0
    N_SEEDS = 20

    def _run_seeds(self, beta: float):
        errors = []
        confs = []
        for seed in range(self.N_SEEDS):
            rng = _rng(seed + 1000)
            sig = _harmonic_source(self.F0, self.N_HARM, N, FS, rng)
            left, right = make_stereo(sig, FS, self.THETA_TRUE, D, C,
                                      self.SNR_DB_HARM, rng)
            result = estimate_bearing(left, right, FS, D, T_C, self.HARM_BAND,
                                      beta=beta)
            errors.append(abs(result["theta_deg"] - self.THETA_TRUE))
            confs.append(result["confidence"])
        return np.array(errors), np.array(confs)

    def test_beta_0p5_median_error_below_2_deg(self):
        errors, _ = self._run_seeds(beta=0.5)
        med = float(np.median(errors))
        assert med < 2.0, f"beta=0.5: median |error|={med:.3f} deg, expected < 2 deg"

    def test_beta_0p0_median_error_below_1_deg(self):
        errors, _ = self._run_seeds(beta=0.0)
        med = float(np.median(errors))
        assert med < 1.0, f"beta=0.0: median |error|={med:.3f} deg, expected < 1 deg"

    def test_beta_1p0_confidence_below_0p1(self):
        """Full PHAT on harmonic source must have median confidence < 0.1 (gate catches it)."""
        _, confs = self._run_seeds(beta=1.0)
        med_conf = float(np.median(confs))
        assert med_conf < 0.1, (
            f"beta=1.0: median confidence={med_conf:.4f}, gate must catch it (< 0.1)"
        )
