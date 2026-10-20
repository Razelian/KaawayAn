"""
kaawayan.bearing — GCC-PHAT bearing estimation.

All functions are pure (no config file access).

Convention (CONTEXT §5, convention_version = 1)
-----------------------------------------------
  tau  = t(M2) - t(M1)   [right minus left]
  theta = asin(c * tau / d)
  Positive tau → sound reaches M1 (left) first → positive theta (CCW).
"""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# speed_of_sound
# ---------------------------------------------------------------------------

def speed_of_sound(temp_c: float) -> float:
    """Return the speed of sound in m/s for air at *temp_c* degrees Celsius.

    Formula: c = 331.3 * sqrt(1 + T / 273.15)

    Parameters
    ----------
    temp_c:
        Air temperature in degrees Celsius.

    Returns
    -------
    float
        Speed of sound in m/s.
    """
    return 331.3 * math.sqrt(1.0 + temp_c / 273.15)


# ---------------------------------------------------------------------------
# gcc_phat
# ---------------------------------------------------------------------------

def gcc_phat(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    fs: float,
    band_hz: tuple[float, float],
    max_lag_s: float,
    beta: float = 1.0,
) -> tuple[float, float, float]:
    """Generalised Cross-Correlation with Phase Transform (GCC-PHAT).

    Estimates the inter-channel delay tau = t(right) - t(left).

    Confidence (CONTEXT §3, rule 9):
        confidence = corr_peak * n / (2 * sum(|G_k|))
    where G_k is the weighted cross-spectrum over band bins only.
    This equals 1 for a pure delay and ~0.02 for independent noise (n=48000).

    Peak-median ratio (CONTEXT §3):
        peak / median(|corr|) over ALL lags (not just the search window).
    Noise gives ~1.5; a real peak gives 5–40.

    Parameters
    ----------
    left, right:
        Mono audio arrays of equal length (float).
    fs:
        Sample rate in Hz.
    band_hz:
        ``(lo, hi)`` pass-band in Hz.  Only bins inside this range participate
        in the weighting.
    max_lag_s:
        Maximum absolute lag to search (seconds).  The returned ``tau_s`` is
        always clipped to ``[-max_lag_s, +max_lag_s]``.
    beta:
        PHAT exponent in [0, 1].  beta=1 → classic PHAT (|G|^-1 weighting).
        beta=0 → plain cross-correlation (no phase whitening).
        G_k = X_k / max(|X_k|, 1e-12)^beta  where X = R * conj(L).
        Added as the LAST keyword argument so existing callers are unaffected.

    Returns
    -------
    tau_s:
        Estimated delay in seconds, sign = t(right) - t(left).
        Positive when sound reaches the LEFT mic first.
    confidence:
        Normalised correlation peak in [0, 1].  ~1 for a pure delay,
        ~0.02 for independent noise at 48 kHz / 1 s.
    peak_median_ratio:
        Ratio of the peak value to the median of |corr| over ALL lags.
        Higher is better.  Noise ~1.5; real peak 5–40.
    """
    n = len(left)
    assert len(right) == n, "left and right must have equal length"

    # --- FFT -----------------------------------------------------------------
    L = np.fft.rfft(left, n=n)
    R = np.fft.rfft(right, n=n)

    # --- band mask -----------------------------------------------------------
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo, hi = band_hz
    mask = (freqs >= lo) & (freqs <= hi)

    # --- weighted cross-spectrum inside the band -----------------------------
    # X_k = R_k * conj(L_k)  (raw cross-spectrum)
    # G_k = X_k / max(|X_k|, 1e-12)^beta
    #   beta=1 → classic PHAT; beta=0 → plain cross-correlation
    XR = R * np.conj(L)
    abs_XR = np.abs(XR)
    denom = np.maximum(abs_XR, 1e-12) ** beta
    gcc_spectrum = np.where(mask, XR / denom, 0.0)

    # --- correlation in time domain ------------------------------------------
    corr = np.fft.irfft(gcc_spectrum, n=n)

    # lags in samples: the IRFFT output has lag-0 at index 0,
    # positive lags at 1..n//2, and negative lags at n//2+1..n-1 (wrapped).
    # Rearrange so the centre is lag=0.
    corr_shift = np.fft.fftshift(corr)
    lags_samples = np.arange(-(n // 2), n - n // 2)   # length n
    lags_s = lags_samples / fs

    # --- restrict search to ±max_lag_s ---------------------------------------
    search_mask = np.abs(lags_s) <= max_lag_s
    corr_search = corr_shift.copy()
    corr_search[~search_mask] = -np.inf

    # --- find the peak -------------------------------------------------------
    peak_idx = int(np.argmax(corr_search))

    # --- parabolic sub-sample interpolation ----------------------------------
    # Use three-point parabola if neighbours are valid.
    if (
        peak_idx > 0
        and peak_idx < n - 1
        and search_mask[peak_idx - 1]
        and search_mask[peak_idx + 1]
    ):
        y_m1 = corr_shift[peak_idx - 1]
        y_0 = corr_shift[peak_idx]
        y_p1 = corr_shift[peak_idx + 1]
        denom_para = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
        if abs(denom_para) > 1e-15:
            delta = (y_p1 - y_m1) / denom_para
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
    else:
        delta = 0.0

    tau_s = (lags_samples[peak_idx] + delta) / fs

    # --- confidence (CONTEXT §3, rule 9) ------------------------------------
    # confidence = corr_peak * n / (2 * sum(|G_k|))  over band bins.
    # For a pure delay, |G_k| = 1 for all n_active band bins and the irfft
    # returns corr_peak = n_active * (2/n) * n/2 * ... the formula below gives
    # exactly 1.  For independent noise it gives ~0.02 (at n=48000, 1 s @ 48 kHz).
    G_abs_band = np.abs(gcc_spectrum[mask])   # |G_k| over band bins only
    sum_G = float(np.sum(G_abs_band))
    n_active = int(np.sum(mask))
    if n_active == 0 or sum_G < 1e-30:
        confidence = 0.0
    else:
        # irfft of a 1-sided spectrum: energy factor is n / 2 for positive freqs
        confidence = float(np.clip(
            corr_shift[peak_idx] * n / (2.0 * sum_G),
            0.0, 1.0
        ))

    # --- peak_median_ratio (over ALL lags) -----------------------------------
    # Use median of |corr| over all n lags — noise ~1.5, real peak 5–40.
    all_abs = np.abs(corr_shift)
    median_val = float(np.median(all_abs))
    peak_val = float(corr_shift[peak_idx])
    if median_val > 1e-30:
        peak_median_ratio = peak_val / median_val
    else:
        peak_median_ratio = float(peak_val)

    return tau_s, confidence, peak_median_ratio


# ---------------------------------------------------------------------------
# tau_to_theta
# ---------------------------------------------------------------------------

def tau_to_theta(
    tau_s: float,
    d_m: float,
    c_mps: float,
    tau0_s: float = 0.0,
) -> float:
    """Convert inter-channel delay to bearing angle.

    Parameters
    ----------
    tau_s:
        Measured delay in seconds (= t(right) - t(left)).
    d_m:
        Mic spacing in metres.
    c_mps:
        Speed of sound in m/s.
    tau0_s:
        Calibration offset in seconds.  Subtracted from *tau_s* before
        conversion.  Defaults to 0.

    Returns
    -------
    float
        Bearing angle in degrees, in the range [-90, 90].
        theta = asin(clip(c * (tau - tau0) / d, -1, 1))
    """
    ratio = c_mps * (tau_s - tau0_s) / d_m
    ratio_clipped = float(np.clip(ratio, -1.0, 1.0))
    return math.degrees(math.asin(ratio_clipped))


# ---------------------------------------------------------------------------
# estimate_bearing
# ---------------------------------------------------------------------------

def estimate_bearing(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    fs: float,
    d_m: float,
    temp_c: float,
    band_hz: tuple[float, float],
    tau0_s: float = 0.0,
    channel_swap: bool = False,
    beta: float = 1.0,
) -> dict:
    """Estimate the bearing of a sound source from a stereo pair.

    Order of operations
    -------------------
    1. If *channel_swap* is True, swap the left and right arrays.
    2. Run GCC-PHAT to obtain *tau_s*, *confidence*, and *peak_median_ratio*.
    3. Subtract *tau0_s* (calibration offset) from *tau_s* before conversion.
    4. Convert the corrected delay to degrees via :func:`tau_to_theta`.

    Convention (CONTEXT §5)
    -----------------------
    Positive *tau_s* → sound reaches the left mic (M1) first → positive
    *theta_deg* (source is counterclockwise, toward M1).

    Parameters
    ----------
    left, right:
        Mono audio arrays (equal length, float).  M1 = left, M2 = right.
    fs:
        Sample rate in Hz.
    d_m:
        Mic spacing in metres.
    temp_c:
        Air temperature in degrees Celsius (used to compute speed of sound).
    band_hz:
        ``(lo, hi)`` pass-band in Hz for the PHAT weighting.
    tau0_s:
        Calibration offset in seconds (default 0).
    channel_swap:
        If True, swap left and right before processing (used when the physical
        wiring is inverted relative to the convention).
    beta:
        PHAT exponent in [0, 1].  beta=1 → classic PHAT; beta=0 → plain
        cross-correlation.  Added as the LAST keyword argument so existing
        callers are unaffected.

    Returns
    -------
    dict with keys:
        ``theta_deg`` : float — bearing in degrees, in [-90, 90].
        ``tau_s``     : float — raw GCC-PHAT delay (before tau0 subtraction).
        ``confidence``: float — normalised correlation peak in [0, 1].
        ``peak_median_ratio``: float — peak-to-median ratio of the correlation.
    """
    # Step 1: optional channel swap
    if channel_swap:
        left, right = right, left

    # Step 2: GCC-PHAT (max_lag = physical maximum + 50 % margin)
    c_mps = speed_of_sound(temp_c)
    max_lag_s = 1.5 * d_m / c_mps
    tau_s, confidence, pmr = gcc_phat(left, right, fs, band_hz, max_lag_s, beta=beta)

    # Step 3 & 4: subtract tau0 inside tau_to_theta, then convert
    theta_deg = tau_to_theta(tau_s, d_m, c_mps, tau0_s=tau0_s)

    return {
        "theta_deg": theta_deg,
        "tau_s": tau_s,
        "confidence": confidence,
        "peak_median_ratio": pmr,
    }
