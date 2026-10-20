"""
kaawayan.windows — Sliding-window helpers for bearing estimation.

Functions
---------
iter_windows(x, fs, window_s, hop_s)
    Yield (start_idx, window) for full windows only (no padding).

band_energy_db(window, fs, band_hz)
    10*log10 of mean power (both channels, band-masked, normalised to int16
    full scale), floored at -120 dB.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

_FLOOR_DB = -120.0


def iter_windows(
    x: NDArray,
    fs: int | float,
    window_s: float,
    hop_s: float,
) -> Iterator[tuple[int, NDArray]]:
    """Yield (start_idx, window) for full windows only.

    Parameters
    ----------
    x : ndarray, shape (n, 2)
        Stereo int16 or float audio.
    fs : int or float
        Sample rate in Hz.
    window_s : float
        Window length in seconds.
    hop_s : float
        Hop (stride) in seconds.

    Yields
    ------
    (start_idx, window) where start_idx is in samples and window has
    shape (window_samples, 2).  Partial windows at the end are dropped.
    """
    n = len(x)
    win_samples = int(round(window_s * fs))
    hop_samples = int(round(hop_s * fs))
    start = 0
    while start + win_samples <= n:
        yield start, x[start : start + win_samples]
        start += hop_samples


def band_energy_db(
    window: NDArray,
    fs: int | float,
    band_hz: tuple[float, float],
) -> float:
    """Return 10*log10 mean band power normalised to int16 full scale.

    Parameters
    ----------
    window : ndarray, shape (n, 2)
        Stereo int16 (or float) audio window.
    fs : int or float
        Sample rate in Hz.
    band_hz : (lo, hi)
        Pass-band in Hz.

    Returns
    -------
    float
        Band energy in dBFS (full-scale reference = 32768), floored at -120 dB.
    """
    n = len(window)
    # Normalise to float in [-1, 1]
    x_f = window.astype(np.float64) / 32768.0  # shape (n, 2)

    # FFT per channel
    lo, hi = band_hz
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= lo) & (freqs <= hi)

    power = 0.0
    for ch in range(x_f.shape[1]):
        X = np.fft.rfft(x_f[:, ch])
        # Mean power in band: |X|^2 / n^2
        # Using Parseval: sum(|x|^2)/n == sum(|X|^2)/n^2  (for rfft up to Nyquist)
        # We compute mean spectral power over the band bins, then scale.
        band_X = X[mask]
        if len(band_X) == 0:
            continue
        # Power = (2 * sum(|X_k|^2) / n^2) for k in band, factor 2 for negative freqs
        # Equivalent to mean(x^2) restricted to the band.
        # Simple approach: mean of |X_k|^2 / n (per-bin power), averaged over band bins.
        # This matches the definition: energy = mean(|x|^2) over the full signal
        # restricted to the band, i.e. Parseval filtered.
        # We use: mean_power = sum(|X_k|^2) * 2 / n^2 / len(band_k) * (number of total bins)
        # Simpler correct approach: reconstruct band-limited signal and compute mean power.
        X_band = np.zeros_like(X)
        X_band[mask] = X[mask]
        x_band = np.fft.irfft(X_band, n=n)
        power += float(np.mean(x_band ** 2))

    # Average over channels
    n_channels = x_f.shape[1]
    power /= n_channels

    if power <= 0:
        return _FLOOR_DB

    db = 10.0 * math.log10(power)
    return max(db, _FLOOR_DB)
