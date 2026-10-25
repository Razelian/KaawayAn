"""
kaawayan.sim — synthetic stereo signal generation for bearing tests.

All functions are pure (no config file access).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_stereo(
    signal: NDArray[np.floating],
    fs: float,
    theta_deg: float,
    d_m: float,
    c_mps: float,
    snr_db: float,
    rng: np.random.Generator,
    echo: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (left, right) stereo channels with a synthesised inter-channel delay.

    Parameters
    ----------
    signal:
        Mono source signal (float array, arbitrary amplitude).
    fs:
        Sample rate in Hz.
    theta_deg:
        True bearing in degrees.  Positive = source is counterclockwise from
        the node front (toward M1/left), per CONTEXT §5.
    d_m:
        Mic spacing in metres.
    c_mps:
        Speed of sound in m/s.
    snr_db:
        Signal-to-noise ratio in dB (power ratio).
    rng:
        NumPy random Generator — guarantees deterministic output for a fixed seed.
    echo:
        Optional ``(delay_s, gain)`` tuple.  A copy of *signal* scaled by *gain*
        and shifted by *delay_s* is added to **both** channels before the noise
        injection, simulating a room reflection.

    Returns
    -------
    left, right:
        Float64 arrays of the same length as *signal*.

    Notes
    -----
    Convention (CONTEXT §5)::

        tau = t(M2) - t(M1)  [right arrival time minus left arrival time]
             = d * sin(theta) / c

        Positive theta → sound comes from the left side → M1 hears it first
        → t(M1) < t(M2) → tau > 0 → the RIGHT channel is delayed relative to
        the left.

    The right channel is produced by an exact FFT phase shift (no interpolation
    error).  The left channel is the unshifted signal.  Independent Gaussian
    noise is added to each channel to achieve the requested SNR.
    """
    n = len(signal)
    theta_rad = np.deg2rad(theta_deg)
    tau = d_m * np.sin(theta_rad) / c_mps   # seconds; positive → right delayed

    # --- apply echo to the source before splitting into channels --------------
    src = signal.copy().astype(np.float64)
    if echo is not None:
        echo_delay_s, echo_gain = echo
        n_echo = int(round(echo_delay_s * fs))
        if n_echo < n:
            echo_sig = np.zeros(n, dtype=np.float64)
            echo_sig[n_echo:] = src[: n - n_echo] * echo_gain
            src = src + echo_sig

    # --- delay the right channel via FFT phase shift -------------------------
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)   # shape (n//2 + 1,)
    SRC = np.fft.rfft(src)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * tau)
    right_clean = np.fft.irfft(SRC * phase_shift, n=n)
    left_clean = src.copy()

    # --- add independent Gaussian noise for the requested SNR ----------------
    sig_power = np.mean(src ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)

    left = left_clean + rng.standard_normal(n) * noise_std
    right = right_clean + rng.standard_normal(n) * noise_std

    return left.astype(np.float64), right.astype(np.float64)
