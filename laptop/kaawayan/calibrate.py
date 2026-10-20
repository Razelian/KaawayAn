"""
kaawayan.calibrate — tau0 estimation and heading fit for per-node calibration.

Functions
---------
estimate_tau0(left, right, fs, d_m, temp_c, band_hz, beta, window_s, hop_s) -> dict
    Median of the raw GCC-PHAT tau_s over windows of a centerline clip.
    Returns dict(tau0_s, mad_s, n_windows).

fit_heading(measurements) -> dict
    Confidence-weighted circular mean of (true_azimuth - measured_theta).
    Returns dict(heading_deg, residual_std_deg).

format_config_snippet(node_id, tau0_s=None, heading_deg=None) -> str
    Returns TOML lines to paste into site.toml (never writes the file).

Note: site.toml is NEVER written automatically; its SHA-256 is stored with
every trial and must remain under the user's explicit control.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from kaawayan.bearing import gcc_phat, speed_of_sound
from kaawayan.windows import iter_windows


# ---------------------------------------------------------------------------
# estimate_tau0
# ---------------------------------------------------------------------------

def estimate_tau0(
    left: NDArray,
    right: NDArray,
    fs: float,
    d_m: float,
    temp_c: float,
    band_hz: tuple[float, float],
    beta: float,
    window_s: float,
    hop_s: float,
) -> dict:
    """Estimate the raw per-node tau0 from a centerline (theta=0) clip.

    The source should be directly ahead so the true tau is 0.  Any residual
    is the hardware/mechanical offset tau0.

    Parameters
    ----------
    left, right:
        1-D float arrays from the stereo pair (left = M1, right = M2).
    fs:
        Sample rate in Hz.
    d_m:
        Mic spacing in metres.
    temp_c:
        Temperature in Celsius (used to compute max_tau_s).
    band_hz:
        (lo, hi) pass-band in Hz.
    beta:
        GCC-PHAT weighting exponent (0 = plain CC, 1 = classic PHAT).
    window_s, hop_s:
        Window and hop lengths in seconds.

    Returns
    -------
    dict with keys:
        tau0_s: float — median raw tau over all windows (seconds).
        mad_s: float — median absolute deviation of tau_s (seconds).
        n_windows: int — number of windows processed.
    """
    c = speed_of_sound(temp_c)
    max_tau_s = d_m / c

    stereo = np.stack([left, right], axis=1)

    taus: list[float] = []
    for _start, win in iter_windows(stereo, fs, window_s, hop_s):
        l_win = win[:, 0].astype(np.float64)
        r_win = win[:, 1].astype(np.float64)
        tau_s, _conf, _pmr = gcc_phat(
            l_win, r_win, fs,
            band_hz=band_hz,
            max_lag_s=max_tau_s,
            beta=beta,
        )
        taus.append(tau_s)

    if not taus:
        return {"tau0_s": 0.0, "mad_s": 0.0, "n_windows": 0}

    tau_arr = np.array(taus)
    tau0 = float(np.median(tau_arr))
    mad = float(np.median(np.abs(tau_arr - tau0)))
    return {"tau0_s": tau0, "mad_s": mad, "n_windows": len(taus)}


# ---------------------------------------------------------------------------
# fit_heading
# ---------------------------------------------------------------------------

def fit_heading(
    measurements: list[tuple[float, float, float, float, float, float]],
) -> dict:
    """Confidence-weighted circular mean of (true_azimuth - measured_theta).

    Parameters
    ----------
    measurements:
        List of (node_x, node_y, source_x, source_y, theta_deg, confidence).
        node_x/y and source_x/y are in metres; theta_deg is the measured
        bearing; confidence is the GCC-PHAT confidence weight.

    Returns
    -------
    dict with keys:
        heading_deg: float — fitted heading angle in [0, 360).
        residual_std_deg: float — circular standard deviation of residuals (deg).
    """
    if not measurements:
        return {"heading_deg": 0.0, "residual_std_deg": 0.0}

    sin_sum = 0.0
    cos_sum = 0.0
    total_w = 0.0

    for node_x, node_y, src_x, src_y, theta_deg, conf in measurements:
        # True azimuth of the source from the node (CCW from +x)
        true_az_deg = math.degrees(math.atan2(src_y - node_y, src_x - node_x))
        # heading = true_az - measured_theta  (mod 360)
        offset = true_az_deg - theta_deg
        offset_rad = math.radians(offset)
        w = max(conf, 1e-9)
        sin_sum += w * math.sin(offset_rad)
        cos_sum += w * math.cos(offset_rad)
        total_w += w

    heading_rad = math.atan2(sin_sum / total_w, cos_sum / total_w)
    heading_deg = math.degrees(heading_rad) % 360.0

    # Circular standard deviation of residuals
    residuals_rad = []
    for node_x, node_y, src_x, src_y, theta_deg, _ in measurements:
        true_az_deg = math.degrees(math.atan2(src_y - node_y, src_x - node_x))
        residual = (true_az_deg - theta_deg) - heading_deg
        # Wrap to [-180, 180]
        residual = ((residual + 180.0) % 360.0) - 180.0
        residuals_rad.append(math.radians(residual))

    if len(residuals_rad) >= 2:
        r_sin = np.mean([math.sin(r) for r in residuals_rad])
        r_cos = np.mean([math.cos(r) for r in residuals_rad])
        R = math.sqrt(r_sin ** 2 + r_cos ** 2)
        R = min(R, 1.0 - 1e-12)
        circ_std_rad = math.sqrt(-2.0 * math.log(R))
        residual_std_deg = math.degrees(circ_std_rad)
    else:
        residual_std_deg = 0.0

    return {
        "heading_deg": heading_deg,
        "residual_std_deg": residual_std_deg,
    }


# ---------------------------------------------------------------------------
# format_config_snippet
# ---------------------------------------------------------------------------

def format_config_snippet(
    node_id: int,
    tau0_s: float | None = None,
    heading_deg: float | None = None,
) -> str:
    """Return TOML lines to paste into site.toml for a node.

    NEVER writes site.toml automatically — its hash is stored with every
    trial and must remain under the user's control.

    Parameters
    ----------
    node_id:
        The node's node_id.
    tau0_s:
        Calibration time offset in seconds (optional).
    heading_deg:
        Node heading in degrees (optional).

    Returns
    -------
    str — TOML lines ready to paste.
    """
    lines: list[str] = [
        f"# Calibration snippet for node_id = {node_id}",
        f"# Paste into the [[nodes]] entry for node_id = {node_id}",
    ]
    if tau0_s is not None:
        lines.append(f"tau0_s = {tau0_s:.9f}  # seconds")
    if heading_deg is not None:
        lines.append(f"heading_deg = {heading_deg:.4f}  # degrees")
    return "\n".join(lines) + "\n"
