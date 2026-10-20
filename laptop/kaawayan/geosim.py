"""
kaawayan.geosim — End-to-end geometric simulation for multi-node localization.

Combines sim.make_stereo (acoustic simulation) with bearing estimation and
fusion to test the full pipeline from a known source position.

Known limitation: asin-based bearing is ambiguous front/back. A source at
true theta=150 deg (behind the node) is estimated as approximately theta=+30.
This is expected and documented; do not try to fix it here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kaawayan.bearing import estimate_bearing, speed_of_sound
from kaawayan.fusion import NodePose, bearing_line, fuse
from kaawayan.sim import make_stereo


# ---------------------------------------------------------------------------
# true_theta_deg
# ---------------------------------------------------------------------------

def true_theta_deg(source_xy: tuple[float, float], pose: NodePose) -> float:
    """Compute the true bearing from *pose* to *source_xy*.

    True azimuth of source from node = atan2(dy, dx) (world frame).
    Bearing theta = wrap180(azimuth - heading_deg).

    Parameters
    ----------
    source_xy:
        (x, y) of the source in metres.
    pose:
        NodePose with surveyed position and heading.

    Returns
    -------
    float
        Bearing angle in degrees, in [-180, 180).
        Positive = CCW from node front = source is toward M1 (left channel).

    Notes
    -----
    asin is used in bearing estimation, so values outside [-90, 90] will be
    mirrored by the estimator. This is a known limitation; tests for that
    path carry the docstring "known limitation".
    """
    dx = source_xy[0] - pose.x_m
    dy = source_xy[1] - pose.y_m
    azimuth_deg = math.degrees(math.atan2(dy, dx))
    theta = ((azimuth_deg - pose.heading_deg + 180.0) % 360.0) - 180.0
    return theta


# ---------------------------------------------------------------------------
# simulate_node
# ---------------------------------------------------------------------------

def simulate_node(
    source_xy: tuple[float, float],
    pose: NodePose,
    signal: NDArray[np.floating],
    fs: float,
    d_m: float,
    temp_c: float,
    snr_db: float,
    rng: np.random.Generator,
    echo: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a stereo recording at *pose* for a source at *source_xy*.

    Uses :func:`sim.make_stereo` with ``theta = true_theta_deg(source_xy, pose)``.
    sin() mirrors behind-node sources (theta outside [-90, 90]) — this is correct
    acoustic behaviour: a source 150 deg off-front sounds like one 30 deg off-front.

    Returns
    -------
    (left, right): float64 arrays of length len(signal).
    """
    theta = true_theta_deg(source_xy, pose)
    c_mps = speed_of_sound(temp_c)
    return make_stereo(signal, fs, theta, d_m, c_mps, snr_db, rng, echo=echo)


# ---------------------------------------------------------------------------
# localize
# ---------------------------------------------------------------------------

def localize(
    source_xy: tuple[float, float],
    poses: dict[int, NodePose],
    signals: dict[int, NDArray],
    fs: float,
    d_m: float,
    temp_c: float,
    band_hz: tuple[float, float],
    beta: float,
    snr_db: float,
    rng: np.random.Generator,
    gate_min_confidence: float,
    min_crossing_deg: float = 30.0,
    echo: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Full pipeline: source position -> per-node stereo -> bearing -> fusion.

    Parameters
    ----------
    source_xy:
        True (x, y) of the source.
    poses:
        Dict mapping node_id to NodePose.
    signals:
        Dict mapping node_id to mono source signal array (same length for all nodes;
        independent noise is added per channel inside make_stereo).
    fs:
        Sample rate in Hz.
    d_m:
        Mic spacing in metres.
    temp_c:
        Air temperature in Celsius.
    band_hz:
        (lo, hi) pass-band in Hz.
    beta:
        GCC-PHAT exponent.
    snr_db:
        Signal-to-noise ratio in dB.
    rng:
        NumPy random Generator (passed through to make_stereo for determinism).
    gate_min_confidence:
        Minimum confidence to include a bearing in fusion.
    min_crossing_deg:
        Minimum crossing angle for fusion to return "ok".
    echo:
        Optional (delay_s, gain) echo added to both channels.

    Returns
    -------
    dict with all fuse() output keys plus:
        "bearings": dict mapping node_id -> {"theta_deg": ..., "confidence": ...}
        "true_theta_deg": dict mapping node_id -> true bearing in degrees
    """
    bearing_results: dict[int, dict] = {}
    true_thetas: dict[int, float] = {}

    for node_id, pose in poses.items():
        sig = signals[node_id]
        left, right = simulate_node(source_xy, pose, sig, fs, d_m, temp_c, snr_db, rng, echo=echo)
        result = estimate_bearing(left, right, fs, d_m, temp_c, band_hz, beta=beta)
        bearing_results[node_id] = {
            "theta_deg": result["theta_deg"],
            "confidence": result["confidence"],
        }
        true_thetas[node_id] = true_theta_deg(source_xy, pose)

    bearing_list = [
        (node_id, br["theta_deg"], br["confidence"])
        for node_id, br in bearing_results.items()
    ]

    fuse_result = fuse(bearing_list, poses, min_crossing_deg=min_crossing_deg,
                       gate_min_confidence=gate_min_confidence)

    fuse_result["bearings"] = bearing_results
    fuse_result["true_theta_deg"] = true_thetas
    return fuse_result
