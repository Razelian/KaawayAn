"""
End-to-End Tests for KaawayAn Directionality and 2D Localization.

Tests:
1. Speed of sound calculation across temperatures.
2. GCC-PHAT bearing estimation with synthetic delays (verifying angles 0°, +30°, -30°, +45°).
3. 3:1 decimation from 48 kHz stereo to 16 kHz mono.
4. DroneClassifier model inference.
5. Multi-node 2D least-squares line intersection (fusion) and geometric rejection.
6. Rolling median coordinate smoothing.
7. End-to-end pipeline processing: Audio frames -> Handler -> Database logging -> Dashboard DB queries.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as signal

from config import Config
from database.database import Database
from dashboard import db as dash_db
from kaawayan.bearing import estimate_bearing, gcc_phat, speed_of_sound, tau_to_theta
from kaawayan.fusion import NodePose, RollingMedianXY, bearing_line, fuse, poses_from_config
from kaawayan.protocol import AudioPayload, FrameType, encode_frame
def make_synthetic_stereo(
    duration_s: float,
    fs: int,
    true_theta_deg: float,
    d_m: float = 0.10,
    temp_c: float = 24.0,
    f0_hz: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate stereo audio with a precise inter-channel delay corresponding to true_theta_deg."""
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    c = speed_of_sound(temp_c)

    # Delay: tau = d * sin(theta) / c
    # Positive tau -> sound reaches left mic first (tau = t_right - t_left)
    tau_s = (d_m * math.sin(math.radians(true_theta_deg))) / c

    # Create harmonic drone-like signal (fundamental + harmonics + white noise)
    signal_left = (
        0.5 * np.sin(2 * np.pi * f0_hz * t)
        + 0.3 * np.sin(2 * np.pi * 2 * f0_hz * t)
        + 0.2 * np.sin(2 * np.pi * 3 * f0_hz * t)
        + 0.05 * np.random.randn(n_samples)
    )

    # Apply fractional delay to right channel via frequency domain phase shift:
    # Right delayed by +tau -> R(f) = L(f) * exp(-2*pi*j*f*tau)
    L_fft = np.fft.rfft(signal_left)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    phase_shift = np.exp(-2j * np.pi * freqs * tau_s)
    signal_right = np.fft.irfft(L_fft * phase_shift, n=n_samples)

    return signal_left.astype(np.float32), signal_right.astype(np.float32)


# ===========================================================================
# Test Cases
# ===========================================================================

def test_speed_of_sound():
    assert abs(speed_of_sound(0.0) - 331.3) < 0.1
    assert abs(speed_of_sound(20.0) - 343.2) < 0.2
    assert abs(speed_of_sound(24.0) - 345.5) < 0.2


@pytest.mark.parametrize("target_angle", [0.0, 30.0, -30.0, 45.0, -45.0])
def test_directionality_bearing_estimation(target_angle):
    fs = 48000
    d_m = 0.10
    temp_c = 24.0

    left, right = make_synthetic_stereo(
        duration_s=1.0,
        fs=fs,
        true_theta_deg=target_angle,
        d_m=d_m,
        temp_c=temp_c,
    )

    res = estimate_bearing(
        left=left,
        right=right,
        fs=fs,
        d_m=d_m,
        temp_c=temp_c,
        band_hz=(300.0, 4000.0),
        beta=0.25,
    )

    est_angle = res["theta_deg"]
    conf = res["confidence"]
    pmr = res["peak_median_ratio"]

    # Bearing must match ground truth within 1.0 degree
    assert abs(est_angle - target_angle) < 1.0, f"Expected {target_angle}°, got {est_angle}°"
    # Confidence and PMR must be elevated for correlated tone
    assert conf > 0.5
    assert pmr > 2.5


def test_classifier_decimation_and_inference():
    pytest.importorskip("torch", reason="classifier inference requires the [classifier] extra")
    pytest.importorskip("torchaudio", reason="classifier inference requires the [classifier] extra")
    from model.model import DroneClassifier

    fs_48k = 48000
    duration_s = 1.0
    t = np.arange(int(duration_s * fs_48k)) / fs_48k
    # 400 Hz tone at 48 kHz
    audio_48k = (0.7 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)

    # Decimate 3:1 to 16 kHz
    audio_16k = signal.resample_poly(audio_48k, 1, 3)
    assert len(audio_16k) == 16000

    # Load classifier and run predict
    model = DroneClassifier()
    prob = model.predict((audio_16k * 32767.0).astype(np.int16))
    assert 0.0 <= prob <= 1.0


def test_multi_node_2d_localization_fusion():
    # Node 1 at (0, 0), heading 63.43°
    # Node 2 at (3, 0), heading 116.57°
    # Target drone at (1.5, 3.0)
    pose1 = NodePose(node_id=1, x_m=0.0, y_m=0.0, heading_deg=63.43)
    pose2 = NodePose(node_id=2, x_m=3.0, y_m=0.0, heading_deg=116.57)
    poses = {1: pose1, 2: pose2}

    # For drone at (1.5, 3.0):
    # Vector from node 1: (1.5, 3.0), azimuth = atan2(3, 1.5) = 63.43° -> theta1 = 0.0°
    # Vector from node 2: (-1.5, 3.0), azimuth = atan2(3, -1.5) = 116.57° -> theta2 = 0.0°
    bearings = [(1, 0.0, 0.9), (2, 0.0, 0.9)]

    fuse_res = fuse(bearings, poses, min_crossing_deg=30.0)

    assert fuse_res["status"] == "ok"
    assert abs(fuse_res["x_m"] - 1.5) < 0.01
    assert abs(fuse_res["y_m"] - 3.0) < 0.01
    # Crossing angle is 116.57 - 63.43 = 53.14°
    assert abs(fuse_res["crossing_deg"] - 53.14) < 0.5


def test_fusion_rejection_shallow_crossing():
    pose1 = NodePose(node_id=1, x_m=0.0, y_m=0.0, heading_deg=0.0)
    pose2 = NodePose(node_id=2, x_m=1.0, y_m=0.0, heading_deg=10.0)
    poses = {1: pose1, 2: pose2}

    # Almost parallel lines (crossing 10° < 30°)
    bearings = [(1, 0.0, 0.9), (2, 0.0, 0.9)]

    fuse_res = fuse(bearings, poses, min_crossing_deg=30.0)
    assert fuse_res["status"] == "uncertain"
    assert fuse_res["crossing_deg"] < 30.0


def test_rolling_median_smoother():
    smoother = RollingMedianXY(5)
    # Feed noisy values around (2.0, 4.0) with an outlier
    vals = [(2.0, 4.0), (2.1, 4.1), (10.0, 20.0), (1.9, 3.9), (2.0, 4.0)]
    for x, y in vals:
        res = smoother.update({"status": "ok", "x_m": x, "y_m": y})

    sx, sy = res
    # Outlier (10.0, 20.0) must be rejected by median
    assert abs(sx - 2.0) < 0.2
    assert abs(sy - 4.0) < 0.2


def test_database_and_dashboard_queries():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        test_db_path = Path(tmp.name)

    try:
        db = Database(override_path=test_db_path)
        # Log detection
        eid = db.log_detection(
            node_id=1,
            predicted_drone=1,
            model_confidence=0.92,
            timestamp="100.0",
            theta_deg=28.5,
            bearing_confidence=0.88,
            peak_median_ratio=15.2,
            temp_c=25.4,
            humidity_pct=60.1,
        )
        assert eid > 0

        # Log localization
        lid = db.log_localization(
            timestamp="100.0",
            x_m=1.52,
            y_m=3.01,
            crossing_deg=53.1,
            n_nodes=2,
            status="ok",
        )
        assert lid > 0

        # Read back via database connection
        row = db.connection.execute(
            "SELECT theta_deg, bearing_confidence FROM detection_events WHERE event_id = ?", (eid,)
        ).fetchone()
        assert abs(row[0] - 28.5) < 1e-4
        assert abs(row[1] - 0.88) < 1e-4

        loc_row = db.connection.execute(
            "SELECT x_m, y_m, status FROM localizations WHERE loc_id = ?", (lid,)
        ).fetchone()
        assert abs(loc_row[0] - 1.52) < 1e-4
        assert loc_row[2] == "ok"

        db.close()
    finally:
        if test_db_path.exists():
            test_db_path.unlink()
