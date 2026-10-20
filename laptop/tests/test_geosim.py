"""
Tests for geosim.py — Slice 6a.

Test parameters (per spec):
  d = 0.10 m, 20°C, 48 kHz, 1 s white noise, SNR 20 dB
  band (300, 1500), beta 0.25, gate 0.3, 5 seeds

Tests:
1. true_theta_deg for node (0,0,90) and source (-1,1) = +45 deg.
2. Mirror test (known limitation): source at true theta=150 estimated within
   1.5 deg of +30 deg.
3. Two nodes (0,0,63.4) and (3,0,116.6); sources (1.5,3.0), (0.5,2.0), (2.0,3.5):
   median position error < 0.25 m.
4. Add node (1.5,6.0,270): median error < 0.25 m.
5. Dead node (given independent noise instead of source) falls below gate,
   n_lines == 2, status ok.
6. Same seed gives identical output twice.
"""

import math
import numpy as np
import pytest

from kaawayan.geosim import true_theta_deg, simulate_node, localize
from kaawayan.fusion import NodePose, fuse
from kaawayan.bearing import speed_of_sound

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FS = 48_000
D = 0.10          # mic spacing, metres
T_C = 20.0
C = speed_of_sound(T_C)
BAND = (300, 1500)
BETA = 0.25
SNR_DB = 20.0
GATE = 0.3
N = FS * 1        # 1 s


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _white(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(N)


def _pose(node_id: int, x: float, y: float, heading: float) -> NodePose:
    return NodePose(node_id=node_id, x_m=x, y_m=y, heading_deg=heading)


POSES_2 = {
    1: _pose(1, 0.0, 0.0, 63.4),
    2: _pose(2, 3.0, 0.0, 116.6),
}
POSES_3 = {
    1: _pose(1, 0.0, 0.0, 63.4),
    2: _pose(2, 3.0, 0.0, 116.6),
    3: _pose(3, 1.5, 6.0, 270.0),
}
SOURCES = [(1.5, 3.0), (0.5, 2.0), (2.0, 3.5)]
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# true_theta_deg
# ---------------------------------------------------------------------------

class TestTrueThetaDeg:
    def test_node_0_0_90_source_minus1_plus1(self):
        """Node at (0,0) heading=90, source at (-1,1).
        Azimuth = atan2(1, -1) = 135 deg.
        theta = 135 - 90 = 45 deg (CCW = toward M1).
        """
        pose = _pose(1, 0.0, 0.0, 90.0)
        theta = true_theta_deg((-1.0, 1.0), pose)
        assert theta == pytest.approx(45.0, abs=1e-10)

    def test_source_directly_ahead_gives_zero(self):
        """Source directly along heading gives theta=0."""
        pose = _pose(1, 0.0, 0.0, 90.0)
        # Source at (0, 5) — directly ahead of heading=90
        theta = true_theta_deg((0.0, 5.0), pose)
        assert theta == pytest.approx(0.0, abs=1e-10)

    def test_source_directly_behind_gives_plus_or_minus_180(self):
        """Source directly behind gives theta=180 or -180."""
        pose = _pose(1, 0.0, 0.0, 90.0)
        # Source at (0, -5) — directly behind
        theta = true_theta_deg((0.0, -5.0), pose)
        assert abs(theta) == pytest.approx(180.0, abs=1e-10)

    def test_wrap180_correctness(self):
        """Wrap is in [-180, 180)."""
        pose = _pose(1, 0.0, 0.0, 350.0)
        # Source at (1, 0): azimuth=0, theta = 0 - 350 + 360 = 10 deg
        theta = true_theta_deg((1.0, 0.0), pose)
        assert theta == pytest.approx(10.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Mirror test (known limitation)
# ---------------------------------------------------------------------------

class TestMirrorKnownLimitation:
    """Known limitation: asin cannot distinguish front from back.

    A source at true theta=150 deg (behind the node front) appears at ~30 deg
    to the estimator (sin(150)=sin(30)=0.5). This is correct acoustic behaviour
    and NOT a bug. Tests here document this limitation.
    """

    def test_source_at_150_estimated_near_30(self):
        """Known limitation: source at true theta=150 deg estimated within 1.5 deg of +30.
        
        sin(150) = sin(30) = 0.5, so the estimator will return approximately 30 deg.
        This is the asin ambiguity and is a known, documented limitation.
        """
        # Create a node facing +x (heading=0), place source at theta=150 deg
        # (bearing from front = 150 = behind-left)
        # True tau = d*sin(150)/c = d*0.5/c — same as theta=30.
        pose = _pose(1, 0.0, 0.0, 0.0)
        # theta=150 → azimuth = 0 + 150 = 150 deg → source at (cos150, sin150) = (-√3/2, 0.5)
        errors = []
        for seed in SEEDS:
            rng = _rng(seed + 200)
            sig = _white(rng)
            left, right = simulate_node((-math.sqrt(3)/2, 0.5), pose, sig, FS, D, T_C, SNR_DB, rng)
            from kaawayan.bearing import estimate_bearing
            result = estimate_bearing(left, right, FS, D, T_C, BAND, beta=BETA)
            # The estimator returns approximately +30, not +150
            estimated = result["theta_deg"]
            errors.append(abs(estimated - 30.0))

        median_err = float(np.median(errors))
        assert median_err < 1.5, (
            f"Mirror test: source at theta=150 should estimate near 30; "
            f"median error={median_err:.2f} deg"
        )


# ---------------------------------------------------------------------------
# Two-node localization
# ---------------------------------------------------------------------------

class TestLocalizeTwo:
    def _run(self, source_xy, seed):
        rng = _rng(seed)
        signals = {nid: _white(rng) for nid in POSES_2}
        return localize(
            source_xy, POSES_2, signals, FS, D, T_C, BAND, BETA, SNR_DB, rng,
            gate_min_confidence=GATE,
        )

    def test_median_position_error_below_0p25_m(self):
        errors = []
        for source_xy in SOURCES:
            seed_errors = []
            for seed in SEEDS:
                result = self._run(source_xy, seed)
                if result["status"] == "ok":
                    err = math.hypot(result["x_m"] - source_xy[0],
                                     result["y_m"] - source_xy[1])
                    seed_errors.append(err)
            if seed_errors:
                errors.append(float(np.median(seed_errors)))

        assert errors, "No successful localizations"
        overall_median = float(np.median(errors))
        assert overall_median < 0.25, (
            f"2-node median position error = {overall_median:.3f} m (threshold 0.25 m)"
        )

    def test_per_source_errors(self):
        """Report per-source median and p95 errors (recorded for PROGRESS.md)."""
        for source_xy in SOURCES:
            errors = []
            for seed in SEEDS:
                result = self._run(source_xy, seed)
                if result["status"] == "ok":
                    err = math.hypot(result["x_m"] - source_xy[0],
                                     result["y_m"] - source_xy[1])
                    errors.append(err)
            if errors:
                med = float(np.median(errors))
                p95 = float(np.percentile(errors, 95))
                print(f"\n2-node source={source_xy}: median={med:.3f} m, p95={p95:.3f} m")


# ---------------------------------------------------------------------------
# Three-node localization
# ---------------------------------------------------------------------------

class TestLocalizeThree:
    def _run(self, source_xy, seed):
        rng = _rng(seed)
        signals = {nid: _white(rng) for nid in POSES_3}
        return localize(
            source_xy, POSES_3, signals, FS, D, T_C, BAND, BETA, SNR_DB, rng,
            gate_min_confidence=GATE,
        )

    def test_median_position_error_below_0p25_m(self):
        errors = []
        for source_xy in SOURCES:
            seed_errors = []
            for seed in SEEDS:
                result = self._run(source_xy, seed)
                if result["status"] == "ok":
                    err = math.hypot(result["x_m"] - source_xy[0],
                                     result["y_m"] - source_xy[1])
                    seed_errors.append(err)
            if seed_errors:
                errors.append(float(np.median(seed_errors)))

        assert errors, "No successful 3-node localizations"
        overall_median = float(np.median(errors))
        assert overall_median < 0.25, (
            f"3-node median position error = {overall_median:.3f} m (threshold 0.25 m)"
        )

    def test_per_source_errors(self):
        """Report per-source median and p95 errors."""
        for source_xy in SOURCES:
            errors = []
            for seed in SEEDS:
                result = self._run(source_xy, seed)
                if result["status"] == "ok":
                    err = math.hypot(result["x_m"] - source_xy[0],
                                     result["y_m"] - source_xy[1])
                    errors.append(err)
            if errors:
                med = float(np.median(errors))
                p95 = float(np.percentile(errors, 95))
                print(f"\n3-node source={source_xy}: median={med:.3f} m, p95={p95:.3f} m")


# ---------------------------------------------------------------------------
# Dead node falls below gate
# ---------------------------------------------------------------------------

class TestDeadNode:
    def test_dead_node_falls_below_gate(self):
        """Node 3 receives only independent noise (dead), so its confidence
        drops below gate=0.3, is dropped, and fuse uses only 2 lines -> ok."""
        source_xy = (1.5, 3.0)

        errors = []
        for seed in SEEDS:
            rng = _rng(seed + 300)
            # Normal signals for nodes 1 and 2
            signals: dict[int, np.ndarray] = {}
            sig = _white(rng)
            signals[1] = sig.copy()
            signals[2] = sig.copy()
            # Node 3 gets pure independent noise instead of the source
            signals[3] = rng.standard_normal(N)

            # Manually build bearing list
            from kaawayan.bearing import estimate_bearing
            bearing_list = []
            for node_id, pose in POSES_3.items():
                s = signals[node_id]
                from kaawayan.geosim import simulate_node
                left, right = simulate_node(source_xy, pose, s, FS, D, T_C, SNR_DB, rng)
                # For the dead node, override with pure noise
                if node_id == 3:
                    n_noise = len(s)
                    left = rng.standard_normal(n_noise)
                    right = rng.standard_normal(n_noise)
                result = estimate_bearing(left, right, FS, D, T_C, BAND, beta=BETA)
                bearing_list.append((node_id, result["theta_deg"], result["confidence"]))

            fuse_result = fuse(bearing_list, POSES_3, gate_min_confidence=GATE)

            # Dead node should be gated out
            assert 3 not in fuse_result["used_node_ids"], (
                f"Seed {seed}: dead node 3 should have been gated out"
            )
            assert fuse_result["n_lines"] == 2, (
                f"Seed {seed}: expected 2 lines, got {fuse_result['n_lines']}"
            )
            if fuse_result["status"] == "ok":
                err = math.hypot(
                    fuse_result["x_m"] - source_xy[0],
                    fuse_result["y_m"] - source_xy[1],
                )
                errors.append(err)

        assert errors, "No successful localizations with dead node"


# ---------------------------------------------------------------------------
# Determinism: same seed gives identical output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_gives_identical_output(self):
        source_xy = (1.5, 3.0)

        def _run():
            rng = _rng(42)
            signals = {nid: _white(rng) for nid in POSES_2}
            return localize(
                source_xy, POSES_2, signals, FS, D, T_C, BAND, BETA, SNR_DB, rng,
                gate_min_confidence=GATE,
            )

        r1 = _run()
        r2 = _run()
        assert r1["status"] == r2["status"]
        assert r1["x_m"] == r2["x_m"]
        assert r1["y_m"] == r2["y_m"]
        assert r1["crossing_deg"] == r2["crossing_deg"]
        # Check bearings too
        for nid in POSES_2:
            assert r1["bearings"][nid]["theta_deg"] == r2["bearings"][nid]["theta_deg"]
            assert r1["bearings"][nid]["confidence"] == r2["bearings"][nid]["confidence"]
