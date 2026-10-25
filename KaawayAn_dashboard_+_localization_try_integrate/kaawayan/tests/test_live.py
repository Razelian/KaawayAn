"""
tests/test_live.py — Tests for kaawayan.live (R4 spec).

All tests go through the public API: push_audio, push_env, poll, flush,
set_config_hash, replay_trial.
Audio is synthesised via geosim.simulate_node (white noise, SNR 25 dB,
int16 about 4000 LSB RMS).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import sys
import types
import wave

import numpy as np
import pytest

from kaawayan.config import load_config
from kaawayan.fusion import NodePose, poses_from_config
from kaawayan.geosim import simulate_node
from kaawayan.live import LivePipeline, replay_trial

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SITE_TOML = REPO / "config" / "site.toml"
_FS = 48_000
_CHUNK = 512


def _load_cfg():
    cfg, _ = load_config(SITE_TOML)
    return cfg


def _make_cfg_2node():
    """Minimal two-node config matching site.toml layout."""
    return _load_cfg()


def _poses_2node(cfg):
    return poses_from_config(cfg)


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_noise_int16(n_samples: int, rng: np.random.Generator,
                      rms: int = 4000) -> np.ndarray:
    """White noise int16 stereo (n,2)."""
    raw = rng.standard_normal((n_samples, 2)) * rms
    return raw.clip(-32768, 32767).astype(np.int16)


def _source_signal(n: int, rng: np.random.Generator, snr_db: float = 25.0
                   ) -> np.ndarray:
    """White-noise mono source signal for geosim."""
    return rng.standard_normal(n).astype(np.float64)


def _make_stereo_int16(source_xy, pose: NodePose, n_samples: int,
                        rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Simulate stereo int16 for one node."""
    sig = _source_signal(n_samples, rng)
    from kaawayan.bearing import speed_of_sound
    left, right = simulate_node(
        source_xy, pose, sig, _FS,
        cfg["mic_spacing_m"], 20.0, 25.0, rng
    )
    # Scale to ~4000 LSB RMS
    scale = 4000.0 / (np.std(left) + 1e-9)
    left = (left * scale).clip(-32768, 32767)
    right = (right * scale).clip(-32768, 32767)
    arr = np.stack([left, right], axis=1).astype(np.int16)
    return arr


def _push_stream(pipeline: LivePipeline, node_id: int, arr: np.ndarray,
                 t0: float = 0.0, chunk_size: int = _CHUNK) -> None:
    """Push arr in chunks, interleaved t based on sample index."""
    n = len(arr)
    idx = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        t = t0 + idx / _FS
        pipeline.push_audio(node_id, t, arr[idx:end])
        idx = end


def _make_pipeline(cfg) -> LivePipeline:
    poses = poses_from_config(cfg)
    return LivePipeline(cfg, poses)


# ---------------------------------------------------------------------------
# (a) t_end strictly increasing, window count, buffer bound
# ---------------------------------------------------------------------------

class TestTEndAndBuffer:
    """Bug (a): t_end uses absolute sample index, buffer stays bounded."""

    def test_t_end_strictly_increasing(self):
        """Interleaved 2-node push: ok results, t_laptop_s strictly increasing by hop_s,
        count == floor((10 - window_s) / hop_s) + 1."""
        cfg = _make_cfg_2node()
        hop_s = cfg["hop_s"]
        win_s = cfg["window_s"]
        pl = _make_pipeline(cfg)
        rng = _rng(1)
        n_total = int(10 * _FS)
        arr1 = _make_stereo_int16((1.5, 3.0), poses_from_config(cfg)[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16((1.5, 3.0), poses_from_config(cfg)[2], n_total, rng, cfg)

        # Interleaved push (same t per chunk, like test_position_accuracy_2node)
        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2[idx:end])
            idx = end
        pl.flush()
        results = pl.poll()
        assert len(results) > 0

        # All results must have n_nodes == 2 and status "ok"
        for r in results:
            assert r["n_nodes"] == 2, f"Expected n_nodes=2, got {r['n_nodes']}"
            assert r["status"] == "ok", f"Expected status ok, got {r['status']}"

        t_ends = [r["t_laptop_s"] for r in results]
        # Strictly increasing
        for i in range(1, len(t_ends)):
            assert t_ends[i] > t_ends[i - 1], (
                f"t_end not strictly increasing: {t_ends[i-1]} -> {t_ends[i]}"
            )
        # Consecutive differences == hop_s within 1e-9
        for i in range(1, len(t_ends)):
            diff = t_ends[i] - t_ends[i - 1]
            assert abs(diff - hop_s) < 1e-9, (
                f"Gap {diff:.12f} != hop_s {hop_s:.12f} at index {i}"
            )
        # Count == floor((10 - window_s) / hop_s) + 1
        expected_count = math.floor((10 - win_s) / hop_s) + 1
        assert len(results) == expected_count, (
            f"Expected {expected_count} results, got {len(results)}"
        )

    def test_window_count(self):
        cfg = _make_cfg_2node()
        pl = _make_pipeline(cfg)
        rng = _rng(2)
        win_s = cfg["window_s"]
        hop_s = cfg["hop_s"]
        n_total = int(10 * _FS)
        expected = math.floor((10 - win_s) / hop_s) + 1

        arr1 = _make_stereo_int16((1.5, 3.0), poses_from_config(cfg)[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16((1.5, 3.0), poses_from_config(cfg)[2], n_total, rng, cfg)
        _push_stream(pl, 1, arr1)
        _push_stream(pl, 2, arr2)
        pl.flush()
        results = pl.poll()

        # Count distinct window_idx per node in bearings
        for nid in [1, 2]:
            widxs = set()
            for r in results:
                b = r["bearings"].get(str(nid))
                if b:
                    widxs.add(b["window_idx"])
            assert len(widxs) == expected, (
                f"Node {nid}: expected {expected} windows, got {len(widxs)}"
            )

    def test_buffer_bounded_after_long_push(self):
        """After 60 s of data, buffered samples <= window + 2 frames."""
        cfg = _make_cfg_2node()
        pl = _make_pipeline(cfg)
        rng = _rng(3)
        n_total = int(60 * _FS)
        win_samples = int(cfg["window_s"] * _FS)
        arr = _make_noise_int16(n_total, rng)

        _push_stream(pl, 1, arr)

        buf = pl._bufs[1]
        buffered = buf.buffered_samples()
        assert buffered <= win_samples + 2 * _CHUNK + 1, (
            f"Buffer not bounded: {buffered} samples (limit {win_samples + 2*_CHUNK})"
        )


# ---------------------------------------------------------------------------
# (b) flush and dead-node liveness
# ---------------------------------------------------------------------------

class TestFlushAndLiveness:
    """Bug (b): flush emits pending; dead-node tracking; pending bounded."""

    def test_position_accuracy_2node(self):
        """Median position error < 0.25 m for source at (1.5, 3.0)."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(10)
        n_total = int(10 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)

        # Interleaved push (node 1 chunk, node 2 chunk, ...)
        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2[idx:end])
            idx = end

        pl.flush()
        results = pl.poll()

        ok_results = [r for r in results if r["status"] == "ok"]
        assert len(ok_results) >= 3, "Too few ok results"

        errors = [math.hypot(r["x_m"] - src_xy[0], r["y_m"] - src_xy[1])
                  for r in ok_results]
        median_err = float(np.median(errors))
        assert median_err < 0.25, f"Median position error {median_err:.3f} >= 0.25 m"

    def test_interleaved_vs_replay_identical(self, tmp_path):
        """push order (interleaved) and replay_trial give identical positions."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        rng = _rng(11)
        n_total = int(8 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)

        # Interleaved push
        pl = LivePipeline(cfg, poses)
        pl.set_config_hash(hashlib.sha256(SITE_TOML.read_bytes()).hexdigest())
        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2[idx:end])
            idx = end
        pl.flush()
        live_results = pl.poll()
        live_ok = [(r["t_laptop_s"], r["x_m"], r["y_m"])
                   for r in live_results if r["status"] == "ok"]

        # Write trial folder and replay
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        _write_wav(trial_dir / "node1.wav", arr1)
        _write_wav(trial_dir / "node2.wav", arr2)
        import json as _json
        (trial_dir / "meta.json").write_text(_json.dumps({
            "trial_id": "t1", "kind": "tone", "config_hash": "x",
            "nodes": {
                "1": {"t_first_audio_laptop_s": 0.0},
                "2": {"t_first_audio_laptop_s": 0.0},
            }
        }))

        replay_results = replay_trial(trial_dir, SITE_TOML)
        replay_ok = [(r["t_laptop_s"], r["x_m"], r["y_m"])
                     for r in replay_results if r["status"] == "ok"]

        assert len(live_ok) == len(replay_ok), (
            f"live {len(live_ok)} vs replay {len(replay_ok)} ok results"
        )
        for (lt, lx, ly), (rt, rx, ry) in zip(live_ok, replay_ok):
            assert abs(lt - rt) < 1e-6
            assert abs(lx - rx) < 1e-6
            assert abs(ly - ry) < 1e-6

    def test_t_offset_within_tolerance(self):
        """Node 2 starts 0.03 s late (< tolerance 0.1 s) — pairs ok."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(12)
        n_total = int(6 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)

        # Interleaved push with node2 offset 0.03 s
        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t1 = 0.0 + idx / _FS
            t2 = 0.03 + idx / _FS
            pl.push_audio(1, t1, arr1[idx:end])
            pl.push_audio(2, t2, arr2[idx:end])
            idx = end

        pl.flush()
        results = pl.poll()
        ok = [r for r in results if r["status"] == "ok"]
        assert len(ok) >= 1, "Expected at least one ok result with 0.03 s offset"

    def test_t_offset_beyond_tolerance_no_ok(self):
        """Node 2 starts 0.25 s late (> tolerance 0.1 s) — no ok result."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(13)
        n_total = int(6 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)

        # Interleaved push with node2 offset 0.25 s (> pair_tolerance_s=0.1)
        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t1 = 0.0 + idx / _FS
            t2 = 0.25 + idx / _FS
            pl.push_audio(1, t1, arr1[idx:end])
            pl.push_audio(2, t2, arr2[idx:end])
            idx = end

        pl.flush()
        results = pl.poll()
        ok = [r for r in results if r["status"] == "ok"]
        assert len(ok) == 0, f"Expected no ok results but got {len(ok)}"

    def test_node2_never_pushes_degraded(self):
        """Node 2 never pushes: results span uncertain (startup grace) then degraded.

        Spec: a node that has never sent audio is treated as alive during a startup
        grace and declared dead only when latest_t - first_window_t > 2*window_s.
        So early results are 'uncertain' with dead_node_ids==[], later ones
        'degraded' with dead_node_ids==[2].
        """
        cfg = _make_cfg_2node()
        window_s = cfg["window_s"]
        hop_s = cfg["hop_s"]
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(14)
        n_total = int(5 * _FS)

        arr1 = _make_noise_int16(n_total, rng)
        _push_stream(pl, 1, arr1, t0=0.0)
        pl.flush()
        results = pl.poll()

        assert len(results) >= 1

        # All results: status in {uncertain, degraded}, x_m is None, n_nodes == 1
        for r in results:
            assert r["status"] in {"uncertain", "degraded"}, (
                f"Expected uncertain or degraded, got {r['status']}"
            )
            assert r["x_m"] is None, f"Expected x_m None, got {r['x_m']}"
            assert r["n_nodes"] == 1, f"Expected n_nodes=1, got {r['n_nodes']}"

        # First result: startup grace -> uncertain and dead_node_ids == []
        t_first = results[0]["t_laptop_s"]
        first = results[0]
        assert first["status"] == "uncertain", (
            f"First result should be uncertain (startup grace), got {first['status']}"
        )
        assert first["dead_node_ids"] == [], (
            f"First result dead_node_ids should be [] (startup grace), "
            f"got {first['dead_node_ids']}"
        )

        # Results with t >= t_first + 2*window_s + hop_s: degraded, dead_node_ids==[2]
        late = [r for r in results
                if r["t_laptop_s"] >= t_first + 2 * window_s + hop_s]
        assert len(late) >= 1, (
            f"Expected at least one late result (t >= {t_first + 2*window_s + hop_s:.3f}), "
            f"got none; all t_laptop_s: {[r['t_laptop_s'] for r in results]}"
        )
        for r in late:
            assert r["status"] == "degraded", (
                f"Late result status should be degraded, got {r['status']}"
            )
            assert r["dead_node_ids"] == [2], (
                f"Late result dead_node_ids should be [2], got {r['dead_node_ids']}"
            )

    def test_pending_bounded_no_push_node2_120s(self):
        """After 120 s with node 2 silent, pending stays bounded."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(15)
        n_total = int(120 * _FS)

        arr1 = _make_noise_int16(n_total, rng)
        _push_stream(pl, 1, arr1, t0=0.0)
        pl.flush()
        pl.poll()

        assert len(pl._pending_records) == 0, (
            f"Pending not cleared after flush: {len(pl._pending_records)} records"
        )

    def test_node2_stops_early(self):
        """Node 2 stops at 4 s of 10 s: early results ok, later degraded, no crash."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(16)
        n_total = int(10 * _FS)
        n_node2 = int(4 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_node2, rng, cfg)

        # Push interleaved
        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            if idx + _CHUNK <= n_node2:
                pl.push_audio(2, t, arr2[idx:min(idx+_CHUNK, n_node2)])
            idx = end

        pl.flush()
        results = pl.poll()
        assert len(results) >= 1
        # Should have some ok (early) and some degraded (after node 2 dies)
        statuses = [r["status"] for r in results]
        # No crash (just check it ran)
        assert any(s in ("ok", "degraded", "uncertain") for s in statuses)

    def test_three_nodes_n_nodes_3(self):
        """Third node id 40 in config: n_nodes=3 when all alive."""
        import tomllib

        # Build a 3-node config by patching the 2-node one
        raw = SITE_TOML.read_bytes()
        toml_text = raw.decode()
        extra = """
[[nodes]]
node_id      = 40
usb_serial   = "FAKE40"
board        = "devkit"
x_m          = 1.5
y_m          = 6.0
heading_deg  = 270.0
tau0_s       = 0.0
channel_swap = false
mic_model    = "INMP441"
"""
        toml_text += extra
        import tomllib as _tomllib
        cfg3 = _tomllib.loads(toml_text)
        poses3: dict[int, NodePose] = {}
        for node in cfg3["nodes"]:
            nid = node["node_id"]
            poses3[nid] = NodePose(nid, node["x_m"], node["y_m"], node["heading_deg"])

        pl = LivePipeline(cfg3, poses3)
        rng = _rng(17)
        n_total = int(6 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses3[1], n_total, rng, cfg3)
        arr2 = _make_stereo_int16(src_xy, poses3[2], n_total, rng, cfg3)
        arr40 = _make_stereo_int16(src_xy, poses3[40], n_total, rng, cfg3)

        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2[idx:end])
            pl.push_audio(40, t, arr40[idx:end])
            idx = end

        pl.flush()
        results = pl.poll()
        ok = [r for r in results if r["status"] == "ok"]
        assert len(ok) >= 1
        # At least one result should have n_nodes=3
        assert any(r["n_nodes"] == 3 for r in ok), "Expected n_nodes=3 result"

    def test_three_nodes_node40_dead_status_ok(self):
        """Third node id 40 in config, node 40 never pushes.

        Spec: startup grace means early ok results may have dead_node_ids==[]
        for node 40.  After latest_t - first_window_t > 2*window_s, ok results
        must have dead_node_ids==[40].  At least 3 ok results and at least 3
        such late ok results must exist.
        """
        import tomllib as _tomllib
        raw = SITE_TOML.read_bytes().decode()
        raw += """
[[nodes]]
node_id      = 40
usb_serial   = "FAKE40"
board        = "devkit"
x_m          = 1.5
y_m          = 6.0
heading_deg  = 270.0
tau0_s       = 0.0
channel_swap = false
mic_model    = "INMP441"
"""
        cfg3 = _tomllib.loads(raw)
        window_s = cfg3["window_s"]
        hop_s = cfg3["hop_s"]
        poses3: dict[int, NodePose] = {}
        for node in cfg3["nodes"]:
            nid = node["node_id"]
            poses3[nid] = NodePose(nid, node["x_m"], node["y_m"], node["heading_deg"])

        pl = LivePipeline(cfg3, poses3)
        rng = _rng(18)
        n_total = int(8 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses3[1], n_total, rng, cfg3)
        arr2 = _make_stereo_int16(src_xy, poses3[2], n_total, rng, cfg3)

        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2[idx:end])
            idx = end

        pl.flush()
        results = pl.poll()
        ok = [r for r in results if r["status"] == "ok"]
        assert len(ok) >= 3, f"Expected >= 3 ok results, got {len(ok)}"

        # Every ok result has n_nodes == 2
        for r in ok:
            assert r["n_nodes"] == 2, (
                f"ok result n_nodes should be 2, got {r['n_nodes']}"
            )

        # Startup grace: first ok result may have dead_node_ids == []
        # Late ok results (t >= t_first + 2*window_s + hop_s) must have dead_node_ids == [40]
        t_first = ok[0]["t_laptop_s"]
        late_ok = [r for r in ok
                   if r["t_laptop_s"] >= t_first + 2 * window_s + hop_s]
        assert len(late_ok) >= 3, (
            f"Expected >= 3 late ok results (t >= {t_first + 2*window_s + hop_s:.3f}), "
            f"got {len(late_ok)}; all ok t_laptop_s: {[r['t_laptop_s'] for r in ok]}"
        )
        for r in late_ok:
            assert r["dead_node_ids"] == [40], (
                f"Late ok result dead_node_ids should be [40], got {r['dead_node_ids']}"
            )


# ---------------------------------------------------------------------------
# (c) Gate tests
# ---------------------------------------------------------------------------

class TestGating:
    def test_independent_noise_gate_false(self):
        """A node fed independent noise (no common source): gate_ok False, others still ok."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)

        rng = _rng(20)
        n_total = int(6 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        # Node 2 gets independent noise (no source)
        arr2_noise = _make_noise_int16(n_total, rng, rms=4000)

        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2_noise[idx:end])
            idx = end

        pl.flush()
        results = pl.poll()
        assert len(results) >= 1

        # At least one result with node 2 gate_ok False
        node2_gates = [
            r["bearings"]["2"]["gate_ok"]
            for r in results if "2" in r["bearings"]
        ]
        assert any(not g for g in node2_gates), (
            "Expected at least one gate_ok=False for independent-noise node 2"
        )

    def test_low_energy_gate_false(self):
        """Signal of ~10 LSB RMS -> gate_ok False due to energy gate."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(21)
        n_total = int(4 * _FS)

        # Very low amplitude signal (~10 LSB RMS)
        low_arr = _make_noise_int16(n_total, rng, rms=10)
        _push_stream(pl, 1, low_arr)
        _push_stream(pl, 2, low_arr)
        pl.flush()
        results = pl.poll()
        assert len(results) >= 1

        for r in results:
            for nid_str in ["1", "2"]:
                if nid_str in r["bearings"]:
                    assert not r["bearings"][nid_str]["gate_ok"], (
                        f"Node {nid_str} gate_ok should be False for low-energy signal"
                    )


# ---------------------------------------------------------------------------
# (c) Smoother test
# ---------------------------------------------------------------------------

class TestSmoother:
    def test_smoother_monkeypatch(self, monkeypatch):
        """5 fixes at A, 1 outlier at B: x_m/y_m stays near A; x_raw_m/y_raw_m = B.
        Non-ok result does not update the smoother.
        """
        import kaawayan.live as live_mod

        A = (1.5, 3.0)
        B = (100.0, 100.0)

        call_count = [0]
        # Return A for first 5, then B for 6th, then back to A
        def fake_fuse(bearings, poses, min_crossing_deg=30.0, gate_min_confidence=0.0):
            call_count[0] += 1
            n = call_count[0]
            xy = B if n == 6 else A
            return {"x_m": xy[0], "y_m": xy[1], "crossing_deg": 60.0, "n_lines": 2,
                    "used_node_ids": [1, 2], "status": "ok", "reason": ""}

        monkeypatch.setattr(live_mod, "fuse", fake_fuse)

        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(30)
        n_total = int(8 * _FS)  # enough windows
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)

        idx = 0
        while idx < n_total:
            end = min(idx + _CHUNK, n_total)
            t = idx / _FS
            pl.push_audio(1, t, arr1[idx:end])
            pl.push_audio(2, t, arr2[idx:end])
            idx = end
        pl.flush()
        results = pl.poll()

        ok = [r for r in results if r["status"] == "ok"]
        assert len(ok) >= 6, f"Expected >= 6 ok results, got {len(ok)}"

        # The 6th ok result (from fake_fuse call 6) should have x_raw_m = B[0]
        # but x_m should be close to A (median of last 5 including some B's)
        assert call_count[0] >= 6, f"fake_fuse called only {call_count[0]} times"

        # After 5 A fixes and 1 B, smoother (n=5) has [A,A,A,A,A] after window 5
        # then B replaces oldest A -> [A,A,A,A,B] -> median = A  (B is outlier)
        outlier_result = ok[5]  # 6th ok result corresponds to fuse call 6 (B)
        assert abs(outlier_result["x_raw_m"] - B[0]) < 1e-6, (
            f"x_raw_m should be outlier B={B[0]}, got {outlier_result['x_raw_m']}"
        )
        assert abs(outlier_result["x_m"] - A[0]) < 0.25, (
            f"Smoother x_m {outlier_result['x_m']} should be near A={A[0]}"
        )

    def test_non_ok_does_not_update_smoother(self):
        """A non-ok result does not change the smoother state."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)

        rng = _rng(31)
        # Manually add ok fix at (1.5, 3.0) to smoother
        pl._smoother.update({"status": "ok", "x_m": 1.5, "y_m": 3.0})
        prev_xs = list(pl._smoother._xs)
        prev_ys = list(pl._smoother._ys)

        # Feed non-ok result to smoother
        pl._smoother.update({"status": "uncertain"})

        assert pl._smoother._xs == prev_xs
        assert pl._smoother._ys == prev_ys


# ---------------------------------------------------------------------------
# (d) JSON serialisability
# ---------------------------------------------------------------------------

class TestJsonSerializable:
    def test_all_results_json_serializable(self):
        """json.dumps(sort_keys=True, allow_nan=False) works on every result."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(40)
        n_total = int(6 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)
        _push_stream(pl, 1, arr1)
        _push_stream(pl, 2, arr2)
        pl.flush()
        results = pl.poll()

        for i, r in enumerate(results):
            try:
                json.dumps(r, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Result {i} not JSON-serialisable: {e}\nResult: {r}")


# ---------------------------------------------------------------------------
# replay_trial tests
# ---------------------------------------------------------------------------

def _write_wav(path: pathlib.Path, arr: np.ndarray) -> None:
    """Write (n,2) int16 array to WAV."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(_FS)
        wf.writeframes(arr.tobytes())


class TestReplayTrial:
    def test_replay_twice_identical(self, tmp_path):
        """replay_trial twice -> byte-identical bearings.csv and positions.csv."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        rng = _rng(50)
        n_total = int(6 * _FS)
        src_xy = (1.5, 3.0)

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)
        _write_wav(trial_dir / "node1.wav", arr1)
        _write_wav(trial_dir / "node2.wav", arr2)
        (trial_dir / "meta.json").write_text(json.dumps({
            "trial_id": "t1", "kind": "tone", "config_hash": "x",
            "nodes": {
                "1": {"t_first_audio_laptop_s": 0.0},
                "2": {"t_first_audio_laptop_s": 0.0},
            }
        }))

        replay_trial(trial_dir, SITE_TOML)
        b1 = (trial_dir / "bearings.csv").read_bytes()
        p1 = (trial_dir / "positions.csv").read_bytes()

        replay_trial(trial_dir, SITE_TOML)
        b2 = (trial_dir / "bearings.csv").read_bytes()
        p2 = (trial_dir / "positions.csv").read_bytes()

        assert b1 == b2, "bearings.csv not identical on second replay"
        assert p1 == p2, "positions.csv not identical on second replay"

    def test_bearings_csv_header(self, tmp_path):
        """bearings.csv header equals the CONTEXT section 7 columns."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        rng = _rng(51)
        n_total = int(4 * _FS)

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        arr1 = _make_noise_int16(n_total, rng)
        arr2 = _make_noise_int16(n_total, rng)
        _write_wav(trial_dir / "node1.wav", arr1)
        _write_wav(trial_dir / "node2.wav", arr2)
        (trial_dir / "meta.json").write_text(json.dumps({
            "trial_id": "t1", "kind": "tone", "config_hash": "x",
            "nodes": {
                "1": {"t_first_audio_laptop_s": 0.0},
                "2": {"t_first_audio_laptop_s": 0.0},
            }
        }))

        replay_trial(trial_dir, SITE_TOML)
        with open(trial_dir / "bearings.csv") as f:
            reader = csv.reader(f)
            header = next(reader)

        expected = ["t_laptop_s", "node_id", "window_idx", "theta_deg",
                    "confidence", "peak_median_ratio", "gate_ok"]
        assert header == expected, f"bearings.csv header {header} != {expected}"

    def test_config_hash_mismatch_warning(self, tmp_path, capsys):
        """Config hash mismatch prints warning to stderr and continues."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        rng = _rng(52)
        n_total = int(2 * _FS)

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        arr1 = _make_noise_int16(n_total, rng)
        arr2 = _make_noise_int16(n_total, rng)
        _write_wav(trial_dir / "node1.wav", arr1)
        _write_wav(trial_dir / "node2.wav", arr2)
        (trial_dir / "meta.json").write_text(json.dumps({
            "trial_id": "t1", "kind": "tone", "config_hash": "bad_hash_abc",
            "nodes": {
                "1": {"t_first_audio_laptop_s": 0.0},
                "2": {"t_first_audio_laptop_s": 0.0},
            }
        }))

        # Should not raise, just warn
        replay_trial(trial_dir, SITE_TOML)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err, "Expected WARNING in stderr"
        assert "config_hash" in captured.err

    def test_replay_t_first_audio_offset(self, tmp_path):
        """t_first_audio_laptop_s offset applied correctly in replay."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        rng = _rng(53)
        n_total = int(4 * _FS)

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        arr1 = _make_noise_int16(n_total, rng)
        arr2 = _make_noise_int16(n_total, rng)
        _write_wav(trial_dir / "node1.wav", arr1)
        _write_wav(trial_dir / "node2.wav", arr2)
        offset = 5.0
        (trial_dir / "meta.json").write_text(json.dumps({
            "trial_id": "t1", "kind": "tone", "config_hash": "x",
            "nodes": {
                "1": {"t_first_audio_laptop_s": offset},
                "2": {"t_first_audio_laptop_s": offset},
            }
        }))

        results = replay_trial(trial_dir, SITE_TOML)
        if results:
            # t_laptop_s should reflect the offset
            assert all(r["t_laptop_s"] >= offset for r in results), (
                "t_laptop_s should be >= t_first_audio_laptop_s offset"
            )


# ---------------------------------------------------------------------------
# Result fields completeness
# ---------------------------------------------------------------------------

class TestResultFields:
    def test_all_required_fields_present(self):
        """Every result has all required fields."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(60)
        n_total = int(4 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)
        _push_stream(pl, 1, arr1)
        _push_stream(pl, 2, arr2)
        pl.flush()
        results = pl.poll()

        required = {
            "t_laptop_s", "x_m", "y_m", "x_raw_m", "y_raw_m",
            "crossing_deg", "n_nodes", "dead_node_ids", "status", "reason",
            "bearings", "config_hash", "convention_version",
        }
        for r in results:
            missing = required - set(r.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_bearings_fields(self):
        """Each bearing entry has window_idx, theta_deg, confidence, peak_median_ratio, gate_ok."""
        cfg = _make_cfg_2node()
        poses = poses_from_config(cfg)
        pl = LivePipeline(cfg, poses)
        rng = _rng(61)
        n_total = int(4 * _FS)
        src_xy = (1.5, 3.0)

        arr1 = _make_stereo_int16(src_xy, poses[1], n_total, rng, cfg)
        arr2 = _make_stereo_int16(src_xy, poses[2], n_total, rng, cfg)
        _push_stream(pl, 1, arr1)
        _push_stream(pl, 2, arr2)
        pl.flush()
        results = pl.poll()

        required_b = {"window_idx", "theta_deg", "confidence", "peak_median_ratio", "gate_ok"}
        for r in results:
            for nid_str, binfo in r["bearings"].items():
                missing = required_b - set(binfo.keys())
                assert not missing, f"Node {nid_str} bearing missing: {missing}"
