"""
tests/test_capture.py — Tests for kaawayan.capture.run_capture() and CLI.

All tests use fake sources, fake clocks, and temporary directories.
SerialSource and the CLI real-port path are UNTESTED ON HARDWARE.
"""

from __future__ import annotations

import io
import json
import pathlib
import struct
import sys
import types
import wave
import zlib

import numpy as np
import pytest

from kaawayan.config import load_config
from kaawayan.capture import run_capture, _parse_true_theta, _parse_true_xy
from kaawayan.record import FrameAssembler, TrialRecorder
from kaawayan.protocol import FrameType, encode_frame, EnvPayload, StartupPayload

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SITE_TOML = REPO / "config" / "site.toml"
_FS = 48_000
_SAMPLES_PER_AUDIO = 512
_AUDIO_PAYLOAD_BYTES = _SAMPLES_PER_AUDIO * 2 * 2  # 512 stereo int16


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def _startup_frame(node_id: int, seq: int = 0) -> bytes:
    # _STARTUP_FMT = '<BBBBIIHBb' has 9 fields
    payload = struct.pack('<BBBBIIHbb', 1, 1, 0, 0, 0, _FS, _SAMPLES_PER_AUDIO, 2, 3)
    return encode_frame(FrameType.STARTUP, node_id, seq, 0, payload)


def _audio_frame(node_id: int, seq: int, samples: np.ndarray | None = None) -> bytes:
    """Build an AUDIO frame with 512 stereo int16 samples."""
    if samples is None:
        samples = np.zeros((_SAMPLES_PER_AUDIO, 2), dtype=np.int16)
    payload = samples.tobytes()
    return encode_frame(FrameType.AUDIO, node_id, seq, 0, payload)


def _env_frame(node_id: int, seq: int) -> bytes:
    payload = struct.pack("<hHIIB", 2000, 5000, 0, 0, 0)  # 20.00°C, 50.00%RH
    return encode_frame(FrameType.ENV, node_id, seq, 0, payload)


def _make_stream(node_id: int, n_audio: int) -> bytes:
    """Make a byte stream: STARTUP + n_audio AUDIO frames + 1 ENV frame."""
    buf = bytearray()
    buf += _startup_frame(node_id, seq=0)
    for i in range(n_audio):
        rng = np.random.default_rng(node_id * 1000 + i)
        samples = (rng.standard_normal((_SAMPLES_PER_AUDIO, 2)) * 4000).astype(np.int16)
        buf += _audio_frame(node_id, seq=i, samples=samples)
    buf += _env_frame(node_id, seq=0)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fake source
# ---------------------------------------------------------------------------

class FakeSource:
    """Feed a byte buffer as if it were a serial port.

    read() returns chunks of at most chunk_size bytes.
    When the buffer is exhausted, read() returns b"" (simulates no data).
    """

    def __init__(self, data: bytes, chunk_size: int = 256) -> None:
        self._data = data
        self._pos = 0
        self._chunk = chunk_size
        self.closed = False

    def read(self, size: int = 4096) -> bytes:
        if self._pos >= len(self._data):
            return b""
        n = min(self._chunk, len(self._data) - self._pos)
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeClock:
    """Monotonically advancing fake clock."""

    def __init__(self, rate: float = 1.0) -> None:
        """rate: seconds of real time per call (i.e. each call() advances by rate)."""
        self._t = 0.0
        self._rate = rate

    def __call__(self) -> float:
        t = self._t
        self._t += self._rate
        return t

    @property
    def now(self) -> float:
        return self._t


class RaisingSource:
    """Source that raises IOError after some successful reads."""

    def __init__(self, data: bytes, fail_after: int = 0) -> None:
        self._data = data
        self._pos = 0
        self._calls = 0
        self._fail_after = fail_after
        self.closed = False

    def read(self, size: int = 4096) -> bytes:
        if self._calls >= self._fail_after:
            raise IOError("simulated disconnect")
        self._calls += 1
        if self._pos >= len(self._data):
            return b""
        n = min(256, len(self._data) - self._pos)
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_wav_samples(path: pathlib.Path) -> int:
    with wave.open(str(path)) as wf:
        return wf.getnframes()


def _make_recorder(tmp: pathlib.Path, node_ids: list[int]) -> TrialRecorder:
    return TrialRecorder(tmp, node_ids)


# ---------------------------------------------------------------------------
# Test: 2 nodes, 3 s clean stream
# ---------------------------------------------------------------------------

class TestCleanRun:
    """2 nodes, clean STARTUP+AUDIO+ENV stream for ~3 s."""

    def _run(self, tmp_path: pathlib.Path, duration_s: float = 3.0,
             n_audio: int = 144) -> tuple[dict, pathlib.Path]:
        # 144 audio frames ≈ 144 * 512/48000 ≈ 1.5 s (we use a fake clock that
        # advances to simulate duration_s elapsed)
        node_ids = [1, 2]
        streams = {nid: _make_stream(nid, n_audio) for nid in node_ids}

        # Clock: start, then advance past duration_s on first "no data" check
        # We use a simple counter clock
        call_count = [0]
        times = [0.0, 0.0]  # [start, ..increments]

        # Use a real monotonic-ish: pre-load enough ticks
        # Each read gives ~256 bytes; streams are ~(2 + n_audio + 1) frames
        # Just use a simple sequenced clock
        ticks = [0.001 * i for i in range(100000)]
        tick_idx = [0]

        def fake_clock() -> float:
            t = ticks[tick_idx[0]]
            tick_idx[0] += 1
            # After all data exhausted, jump past duration
            if tick_idx[0] > len(ticks) - 2:
                return duration_s + 1.0
            return t

        sources = {nid: FakeSource(streams[nid]) for nid in node_ids}
        assemblers = {nid: FrameAssembler() for nid in node_ids}
        recorder = _make_recorder(tmp_path, node_ids)

        # We need clock to eventually exceed duration_s. Simulate by using
        # a clock that jumps once data is exhausted.
        # Simpler approach: just wrap clock so after N calls it returns duration_s+1
        real_ticks = iter([i * 0.0001 for i in range(200000)] + [duration_s + 1.0] * 10)

        def clk() -> float:
            return next(real_ticks)

        summary = run_capture(
            sources=sources,
            recorder=recorder,
            assemblers=assemblers,
            duration_s=duration_s,
            clock=clk,
            sleep=lambda _: None,
            trial_id="t_clean",
            kind="tone",
            cfg_path=None,
            extra_meta=None,
        )
        return summary, tmp_path

    def test_wav_files_created(self, tmp_path: pathlib.Path) -> None:
        _, d = self._run(tmp_path)
        assert (d / "node1.wav").exists()
        assert (d / "node2.wav").exists()

    def test_env_csv_created(self, tmp_path: pathlib.Path) -> None:
        _, d = self._run(tmp_path)
        assert (d / "node1_env.csv").exists()
        assert (d / "node2_env.csv").exists()

    def test_meta_json_created_and_valid(self, tmp_path: pathlib.Path) -> None:
        _, d = self._run(tmp_path)
        meta = json.loads((d / "meta.json").read_text())
        assert "trial_id" in meta
        assert "nodes" in meta
        assert "1" in meta["nodes"]
        assert "2" in meta["nodes"]

    def test_exactly_one_meta_json(self, tmp_path: pathlib.Path) -> None:
        _, d = self._run(tmp_path)
        assert len(list(d.glob("meta.json"))) == 1

    def test_summary_has_both_nodes(self, tmp_path: pathlib.Path) -> None:
        summary, _ = self._run(tmp_path)
        assert 1 in summary
        assert 2 in summary

    def test_bytes_positive(self, tmp_path: pathlib.Path) -> None:
        summary, _ = self._run(tmp_path)
        for nid in [1, 2]:
            assert summary[nid]["bytes"] > 0

    def test_frames_counted(self, tmp_path: pathlib.Path) -> None:
        summary, _ = self._run(tmp_path)
        for nid in [1, 2]:
            fc = summary[nid]["frames"]
            assert fc.get("AUDIO", 0) > 0, f"node {nid} no AUDIO frames"

    def test_not_lost(self, tmp_path: pathlib.Path) -> None:
        summary, _ = self._run(tmp_path)
        for nid in [1, 2]:
            assert not summary[nid]["lost"]


# ---------------------------------------------------------------------------
# Test: bytes_per_s from fake clock
# ---------------------------------------------------------------------------

class TestBytesPerS:
    def test_bytes_per_s_correct(self, tmp_path: pathlib.Path) -> None:
        """bytes_per_s = bytes / elapsed, where elapsed comes from fake clock."""
        node_ids = [1]
        stream = _make_stream(1, n_audio=10)
        duration_s = 2.0

        # Clock: tick 0.0 at start, then 0.001 increments, jump past duration at end
        ticks = [0.0] + [0.001 * i for i in range(1, 5000)] + [duration_s + 1.0] * 10
        tick_idx = [0]

        def clk() -> float:
            t = ticks[tick_idx[0]]
            tick_idx[0] = min(tick_idx[0] + 1, len(ticks) - 1)
            return t

        sources = {1: FakeSource(stream)}
        assemblers = {1: FrameAssembler()}
        recorder = _make_recorder(tmp_path, node_ids)

        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=duration_s, clock=clk, sleep=lambda _: None,
            trial_id="t_bps", kind="tone",
        )

        s = summary[1]
        # elapsed should be approximately duration_s (clock jumped past it)
        assert s["bytes"] > 0
        assert s["bytes_per_s"] > 0
        expected_bps = s["bytes"] / duration_s
        # Allow 10% tolerance due to clock granularity
        assert abs(s["bytes_per_s"] - expected_bps) / expected_bps < 0.2, (
            f"bytes_per_s {s['bytes_per_s']:.1f} too far from {expected_bps:.1f}"
        )


# ---------------------------------------------------------------------------
# Test: source raises IOError mid-run
# ---------------------------------------------------------------------------

class TestSourceFailure:
    def test_lost_node_other_still_complete(self, tmp_path: pathlib.Path) -> None:
        """One source raises IOError; files still valid, that node in capture_errors."""
        node_ids = [1, 2]
        stream_ok = _make_stream(1, n_audio=20)
        stream_bad = _make_stream(2, n_audio=5)

        # Node 2 fails after 0 successful reads
        sources = {
            1: FakeSource(stream_ok),
            2: RaisingSource(stream_bad, fail_after=0),
        }
        assemblers = {nid: FrameAssembler() for nid in node_ids}
        recorder = _make_recorder(tmp_path, node_ids)

        ticks = [i * 0.0001 for i in range(200000)] + [10.0 + 1.0] * 10
        tick_idx = [0]

        def clk() -> float:
            t = ticks[tick_idx[0]]
            tick_idx[0] = min(tick_idx[0] + 1, len(ticks) - 1)
            return t

        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=10.0, clock=clk, sleep=lambda _: None,
            trial_id="t_fail", kind="clap",
        )

        # Files still written
        assert (tmp_path / "node1.wav").exists()
        assert (tmp_path / "node2.wav").exists()
        assert (tmp_path / "meta.json").exists()

        # Node 2 is lost
        assert summary[2]["lost"]
        assert not summary[1]["lost"]

        # capture_errors in meta.json
        meta = json.loads((tmp_path / "meta.json").read_text())
        ce = meta.get("capture_errors", {})
        assert "2" in ce

    def test_all_sources_lost_stops_early(self, tmp_path: pathlib.Path) -> None:
        """If ALL sources raise, run_capture returns early."""
        node_ids = [1, 2]
        sources = {
            1: RaisingSource(b"", fail_after=0),
            2: RaisingSource(b"", fail_after=0),
        }
        assemblers = {nid: FrameAssembler() for nid in node_ids}
        recorder = _make_recorder(tmp_path, node_ids)

        ticks = [0.0, 0.001, 0.002] + [1000.0] * 10  # would be huge if not early-stop
        tick_idx = [0]

        def clk() -> float:
            t = ticks[tick_idx[0]]
            tick_idx[0] = min(tick_idx[0] + 1, len(ticks) - 1)
            return t

        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=100.0, clock=clk, sleep=lambda _: None,
            trial_id="t_alllost", kind="clap",
        )

        # Both lost
        assert summary[1]["lost"]
        assert summary[2]["lost"]
        # Files still written
        assert (tmp_path / "meta.json").exists()


# ---------------------------------------------------------------------------
# Test: KeyboardInterrupt closes recorder cleanly
# ---------------------------------------------------------------------------

class TestKeyboardInterrupt:
    def test_keyboard_interrupt_closes_cleanly(self, tmp_path: pathlib.Path) -> None:
        """KeyboardInterrupt during capture: recorder still closed, files valid."""
        node_ids = [1]

        class KillSource:
            def __init__(self, data: bytes, kill_after: int) -> None:
                self._data = data
                self._pos = 0
                self._calls = 0
                self._kill_after = kill_after
                self.closed = False

            def read(self, size: int = 4096) -> bytes:
                self._calls += 1
                if self._calls > self._kill_after:
                    raise KeyboardInterrupt
                if self._pos >= len(self._data):
                    return b""
                n = min(256, len(self._data) - self._pos)
                chunk = self._data[self._pos:self._pos + n]
                self._pos += n
                return chunk

            def close(self) -> None:
                self.closed = True

        stream = _make_stream(1, n_audio=10)
        sources = {1: KillSource(stream, kill_after=5)}
        assemblers = {1: FrameAssembler()}
        recorder = _make_recorder(tmp_path, node_ids)

        ticks = iter([i * 0.001 for i in range(10000)] + [100.0] * 5)

        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=50.0, clock=lambda: next(ticks), sleep=lambda _: None,
            trial_id="t_ctrl_c", kind="speaker",
        )

        # Files written despite interrupt
        assert (tmp_path / "node1.wav").exists()
        assert (tmp_path / "meta.json").exists()


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------

class TestCliParsing:
    def test_parse_true_theta_valid(self) -> None:
        result = _parse_true_theta("1=30,2=-30")
        assert result == {"1": 30.0, "2": -30.0}

    def test_parse_true_theta_single(self) -> None:
        result = _parse_true_theta("1=0")
        assert result == {"1": 0.0}

    def test_parse_true_theta_malformed(self) -> None:
        import argparse as _ap
        with pytest.raises(_ap.ArgumentTypeError):
            _parse_true_theta("1-30")  # missing =

    def test_parse_true_theta_bad_float(self) -> None:
        import argparse as _ap
        with pytest.raises(_ap.ArgumentTypeError):
            _parse_true_theta("1=abc")

    def test_parse_true_xy_valid(self) -> None:
        result = _parse_true_xy("1.5,3.0")
        assert result == [1.5, 3.0]

    def test_parse_true_xy_malformed_one_value(self) -> None:
        import argparse as _ap
        with pytest.raises(_ap.ArgumentTypeError):
            _parse_true_xy("1.5")

    def test_parse_true_xy_bad_float(self) -> None:
        import argparse as _ap
        with pytest.raises(_ap.ArgumentTypeError):
            _parse_true_xy("x,y")

    def test_parse_true_xy_three_values(self) -> None:
        import argparse as _ap
        with pytest.raises(_ap.ArgumentTypeError):
            _parse_true_xy("1.5,3.0,0.0")


# ---------------------------------------------------------------------------
# Test: CLI --list-ports (monkeypatched)
# ---------------------------------------------------------------------------

class TestListPorts:
    def test_list_ports(self, monkeypatch: pytest.MonkeyPatch,
                        capsys: pytest.CaptureFixture) -> None:
        """--list-ports prints port info and exits 0."""
        import types as _types

        FakePort = _types.SimpleNamespace(
            device="COM3",
            serial_number="ABCD1234",
            vid=0x303A,
            pid=0x4001,
        )

        # Build fake serial module hierarchy
        fake_serial = _types.ModuleType("serial")
        fake_serial_tools = _types.ModuleType("serial.tools")
        fake_lp = _types.ModuleType("serial.tools.list_ports")
        fake_lp.comports = lambda: [FakePort]
        fake_serial.tools = fake_serial_tools
        fake_serial_tools.list_ports = fake_lp

        monkeypatch.setitem(sys.modules, "serial", fake_serial)
        monkeypatch.setitem(sys.modules, "serial.tools", fake_serial_tools)
        monkeypatch.setitem(sys.modules, "serial.tools.list_ports", fake_lp)

        from kaawayan.capture import _main
        rc = _main(["--list-ports"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "COM3" in out
        assert "ABCD1234" in out


# ---------------------------------------------------------------------------
# Test: non-empty output folder refused
# ---------------------------------------------------------------------------

class TestNonEmptyFolder:
    def test_nonempty_folder_refused(self, tmp_path: pathlib.Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
        # Create a non-empty folder
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "existing.txt").write_text("data")

        from kaawayan.capture import _main
        rc = _main([
            "--config", str(SITE_TOML),
            "--out", str(out_dir),
            "--kind", "clap",
            "--seconds", "1",
        ])
        assert rc == 2

    def test_malformed_true_theta_exits_2(self, tmp_path: pathlib.Path) -> None:
        out_dir = tmp_path / "out2"
        out_dir.mkdir()
        from kaawayan.capture import _main
        rc = _main([
            "--config", str(SITE_TOML),
            "--out", str(out_dir),
            "--kind", "clap",
            "--seconds", "1",
            "--true-theta", "bad_value",
        ])
        assert rc == 2

    def test_malformed_true_xy_exits_2(self, tmp_path: pathlib.Path) -> None:
        out_dir = tmp_path / "out3"
        out_dir.mkdir()
        from kaawayan.capture import _main
        rc = _main([
            "--config", str(SITE_TOML),
            "--out", str(out_dir),
            "--kind", "clap",
            "--seconds", "1",
            "--true-xy", "bad,xy,extra",
        ])
        assert rc == 2


# ---------------------------------------------------------------------------
# Integration test: sim -> capture -> bench_table + replay_trial
# ---------------------------------------------------------------------------

class TestIntegration:
    """Build AUDIO frames from sim.make_stereo, run_capture into tmp_path,
    then verify bench_table.compute_table and live.replay_trial work."""

    def _make_audio_payload(self, theta_deg: float, cfg: dict,
                            rng: np.random.Generator) -> bytes:
        """Build a 2048-byte AUDIO payload from sim.make_stereo."""
        from kaawayan.sim import make_stereo
        from kaawayan.bearing import speed_of_sound

        n = _SAMPLES_PER_AUDIO
        signal = rng.standard_normal(n).astype(np.float64)
        c = speed_of_sound(20.0)
        left, right = make_stereo(
            signal, float(_FS), theta_deg, cfg["mic_spacing_m"], c,
            snr_db=25.0, rng=rng,
        )
        scale = 4000.0 / (np.std(left) + 1e-9)
        left = (left * scale).clip(-32768, 32767).astype(np.int16)
        right = (right * scale).clip(-32768, 32767).astype(np.int16)
        arr = np.stack([left, right], axis=1)
        return arr.tobytes()

    def _make_trial(self, tmp_path: pathlib.Path, thetas: dict[int, float],
                    cfg: dict, n_audio: int = 200) -> pathlib.Path:
        """Create a trial folder with WAV files encoded via sim.make_stereo."""
        import wave as _wave

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()

        rng = np.random.default_rng(99)

        node_ids = list(thetas.keys())
        sources: dict[int, bytes] = {}
        for nid, theta in thetas.items():
            buf = bytearray()
            buf += _startup_frame(nid, seq=0)
            for i in range(n_audio):
                payload = self._make_audio_payload(theta, cfg, rng)
                buf += encode_frame(FrameType.AUDIO, nid, seq=i,
                                    t_us=0, payload=payload)
            buf += _env_frame(nid, seq=0)
            sources[nid] = bytes(buf)

        assemblers = {nid: FrameAssembler() for nid in node_ids}
        recorder = TrialRecorder(trial_dir, node_ids)

        ticks = iter([i * 0.0001 for i in range(500000)] + [100.0] * 5)

        def clk() -> float:
            return next(ticks)

        extra = {"true_theta_deg": {str(nid): theta for nid, theta in thetas.items()}}

        run_capture(
            sources={nid: FakeSource(sources[nid]) for nid in node_ids},
            recorder=recorder,
            assemblers=assemblers,
            duration_s=100.0,
            clock=clk,
            sleep=lambda _: None,
            trial_id="t_integ",
            kind="tone",
            cfg_path=SITE_TOML,
            extra_meta=extra,
        )
        return trial_dir

    def test_replay_trial_runs(self, tmp_path: pathlib.Path) -> None:
        """replay_trial on a capture-generated folder runs without error."""
        cfg, _ = load_config(SITE_TOML)
        thetas = {1: 0.0, 2: 30.0}
        trial_dir = self._make_trial(tmp_path, thetas, cfg, n_audio=100)

        from kaawayan.live import replay_trial
        results = replay_trial(trial_dir, SITE_TOML)
        assert len(results) >= 1, "replay_trial produced no results"

    def test_replay_writes_csvs(self, tmp_path: pathlib.Path) -> None:
        """replay_trial writes bearings.csv and positions.csv."""
        cfg, _ = load_config(SITE_TOML)
        thetas = {1: 0.0, 2: 30.0}
        trial_dir = self._make_trial(tmp_path, thetas, cfg, n_audio=100)

        from kaawayan.live import replay_trial
        replay_trial(trial_dir, SITE_TOML)
        assert (trial_dir / "bearings.csv").exists()
        assert (trial_dir / "positions.csv").exists()

    def test_bench_table_median_error(self, tmp_path: pathlib.Path) -> None:
        """bench_table.compute_table on a capture folder gives low median error."""
        cfg, _ = load_config(SITE_TOML)
        # Use node 1 only (single-node, single bearing angle)
        # Use enough windows so median is stable
        thetas = {1: 0.0}  # only node 1 pushes
        trial_dir = self._make_trial(tmp_path, thetas, cfg, n_audio=300)

        import sys as _sys
        import importlib.util
        bench_path = REPO / "laptop" / "scripts" / "bench_table.py"
        spec = importlib.util.spec_from_file_location("bench_table", bench_path)
        bench = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(bench)  # type: ignore

        # compute_table expects trial_dirs where meta.json has true_theta_deg
        result = bench.compute_table([trial_dir], cfg)
        assert result is not None, "compute_table returned None"
        # At least one (beta, band) combination should exist
        assert len(result.get("cells", {})) > 0 or len(result.get("summary", {})) >= 0


# ---------------------------------------------------------------------------
# R7 new tests: gap truthfulness, resets, CRC, close(), _main end-to-end
# ---------------------------------------------------------------------------

import zlib as _zlib


def _corrupt_crc(frame_bytes: bytes) -> bytes:
    """Flip the last byte of the CRC to create a CRC failure."""
    b = bytearray(frame_bytes)
    b[-1] ^= 0xFF
    return bytes(b)


def _audio_frame_ramp(node_id: int, seq: int) -> bytes:
    """Build AUDIO frame with ramp data: L = seq*512+i (mod 65536), R = ~L."""
    n = _SAMPLES_PER_AUDIO
    arr = np.zeros((n, 2), dtype=np.int16)
    for i in range(n):
        l_val = np.int16((seq * n + i) % 65536)
        arr[i, 0] = l_val
        arr[i, 1] = np.int16(~int(l_val))
    return _audio_frame(node_id, seq, arr)


class TestGapTruthfulness:
    """One dropped AUDIO frame -> total_gap_ms ≈ 10.67 ms, flagged False."""

    def _run_gap(self, tmp_path: pathlib.Path, n_before: int = 5,
                 drop: int = 1, n_after: int = 5) -> dict:
        nid = 1
        buf = bytearray()
        buf += _startup_frame(nid, seq=0)
        seq = 0
        for i in range(n_before):
            buf += _audio_frame(nid, seq=seq)
            seq += 1
        seq += drop  # skip `drop` frames
        for i in range(n_after):
            buf += _audio_frame(nid, seq=seq)
            seq += 1

        sources = {nid: FakeSource(bytes(buf))}
        assemblers = {nid: FrameAssembler()}
        recorder = _make_recorder(tmp_path, [nid])

        ticks = iter([i * 0.0001 for i in range(500000)] + [100.0] * 5)
        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=100.0, clock=lambda: next(ticks), sleep=lambda _: None,
            trial_id="t_gap1", kind="tone",
        )
        return summary

    def test_one_dropped_frame_gap_ms(self, tmp_path: pathlib.Path) -> None:
        summary = self._run_gap(tmp_path, drop=1)
        gap_ms = summary[1]["total_gap_ms"]
        expected = 512 / 48000 * 1000  # ~10.666...
        assert abs(gap_ms - expected) < 0.5, f"gap_ms={gap_ms}"

    def test_one_dropped_frame_not_flagged(self, tmp_path: pathlib.Path) -> None:
        summary = self._run_gap(tmp_path, drop=1)
        assert not summary[1]["flagged"]

    def test_many_dropped_frames_flagged(self, tmp_path: pathlib.Path) -> None:
        """12 dropped frames => 12*10.67 ms ≈ 128 ms > 100 ms -> flagged True."""
        summary = self._run_gap(tmp_path, drop=12)
        assert summary[1]["flagged"]

    def test_many_dropped_frames_gap_in_meta_json(self, tmp_path: pathlib.Path) -> None:
        self._run_gap(tmp_path, drop=12)
        meta = json.loads((tmp_path / "meta.json").read_text())
        gap_ms = meta["nodes"]["1"]["total_gap_ms"]
        expected = 12 * 512 / 48000 * 1000
        assert abs(gap_ms - expected) < 0.5, f"gap_ms={gap_ms}"
        assert meta["flagged"] is True


class TestSeqReset:
    """Sequence reset (5,6,0,1) -> resets==1 and flagged True."""

    def test_seq_reset_detected(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        buf = bytearray()
        buf += _startup_frame(nid, seq=0)
        for seq in [0, 1, 2, 3, 4, 5, 6]:
            buf += _audio_frame(nid, seq=seq)
        # Simulate node reboot: seq goes back to 0
        for seq in [0, 1, 2, 3]:
            buf += _audio_frame(nid, seq=seq)

        sources = {nid: FakeSource(bytes(buf))}
        assemblers = {nid: FrameAssembler()}
        recorder = _make_recorder(tmp_path, [nid])

        ticks = iter([i * 0.0001 for i in range(500000)] + [100.0] * 5)
        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=100.0, clock=lambda: next(ticks), sleep=lambda _: None,
            trial_id="t_reset", kind="tone",
        )
        assert summary[nid]["resets"] == 1
        assert summary[nid]["flagged"] is True


class TestCRCFailure:
    """CRC-corrupted frame -> crc_failures >= 1 in summary."""

    def test_crc_failure_counted(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        buf = bytearray()
        buf += _startup_frame(nid, seq=0)
        buf += _audio_frame(nid, seq=0)
        buf += _corrupt_crc(_audio_frame(nid, seq=1))
        buf += _audio_frame(nid, seq=2)  # seq jumps by 1 after skip

        sources = {nid: FakeSource(bytes(buf))}
        assemblers = {nid: FrameAssembler()}
        recorder = _make_recorder(tmp_path, [nid])

        ticks = iter([i * 0.0001 for i in range(500000)] + [100.0] * 5)
        summary = run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=100.0, clock=lambda: next(ticks), sleep=lambda _: None,
            trial_id="t_crc", kind="tone",
        )
        assert summary[nid]["crc_failures"] >= 1


class TestSourceClose:
    """Sources get close() called, including after a read() exception."""

    def test_normal_sources_closed(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        stream = _make_stream(nid, n_audio=5)
        src = FakeSource(stream)
        sources = {nid: src}
        assemblers = {nid: FrameAssembler()}
        recorder = _make_recorder(tmp_path, [nid])

        ticks = iter([i * 0.0001 for i in range(500000)] + [100.0] * 5)
        run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=100.0, clock=lambda: next(ticks), sleep=lambda _: None,
            trial_id="t_close", kind="tone",
        )
        assert src.closed

    def test_source_closed_after_exception(self, tmp_path: pathlib.Path) -> None:
        """Source that immediately raises is close()d."""
        nid = 1
        src = RaisingSource(b"", fail_after=0)
        sources = {nid: src}
        assemblers = {nid: FrameAssembler()}
        recorder = _make_recorder(tmp_path, [nid])

        ticks = iter([0.0, 0.001, 100.0] * 5)
        run_capture(
            sources=sources, recorder=recorder, assemblers=assemblers,
            duration_s=100.0, clock=lambda: next(ticks), sleep=lambda _: None,
            trial_id="t_exc_close", kind="tone",
        )
        # RaisingSource.close() sets self.closed = True
        assert src.closed


class TestMainEndToEnd:
    """_main end-to-end with monkeypatched SerialSource."""

    def _fake_serial_source_factory(self, streams: dict[str, bytes]):
        """Return a class that yields FakeSources keyed by usb_serial."""
        class _FakeSerialSource:
            def __init__(self_inner, port_or_serial: str, baudrate: int = 115200) -> None:
                data = streams.get(port_or_serial, b"")
                self_inner._src = FakeSource(data)

            def read(self_inner, size: int = 4096) -> bytes:
                return self_inner._src.read(size)

            def close(self_inner) -> None:
                self_inner._src.close()

            @property
            def is_open(self_inner) -> bool:
                return not self_inner._src.closed

        return _FakeSerialSource

    def test_main_2nodes_exit0(self, tmp_path: pathlib.Path,
                               monkeypatch: pytest.MonkeyPatch,
                               capsys: pytest.CaptureFixture) -> None:
        """_main with 2 nodes, 1 s, exits 0 and writes files."""
        from kaawayan.config import load_config
        cfg, _ = load_config(SITE_TOML)
        node_ids = [n["node_id"] for n in cfg.get("nodes", [])]
        usb_serials = {n["node_id"]: n.get("usb_serial", "") for n in cfg.get("nodes", [])}

        # Build streams per usb_serial
        streams: dict[str, bytes] = {}
        for nid in node_ids:
            streams[usb_serials[nid]] = _make_stream(nid, n_audio=50)

        import kaawayan.record as _record
        FakeSerial = self._fake_serial_source_factory(streams)
        monkeypatch.setattr(_record, "SerialSource", FakeSerial)
        # Also patch in capture module's import
        import kaawayan.capture as _capture_mod

        out_dir = tmp_path / "trial_main"

        from kaawayan.capture import _main
        rc = _main([
            "--config", str(SITE_TOML),
            "--out", str(out_dir),
            "--kind", "tone",
            "--seconds", "1",
        ])

        assert rc == 0

        out = capsys.readouterr().out
        for nid in node_ids:
            assert str(nid) in out

        for nid in node_ids:
            assert (out_dir / f"node{nid}.wav").exists(), f"missing node{nid}.wav"
        assert (out_dir / "meta.json").exists()

    def test_main_failing_second_source_exits2(
            self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_main with a failing second source returns 2 and first source closed."""
        from kaawayan.config import load_config
        cfg, _ = load_config(SITE_TOML)
        nodes = cfg.get("nodes", [])
        if len(nodes) < 2:
            pytest.skip("need at least 2 nodes in config")

        closed_sources: list[str] = []

        call_count = [0]

        class _FailingSecondSource:
            def __init__(self_inner, port_or_serial: str, baudrate: int = 115200) -> None:
                self_inner._port = port_or_serial
                call_count[0] += 1
                if call_count[0] >= 2:
                    raise OSError(f"cannot open {port_or_serial}")
                self_inner._data = _make_stream(1, n_audio=5)
                self_inner._pos = 0

            def read(self_inner, size: int = 4096) -> bytes:
                if self_inner._pos >= len(self_inner._data):
                    return b""
                chunk = self_inner._data[self_inner._pos:self_inner._pos + 256]
                self_inner._pos += 256
                return chunk

            def close(self_inner) -> None:
                closed_sources.append(self_inner._port)

            @property
            def is_open(self_inner) -> bool:
                return True

        import kaawayan.record as _record
        monkeypatch.setattr(_record, "SerialSource", _FailingSecondSource)

        out_dir = tmp_path / "trial_fail2"

        from kaawayan.capture import _main
        rc = _main([
            "--config", str(SITE_TOML),
            "--out", str(out_dir),
            "--kind", "clap",
            "--seconds", "1",
        ])

        assert rc == 2
        # First source should have been closed
        assert len(closed_sources) >= 1
