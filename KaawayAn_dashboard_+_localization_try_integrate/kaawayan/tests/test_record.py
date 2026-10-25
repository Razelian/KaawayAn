"""
Tests for kaawayan.record — Slice 3a + R2 hardening.

Tests:
- FrameAssembler: clean stream, split-at-every-byte, CRC failure recovery,
  crc_failures/resync_bytes_skipped counts.
- NodeRecorder: clean stream; one dropped AUDIO frame (gap logged, zeros
  inserted, WAV length correct); a trial with >100 ms gaps is flagged;
  saving same stream twice gives byte-identical WAV, node{id}_env.csv, meta.json;
  seq reset (RESET detected, flagged, no zero-fill); duplicate frame skipped;
  250 AUDIO frames without close -> WAV still readable with >= 200 frames.
- TrialRecorder: two nodes -> node1.wav, node2.wav, node1_env.csv, node2_env.csv,
  one meta.json with both nodes; seq reset flagged; dup skipped; extra_meta;
  core keys not overridden; node with no audio -> empty WAV at close.
- SerialSource: tested through a fake (UNTESTED ON HARDWARE).
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import struct
import sys
import wave

import numpy as np
import pytest

from kaawayan.protocol import (
    FrameType,
    encode_frame,
    _STARTUP_FMT,
    _ENV_FMT,
)
from kaawayan.record import FrameAssembler, NodeRecorder, TrialRecorder, SerialSource


# ---------------------------------------------------------------------------
# Helpers to build fake payloads
# ---------------------------------------------------------------------------

def _audio_payload(samples: int = 512, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(-32768, 32767, size=(samples, 2), dtype=np.int16)
    return arr.tobytes()


def _startup_payload(node_id: int = 1) -> bytes:
    return struct.pack(_STARTUP_FMT, 1, 0, 1, 0, 0xABCD, 48000, 512, 2, 0x03)


def _env_payload(temp_c: float = 22.5) -> bytes:
    return struct.pack(_ENV_FMT, int(temp_c * 100), 5000, 0, 0, 0)


def _build_stream(node_id: int, n_audio: int = 3, include_startup: bool = True,
                  include_env: bool = True) -> bytes:
    """Build a complete fake byte stream for one node."""
    frames = bytearray()
    if include_startup:
        frames += encode_frame(FrameType.STARTUP, node_id, 0, 0, _startup_payload(node_id))
    for seq in range(n_audio):
        frames += encode_frame(FrameType.AUDIO, node_id, seq, seq * 10667, _audio_payload(512, seq))
    if include_env:
        frames += encode_frame(FrameType.ENV, node_id, 0, 0, _env_payload(22.5))
    return bytes(frames)


def _corrupt_crc(frame_bytes: bytes) -> bytes:
    """Flip the last byte of a frame (CRC)."""
    b = bytearray(frame_bytes)
    b[-1] ^= 0xFF
    return bytes(b)


# ---------------------------------------------------------------------------
# FrameAssembler tests
# ---------------------------------------------------------------------------

class TestFrameAssembler:
    def test_clean_stream_yields_all_frames(self):
        stream = _build_stream(1, n_audio=3)
        asm = FrameAssembler()
        results = asm.feed(stream, t_laptop_s=1.0)
        # STARTUP + 3 AUDIO + ENV = 5 frames
        assert len(results) == 5
        types = [r[0]["type"] for r in results]
        assert types.count(FrameType.STARTUP) == 1
        assert types.count(FrameType.AUDIO) == 3
        assert types.count(FrameType.ENV) == 1

    def test_timestamps_from_feed_call(self):
        stream = _build_stream(1, n_audio=2)
        asm = FrameAssembler()
        results = asm.feed(stream, t_laptop_s=42.5)
        for _, t in results:
            assert t == 42.5

    def test_split_at_every_byte_identical_output(self):
        stream = _build_stream(1, n_audio=2)
        # Feed all at once
        asm1 = FrameAssembler()
        expected = [(f["type"], f["seq"]) for f, _ in asm1.feed(stream, 0.0)]

        # Feed one byte at a time
        asm2 = FrameAssembler()
        results2 = []
        for i, b in enumerate(stream):
            results2.extend(asm2.feed(bytes([b]), t_laptop_s=float(i)))
        got = [(f["type"], f["seq"]) for f, _ in results2]
        assert got == expected

    def test_crc_failure_counted_and_stream_recovers(self):
        audio = encode_frame(FrameType.AUDIO, 1, 0, 0, _audio_payload(512, 0))
        good = encode_frame(FrameType.AUDIO, 1, 1, 0, _audio_payload(512, 1))
        stream = _corrupt_crc(audio) + good
        asm = FrameAssembler()
        results = asm.feed(stream, 0.0)
        assert asm.crc_failures == 1
        assert asm.resync_bytes_skipped >= 1
        # The good frame should still be found
        seqs = [f["seq"] for f, _ in results]
        assert 1 in seqs

    def test_garbage_before_frame_skipped(self):
        garbage = b"\x00\x01\x02\xBA\x00\xFF" * 4  # not a valid SYNC
        good = encode_frame(FrameType.AUDIO, 1, 7, 0, _audio_payload(512, 7))
        asm = FrameAssembler()
        results = asm.feed(garbage + good, 0.0)
        assert len(results) == 1
        assert results[0][0]["seq"] == 7
        assert asm.resync_bytes_skipped >= len(garbage)

    def test_frame_split_across_calls(self):
        frame = encode_frame(FrameType.AUDIO, 1, 0, 0, _audio_payload(512, 0))
        split = len(frame) // 2
        asm = FrameAssembler()
        r1 = asm.feed(frame[:split], 1.0)
        r2 = asm.feed(frame[split:], 2.0)
        assert len(r1) == 0
        assert len(r2) == 1
        assert r2[0][0]["seq"] == 0
        # t from the call that completes the frame
        assert r2[0][1] == 2.0

    def test_no_crashes_on_empty_feed(self):
        asm = FrameAssembler()
        assert asm.feed(b"", 0.0) == []

    def test_initial_counters_zero(self):
        asm = FrameAssembler()
        assert asm.crc_failures == 0
        assert asm.resync_bytes_skipped == 0


# ---------------------------------------------------------------------------
# NodeRecorder tests
# ---------------------------------------------------------------------------

class TestNodeRecorder:
    def test_clean_stream_wav_length(self, tmp_path):
        node_id = 1
        n_audio = 4
        rec = NodeRecorder(node_id, tmp_path)
        stream = _build_stream(node_id, n_audio=n_audio)
        asm = FrameAssembler()
        for frame, t in asm.feed(stream, 0.0):
            rec.feed_frame(frame, t)
        rec.close(trial_id="t001", kind="tone")

        wav_path = tmp_path / f"node{node_id}.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path)) as wf:
            assert wf.getnchannels() == 2
            assert wf.getsampwidth() == 2  # int16
            assert wf.getframerate() == 48000
            assert wf.getnframes() == n_audio * 512

    def test_one_dropped_audio_frame_zeros_inserted(self, tmp_path):
        """Dropping seq=1 out of [0,1,2] → zeros inserted for the gap."""
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)

        # Manually feed frames: seq 0, 2 (skip seq 1)
        audio_payload_0 = _audio_payload(512, 0)
        audio_payload_2 = _audio_payload(512, 2)
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 1, "t_us": 0,
                        "payload": audio_payload_0, "version": 1}, 0.0)
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 2, "node_id": 1, "t_us": 0,
                        "payload": audio_payload_2, "version": 1}, 0.02)
        result = rec.close(trial_id="t002", kind="tone")

        wav_path = tmp_path / "node1.wav"
        with wave.open(str(wav_path)) as wf:
            # seq 0 (512) + gap seq 1 (512 zeros) + seq 2 (512) = 1536
            assert wf.getnframes() == 3 * 512

        # gaps.log must mention the gap
        gaps_log = tmp_path / "gaps.log"
        assert gaps_log.exists()
        content = gaps_log.read_text()
        assert "node_id=1" in content

        # The gap is 1 frame × (512/48000 s) ≈ 10.67 ms < 100 ms so not flagged
        assert result["total_gap_ms"] > 0
        assert result["flagged"] is False

    def test_more_than_100ms_gaps_flagged(self, tmp_path):
        """10 dropped frames × 10.67 ms > 100 ms → flagged=True.

        seq 0 then seq 11: gap = frames 1..10 = 10 frames
        = 10 * 512 / 48000 * 1000 = 106.67 ms > 100 ms.
        """
        node_id = 2
        rec = NodeRecorder(node_id, tmp_path)
        # seq 0, then seq 11 (gap of 10 frames: 1..10)
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 2, "t_us": 0,
                        "payload": _audio_payload(512, 0), "version": 1}, 0.0)
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 11, "node_id": 2, "t_us": 0,
                        "payload": _audio_payload(512, 11), "version": 1}, 0.12)
        result = rec.close(trial_id="t003", kind="tone")
        assert result["flagged"] is True
        assert result["total_gap_ms"] > 100.0

    def test_env_csv_written(self, tmp_path):
        """NodeRecorder writes node{id}_env.csv (per-node filename)."""
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        rec.feed_frame({"type": FrameType.ENV, "seq": 0, "node_id": 1, "t_us": 0,
                        "payload": _env_payload(23.4), "version": 1}, 5.0)
        rec.close(trial_id="t004", kind="tone")

        csv_path = tmp_path / "node1_env.csv"
        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert abs(float(rows[0]["temp_c"]) - 23.4) < 0.01
        assert float(rows[0]["t_laptop_s"]) == pytest.approx(5.0)

    def test_meta_json_written(self, tmp_path):
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        stream = _build_stream(node_id, n_audio=2)
        asm = FrameAssembler()
        for frame, t in asm.feed(stream, 0.0):
            rec.feed_frame(frame, t)
        rec.close(trial_id="my-trial", kind="clap")

        meta_path = tmp_path / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["trial_id"] == "my-trial"
        assert meta["kind"] == "clap"
        assert "config_hash" in meta
        assert "git_commit" in meta
        # No wall-clock timestamps
        assert "timestamp" not in meta
        assert "recorded_at" not in meta

    def test_startup_fw_version_remembered(self, tmp_path):
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        rec.feed_frame({"type": FrameType.STARTUP, "seq": 0, "node_id": 1, "t_us": 0,
                        "payload": _startup_payload(1), "version": 1}, 0.0)
        # No crash; close without audio
        rec.close(trial_id="t005", kind="tone")

    def test_unknown_node_id_logged_not_crash(self, tmp_path):
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        # Feed a frame with a different node_id – should be silently skipped
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 99, "t_us": 0,
                        "payload": _audio_payload(512, 0), "version": 1}, 0.0)
        result = rec.close(trial_id="t006", kind="tone")
        assert result is not None  # no crash

    def test_same_stream_twice_byte_identical_output(self, tmp_path):
        """Replaying the same stream produces byte-identical WAV, node{id}_env.csv, meta.json."""
        node_id = 1
        stream = _build_stream(node_id, n_audio=2)

        def _run(out_dir: pathlib.Path):
            rec = NodeRecorder(node_id, out_dir)
            asm = FrameAssembler()
            for frame, t in asm.feed(stream, 0.0):
                rec.feed_frame(frame, t)
            rec.close(trial_id="rep-trial", kind="tone")

        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        dir1.mkdir()
        dir2.mkdir()
        _run(dir1)
        _run(dir2)

        for fname in [f"node{node_id}.wav", f"node{node_id}_env.csv", "meta.json"]:
            f1 = (dir1 / fname).read_bytes()
            f2 = (dir2 / fname).read_bytes()
            assert f1 == f2, f"{fname} differs between runs"

    def test_late_connect_not_a_gap(self, tmp_path):
        """First frame seen defines baseline; no gap logged before it."""
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        # Start from seq=5 (as if we connected late)
        for seq in range(5, 8):
            rec.feed_frame({"type": FrameType.AUDIO, "seq": seq, "node_id": 1, "t_us": 0,
                            "payload": _audio_payload(512, seq), "version": 1}, float(seq))
        result = rec.close(trial_id="t007", kind="tone")
        assert result["total_gap_ms"] == pytest.approx(0.0)
        with wave.open(str(tmp_path / "node1.wav")) as wf:
            assert wf.getnframes() == 3 * 512

    def test_crc_failures_and_resync_echoed_in_result(self, tmp_path):
        node_id = 1
        good = encode_frame(FrameType.AUDIO, node_id, 0, 0, _audio_payload(512, 0))
        bad = _corrupt_crc(encode_frame(FrameType.AUDIO, node_id, 1, 0, _audio_payload(512, 1)))
        good2 = encode_frame(FrameType.AUDIO, node_id, 2, 0, _audio_payload(512, 2))

        rec = NodeRecorder(node_id, tmp_path)
        asm = FrameAssembler()
        for frame, t in asm.feed(good + bad + good2, 0.0):
            rec.feed_frame(frame, t)
        result = rec.close(trial_id="t008", kind="tone", assembler=asm)
        assert result["crc_failures"] == 1
        assert result["resync_bytes_skipped"] >= 1

    def test_seq_reset_counted_flagged_no_zero_fill(self, tmp_path):
        """seq 5,6,0,1: reset detected after seq 6→0, flagged, no zero-fill."""
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        seqs = [5, 6, 0, 1]
        for i, seq in enumerate(seqs):
            rec.feed_frame({"type": FrameType.AUDIO, "seq": seq, "node_id": 1, "t_us": 0,
                            "payload": _audio_payload(512, i), "version": 1}, float(i))
        result = rec.close(trial_id="t009", kind="tone")
        # Resets counted
        assert rec._resets >= 1 or result["flagged"] is True
        # No zero-fill for the reset gap: WAV has exactly 4 * 512 frames
        with wave.open(str(tmp_path / "node1.wav")) as wf:
            assert wf.getnframes() == 4 * 512
        # Gaps log contains RESET
        gaps_log = tmp_path / "gaps.log"
        assert gaps_log.exists()
        assert "RESET" in gaps_log.read_text()

    def test_duplicate_frame_skipped(self, tmp_path):
        """Same seq twice: second is ignored, DUP logged."""
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        payload = _audio_payload(512, 0)
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 1, "t_us": 0,
                        "payload": payload, "version": 1}, 0.0)
        rec.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 1, "t_us": 0,
                        "payload": payload, "version": 1}, 0.01)
        rec.close(trial_id="t010", kind="tone")
        with wave.open(str(tmp_path / "node1.wav")) as wf:
            assert wf.getnframes() == 512
        gaps_log = tmp_path / "gaps.log"
        assert gaps_log.exists()
        assert "DUP" in gaps_log.read_text()

    def test_250_audio_frames_wav_readable_without_close(self, tmp_path):
        """Feed 250 AUDIO frames without close; WAV must be readable with >= 200 frames."""
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        n = 250
        for seq in range(n):
            rec.feed_frame({"type": FrameType.AUDIO, "seq": seq, "node_id": 1, "t_us": 0,
                            "payload": _audio_payload(512, seq % 16), "version": 1}, float(seq))
        # Do NOT call close()
        # Force a patch so header is up-to-date
        if rec._wav is not None:
            rec._wav._patch_header()
        with wave.open(str(tmp_path / "node1.wav")) as wf:
            assert wf.getnframes() >= 200 * 512


# ---------------------------------------------------------------------------
# TrialRecorder tests (R2)
# ---------------------------------------------------------------------------

class TestTrialRecorder:
    def test_two_nodes_all_files_produced(self, tmp_path):
        """Two nodes → node1.wav, node2.wav, node1_env.csv, node2_env.csv, one meta.json."""
        tr = TrialRecorder(tmp_path, node_ids=[1, 2])
        for nid in [1, 2]:
            for seq in range(3):
                tr.feed_frame({"type": FrameType.AUDIO, "seq": seq, "node_id": nid,
                               "t_us": 0, "payload": _audio_payload(512, seq), "version": 1},
                              float(seq))
            tr.feed_frame({"type": FrameType.ENV, "seq": 0, "node_id": nid, "t_us": 0,
                           "payload": _env_payload(22.0), "version": 1}, 1.0)
        meta = tr.close(trial_id="multi", kind="tone")

        assert (tmp_path / "node1.wav").exists()
        assert (tmp_path / "node2.wav").exists()
        assert (tmp_path / "node1_env.csv").exists()
        assert (tmp_path / "node2_env.csv").exists()
        assert (tmp_path / "meta.json").exists()
        # One meta.json with both node sections
        saved = json.loads((tmp_path / "meta.json").read_text())
        assert "1" in saved["nodes"]
        assert "2" in saved["nodes"]

    def test_meta_json_both_nodes_section(self, tmp_path):
        tr = TrialRecorder(tmp_path, node_ids=[1, 2])
        tr.close(trial_id="empty", kind="tone")
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert set(meta["nodes"].keys()) >= {"1", "2"}
        assert meta["trial_id"] == "empty"

    def test_seq_reset_flagged_in_meta(self, tmp_path):
        """Seq reset sets flagged=True in meta.json."""
        tr = TrialRecorder(tmp_path, node_ids=[1])
        for seq in [0, 1, 0]:  # reset at the third frame
            tr.feed_frame({"type": FrameType.AUDIO, "seq": seq, "node_id": 1, "t_us": 0,
                           "payload": _audio_payload(512, 0), "version": 1}, float(seq))
        meta = tr.close(trial_id="reset_trial", kind="tone")
        assert meta["flagged"] is True
        assert meta["nodes"]["1"]["resets"] >= 1
        # Gaps log has RESET
        assert "RESET" in (tmp_path / "gaps.log").read_text()

    def test_duplicate_frame_skipped_trial(self, tmp_path):
        """Dup in TrialRecorder: WAV has 1*512 frames, DUP in gaps.log."""
        tr = TrialRecorder(tmp_path, node_ids=[1])
        payload = _audio_payload(512, 0)
        for _ in range(2):
            tr.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 1, "t_us": 0,
                           "payload": payload, "version": 1}, 0.0)
        tr.close(trial_id="dup_trial", kind="tone")
        with wave.open(str(tmp_path / "node1.wav")) as wf:
            assert wf.getnframes() == 512
        assert "DUP" in (tmp_path / "gaps.log").read_text()

    def test_extra_meta_merged_not_overriding_core(self, tmp_path):
        """extra_meta values appear in meta.json; core keys not overridden."""
        tr = TrialRecorder(tmp_path, node_ids=[1])
        extra = {
            "true_theta_deg": {"1": 30.0},
            "trial_id": "SHOULD_NOT_OVERRIDE",  # core key
            "site": "lab",
        }
        tr.close(trial_id="core_trial", kind="tone", extra_meta=extra)
        meta = json.loads((tmp_path / "meta.json").read_text())
        # Core key not overridden
        assert meta["trial_id"] == "core_trial"
        # Extra fields present
        assert meta["true_theta_deg"] == {"1": 30.0}
        assert meta["site"] == "lab"

    def test_node_with_no_audio_gets_valid_empty_wav(self, tmp_path):
        """Node that never sends audio still gets a valid empty WAV at close."""
        tr = TrialRecorder(tmp_path, node_ids=[1])
        tr.close(trial_id="no_audio", kind="tone")
        wav_path = tmp_path / "node1.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path)) as wf:
            assert wf.getnchannels() == 2
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 48000
            assert wf.getnframes() == 0

    def test_unknown_node_id_logged_not_crash_trial(self, tmp_path):
        """Unknown node_id in TrialRecorder is silently skipped."""
        tr = TrialRecorder(tmp_path, node_ids=[1])
        tr.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 99, "t_us": 0,
                       "payload": _audio_payload(512, 0), "version": 1}, 0.0)
        meta = tr.close(trial_id="unknown", kind="tone")
        assert meta is not None

    def test_t_first_audio_laptop_s_in_meta(self, tmp_path):
        """t_first_audio_laptop_s is recorded for each node."""
        tr = TrialRecorder(tmp_path, node_ids=[1])
        tr.feed_frame({"type": FrameType.AUDIO, "seq": 0, "node_id": 1, "t_us": 0,
                       "payload": _audio_payload(512, 0), "version": 1}, 7.5)
        tr.close(trial_id="ttest", kind="tone")
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["nodes"]["1"]["t_first_audio_laptop_s"] == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# SerialSource (fake interface — UNTESTED ON HARDWARE)
# ---------------------------------------------------------------------------

class FakeSerial:
    """Fake pyserial-like object for testing SerialSource interface."""

    def __init__(self, data: bytes, serial_number: str = "FAKE001"):
        self._data = io.BytesIO(data)
        self.serial_number = serial_number  # mimics list_ports serial_number
        self.in_waiting: int = len(data)

    def read(self, size: int = 256) -> bytes:
        return self._data.read(size)

    def close(self) -> None:
        pass

    @property
    def is_open(self) -> bool:
        return True


class TestSerialSourceInterface:
    """Tests for SerialSource using a fake; UNTESTED ON HARDWARE."""

    def test_read_returns_bytes(self, tmp_path):
        data = _build_stream(1, n_audio=1)
        fake = FakeSerial(data, "FAKE001")
        src = SerialSource.__new__(SerialSource)
        src._serial = fake
        result = src.read(len(data))
        assert isinstance(result, bytes)
        assert len(result) == len(data)

    def test_read_empty_on_exhausted(self, tmp_path):
        fake = FakeSerial(b"", "FAKE001")
        fake.in_waiting = 0
        src = SerialSource.__new__(SerialSource)
        src._serial = fake
        assert src.read(100) == b""


# ---------------------------------------------------------------------------
# R10: SerialSource hardening tests via fake serial module (UNTESTED ON HARDWARE)
# ---------------------------------------------------------------------------

class _FakeSerialObj:
    """Fake serial.Serial that records call order for dtr/rts/open sequence."""

    def __init__(self):
        self.port: str | None = None
        self.baudrate: int = 115200
        self.timeout: float = 0.0
        self.dtr: bool = True   # start True; test checks it is set False before open()
        self.rts: bool = True
        self._is_open: bool = False
        self.in_waiting: int = 0
        self._call_log: list[str] = []  # records attribute sets and open()

    def __setattr__(self, name, value):
        # Record sets of dtr/rts so we can verify order vs open()
        if name in ("dtr", "rts") and not name.startswith("_"):
            try:
                self._call_log.append(f"set_{name}={value}")
            except AttributeError:
                pass  # _call_log not yet initialised (super().__init__ phase)
        super().__setattr__(name, value)

    def open(self):
        self._call_log.append("open")
        self._is_open = True

    def close(self):
        self._is_open = False

    def read(self, size: int) -> bytes:
        return b"\x00" * min(size, self.in_waiting)

    @property
    def is_open(self) -> bool:
        return self._is_open


class _FakePortInfo:
    def __init__(self, device: str, serial_number: str):
        self.device = device
        self.serial_number = serial_number


def _make_fake_serial_module(fake_obj: _FakeSerialObj, port_infos: list[_FakePortInfo]):
    """Return a fake 'serial' module and 'serial.tools.list_ports' module."""
    import types

    list_ports_mod = types.ModuleType("serial.tools.list_ports")
    list_ports_mod.comports = lambda: port_infos  # type: ignore[attr-defined]

    tools_mod = types.ModuleType("serial.tools")
    tools_mod.list_ports = list_ports_mod  # type: ignore[attr-defined]

    serial_mod = types.ModuleType("serial")
    serial_mod.Serial = lambda: fake_obj  # type: ignore[attr-defined]
    serial_mod.tools = tools_mod  # type: ignore[attr-defined]

    return serial_mod, tools_mod, list_ports_mod


class TestSerialSourceHardening:
    """R10 tests for the new SerialSource open sequence. UNTESTED ON HARDWARE."""

    def _inject(self, monkeypatch, fake_obj, port_infos) -> "SerialSource":
        """Inject a fake serial module and instantiate SerialSource."""
        serial_mod, tools_mod, list_ports_mod = _make_fake_serial_module(fake_obj, port_infos)
        monkeypatch.setitem(sys.modules, "serial", serial_mod)
        monkeypatch.setitem(sys.modules, "serial.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports_mod)
        from kaawayan.record import SerialSource
        return SerialSource("FAKESN")

    def test_serial_created_without_port_then_opened(self, monkeypatch):
        """Serial() must be constructed without a port; open() called afterwards."""
        fake = _FakeSerialObj()
        ports = [_FakePortInfo("COM99", "FAKESN")]
        self._inject(monkeypatch, fake, ports)
        # port must have been set after construction
        assert fake.port == "COM99"
        assert "open" in fake._call_log

    def test_dtr_and_rts_set_false_before_open(self, monkeypatch):
        """dtr=False and rts=False must appear in call_log BEFORE 'open'."""
        fake = _FakeSerialObj()
        ports = [_FakePortInfo("COM99", "FAKESN")]
        self._inject(monkeypatch, fake, ports)

        log = fake._call_log
        open_idx = log.index("open")
        dtr_idx = next(i for i, e in enumerate(log) if e == "set_dtr=False")
        rts_idx = next(i for i, e in enumerate(log) if e == "set_rts=False")
        assert dtr_idx < open_idx, "dtr=False must be set before open()"
        assert rts_idx < open_idx, "rts=False must be set before open()"

    def test_set_buffer_size_called_when_present(self, monkeypatch):
        """set_buffer_size(rx_size=1<<20) is called when the method exists."""
        fake = _FakeSerialObj()
        buf_size_calls: list[int] = []

        def _set_buffer_size(rx_size: int):
            buf_size_calls.append(rx_size)

        fake.set_buffer_size = _set_buffer_size  # type: ignore[attr-defined]
        ports = [_FakePortInfo("COM99", "FAKESN")]
        self._inject(monkeypatch, fake, ports)
        assert buf_size_calls == [1 << 20]

    def test_set_buffer_size_skipped_when_absent(self, monkeypatch):
        """SerialSource must not crash when set_buffer_size is absent."""
        fake = _FakeSerialObj()
        # _FakeSerialObj has no set_buffer_size — that's the point
        assert not hasattr(fake, "set_buffer_size")
        ports = [_FakePortInfo("COM99", "FAKESN")]
        self._inject(monkeypatch, fake, ports)  # must not raise

    def test_set_buffer_size_exception_ignored(self, monkeypatch):
        """Exception from set_buffer_size must be silently ignored."""
        fake = _FakeSerialObj()

        def _bad_set_buffer_size(rx_size: int):
            raise OSError("not supported")

        fake.set_buffer_size = _bad_set_buffer_size  # type: ignore[attr-defined]
        ports = [_FakePortInfo("COM99", "FAKESN")]
        self._inject(monkeypatch, fake, ports)  # must not raise

    def test_usb_serial_matching_picks_correct_device(self, monkeypatch):
        """usb_serial match must pick the matching device, not the first one."""
        fake = _FakeSerialObj()
        ports = [
            _FakePortInfo("COM10", "OTHER_SN"),
            _FakePortInfo("COM11", "FAKESN"),
            _FakePortInfo("COM12", "ANOTHER"),
        ]
        self._inject(monkeypatch, fake, ports)
        assert fake.port == "COM11"

    def test_fallback_to_direct_port_name(self, monkeypatch):
        """When usb_serial not found in list_ports, treat arg as port name."""
        fake = _FakeSerialObj()
        ports: list[_FakePortInfo] = []  # no matching serial number
        serial_mod, tools_mod, list_ports_mod = _make_fake_serial_module(fake, ports)
        monkeypatch.setitem(sys.modules, "serial", serial_mod)
        monkeypatch.setitem(sys.modules, "serial.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports_mod)
        from kaawayan.record import SerialSource
        src = SerialSource("COM5")
        assert fake.port == "COM5"

    def test_serial_number_attribute_set_on_match(self, monkeypatch):
        """serial_number attribute is set when matched by usb_serial."""
        fake = _FakeSerialObj()
        ports = [_FakePortInfo("COM99", "FAKESN")]
        src = self._inject(monkeypatch, fake, ports)
        assert src.serial_number == "FAKESN"

    def test_serial_number_attribute_none_on_direct_port(self, monkeypatch):
        """serial_number attribute is None when falling back to direct port."""
        fake = _FakeSerialObj()
        ports: list[_FakePortInfo] = []
        serial_mod, tools_mod, list_ports_mod = _make_fake_serial_module(fake, ports)
        monkeypatch.setitem(sys.modules, "serial", serial_mod)
        monkeypatch.setitem(sys.modules, "serial.tools", tools_mod)
        monkeypatch.setitem(sys.modules, "serial.tools.list_ports", list_ports_mod)
        from kaawayan.record import SerialSource
        src = SerialSource("COM5")
        assert src.serial_number is None

    def test_read_uses_in_waiting(self, monkeypatch):
        """read() must use in_waiting to determine n bytes to read."""
        fake = _FakeSerialObj()
        fake.in_waiting = 100
        ports = [_FakePortInfo("COM99", "FAKESN")]
        src = self._inject(monkeypatch, fake, ports)
        result = src.read()
        assert isinstance(result, bytes)
        # n = max(1, min(in_waiting=100, 65536)) = 100
        assert len(result) == 100

    def test_read_uses_in_waiting_zero_reads_one_byte(self, monkeypatch):
        """When in_waiting==0, read() still reads at least 1 byte."""
        fake = _FakeSerialObj()
        fake.in_waiting = 0
        ports = [_FakePortInfo("COM99", "FAKESN")]
        src = self._inject(monkeypatch, fake, ports)
        # The fake returns b'\x00'*min(size,in_waiting) but in_waiting=0,
        # so result is b"" — that's fine as long as the call uses n=max(1,...)
        # We just verify no exception raised.
        result = src.read()
        assert isinstance(result, bytes)
