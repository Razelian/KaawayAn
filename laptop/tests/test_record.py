"""
Tests for kaawayan.record — Slice 3a.

Tests:
- FrameAssembler: clean stream, split-at-every-byte, CRC failure recovery,
  crc_failures/resync_bytes_skipped counts.
- NodeRecorder: clean stream; one dropped AUDIO frame (gap logged, zeros
  inserted, WAV length correct); a trial with >100 ms gaps is flagged;
  saving same stream twice gives byte-identical WAV, env.csv, meta.json.
- SerialSource: tested through a fake (UNTESTED ON HARDWARE).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pathlib
import struct
import wave
import zlib

import numpy as np
import pytest

from kaawayan.protocol import (
    FrameType,
    encode_frame,
    _STARTUP_FMT,
    _ENV_FMT,
)
from kaawayan.record import FrameAssembler, NodeRecorder, SerialSource


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
        assert "node1" in content or "node_id=1" in content

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
        node_id = 1
        rec = NodeRecorder(node_id, tmp_path)
        rec.feed_frame({"type": FrameType.ENV, "seq": 0, "node_id": 1, "t_us": 0,
                        "payload": _env_payload(23.4), "version": 1}, 5.0)
        rec.close(trial_id="t004", kind="tone")

        csv_path = tmp_path / "env.csv"
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
        """Replaying the same stream produces byte-identical WAV, env.csv, meta.json."""
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

        for fname in [f"node{node_id}.wav", "env.csv", "meta.json"]:
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


# ---------------------------------------------------------------------------
# SerialSource (fake interface — UNTESTED ON HARDWARE)
# ---------------------------------------------------------------------------

class FakeSerial:
    """Fake pyserial-like object for testing SerialSource interface."""

    def __init__(self, data: bytes, serial_number: str = "FAKE001"):
        self._data = io.BytesIO(data)
        self.serial_number = serial_number  # mimics list_ports serial_number

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
        src = SerialSource.__new__(SerialSource)
        src._serial = fake
        assert src.read(100) == b""
