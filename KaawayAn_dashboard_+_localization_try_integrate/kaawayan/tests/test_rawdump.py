"""
tests/test_rawdump.py — Tests for kaawayan.rawdump.

Coverage:
- split_raw_frames: garbage before/after frames, corrupt frame skipped, raw bytes match
- select_golden: STARTUP+ENV+AUDIO selection, consecutive-seq logic, missing types
- --force behaviour
- CLI through monkeypatched SerialSource
- Written file passes the same 4 checks as test_golden_fw.TestFramesFwHex
"""

from __future__ import annotations

import pathlib
import struct
import sys

import pytest

from kaawayan.protocol import (
    FrameType,
    HDR_SIZE,
    CRC_SIZE,
    encode_frame,
    decode_frame,
)
from kaawayan.rawdump import split_raw_frames, select_golden


# ---------------------------------------------------------------------------
# Helpers: build minimal valid frames
# ---------------------------------------------------------------------------

_STARTUP_PAYLOAD = struct.pack("<BBBBIIHBb", 1, 0, 1, 0, 0xDEAD, 48000, 512, 2, 0)
_ENV_PAYLOAD     = struct.pack("<hHIIB", 2500, 5000, 0, 0, 0)
_AUDIO_PAYLOAD   = bytes(512 * 2 * 2)  # 2048 zero bytes


def _make_startup(node_id: int = 1, seq: int = 0) -> bytes:
    return encode_frame(FrameType.STARTUP, node_id, seq, 0, _STARTUP_PAYLOAD)


def _make_env(node_id: int = 1, seq: int = 0) -> bytes:
    return encode_frame(FrameType.ENV, node_id, seq, 0, _ENV_PAYLOAD)


def _make_audio(node_id: int = 1, seq: int = 0) -> bytes:
    return encode_frame(FrameType.AUDIO, node_id, seq, 0, _AUDIO_PAYLOAD)


# ---------------------------------------------------------------------------
# Tests: split_raw_frames
# ---------------------------------------------------------------------------

class TestSplitRawFrames:
    def test_empty_bytes(self) -> None:
        assert split_raw_frames(b"") == []

    def test_single_startup(self) -> None:
        raw = _make_startup()
        result = split_raw_frames(raw)
        assert len(result) == 1
        frame, raw_bytes = result[0]
        assert frame["type"] == FrameType.STARTUP
        assert raw_bytes == raw  # exact original bytes

    def test_raw_bytes_never_re_encoded(self) -> None:
        """raw_bytes must be a slice of the input, identical to the encoded frame."""
        s = _make_startup(seq=7)
        e = _make_env(seq=3)
        data = s + e
        result = split_raw_frames(data)
        assert len(result) == 2
        assert result[0][1] == s
        assert result[1][1] == e

    def test_garbage_before_frame_skipped(self) -> None:
        garbage = b"\x00\x01\x02\x03\xFF\xFE"
        raw = _make_audio(seq=0)
        result = split_raw_frames(garbage + raw)
        assert len(result) == 1
        assert result[0][1] == raw

    def test_garbage_after_frame_ignored(self) -> None:
        raw = _make_audio(seq=0)
        result = split_raw_frames(raw + b"\xFF\xFE\x00")
        assert len(result) == 1
        assert result[0][1] == raw

    def test_corrupt_frame_between_good_frames_skipped(self) -> None:
        """A CRC-corrupted frame between two good frames must be skipped."""
        good1 = _make_startup(seq=0)
        good2 = _make_env(seq=0)
        # Build a corrupt frame: take a valid frame and flip the last CRC byte
        corrupt = bytearray(_make_audio(seq=99))
        corrupt[-1] ^= 0xFF
        data = good1 + bytes(corrupt) + good2
        result = split_raw_frames(data)
        # Both good frames must appear; corrupt one is skipped
        assert len(result) == 2
        assert result[0][1] == good1
        assert result[1][1] == good2

    def test_multiple_audio_frames_consecutive(self) -> None:
        frames_raw = [_make_audio(seq=i) for i in range(5)]
        data = b"".join(frames_raw)
        result = split_raw_frames(data)
        assert len(result) == 5
        for i, (frame, raw) in enumerate(result):
            assert frame["type"] == FrameType.AUDIO
            assert frame["seq"] == i
            assert raw == frames_raw[i]

    def test_truncated_at_end_does_not_crash(self) -> None:
        raw = _make_audio(seq=0)
        # Feed only half of a frame (but after a complete startup)
        s = _make_startup()
        result = split_raw_frames(s + raw[:len(raw) // 2])
        assert len(result) == 1  # only startup; truncated audio not returned

    def test_garbage_only_returns_empty(self) -> None:
        result = split_raw_frames(b"\xFF" * 100)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: select_golden
# ---------------------------------------------------------------------------

class TestSelectGolden:
    def _frames(self, seq_list: list[int]) -> list[tuple[dict, bytes]]:
        out = []
        for seq in seq_list:
            raw = _make_audio(seq=seq)
            frame = decode_frame(raw)
            out.append((frame, raw))
        return out

    def test_startup_env_and_n_audio_consecutive(self) -> None:
        s_raw = _make_startup(seq=0)
        e_raw = _make_env(seq=0)
        a_raws = [_make_audio(seq=i) for i in range(10)]
        all_frames: list[tuple[dict, bytes]] = []
        for raw in [s_raw, e_raw] + a_raws:
            all_frames.append((decode_frame(raw), raw))

        result = select_golden(all_frames, n_audio=8)
        assert result[0] == ("STARTUP", s_raw)
        assert result[1] == ("ENV", e_raw)
        assert len(result) == 10  # STARTUP + ENV + 8 AUDIO
        for i, (name, raw) in enumerate(result[2:]):
            assert name == "AUDIO"
            assert raw == a_raws[i]

    def test_missing_startup_omitted(self) -> None:
        e_raw = _make_env(seq=0)
        a_raws = [_make_audio(seq=i) for i in range(3)]
        all_frames = [(decode_frame(r), r) for r in [e_raw] + a_raws]
        result = select_golden(all_frames, n_audio=3)
        assert result[0][0] == "ENV"
        assert all(name == "AUDIO" for name, _ in result[1:])

    def test_missing_env_omitted(self) -> None:
        s_raw = _make_startup()
        a_raws = [_make_audio(seq=i) for i in range(3)]
        all_frames = [(decode_frame(r), r) for r in [s_raw] + a_raws]
        result = select_golden(all_frames, n_audio=3)
        assert result[0][0] == "STARTUP"
        assert all(name == "AUDIO" for name, _ in result[1:])

    def test_missing_both_startup_and_env(self) -> None:
        a_raws = [_make_audio(seq=i) for i in range(4)]
        all_frames = [(decode_frame(r), r) for r in a_raws]
        result = select_golden(all_frames, n_audio=4)
        assert all(name == "AUDIO" for name, _ in result)
        assert len(result) == 4

    def test_first_run_shorter_uses_longest_run(self, capsys) -> None:
        """If first consecutive run is shorter, longest run is used and notice printed."""
        # seq: 0,1,2, gap, 10,11,12,13,14 — first run=3, longest=5
        seqs = [0, 1, 2, 10, 11, 12, 13, 14]
        all_frames = self._frames(seqs)
        result = select_golden(all_frames, n_audio=5)
        audio_results = [(n, r) for n, r in result if n == "AUDIO"]
        # Should use run starting at seq=10
        assert len(audio_results) == 5
        for i, (_, raw) in enumerate(audio_results):
            f = decode_frame(raw)
            assert f["seq"] == 10 + i
        # Notice should be printed
        captured = capsys.readouterr()
        assert "notice" in captured.err.lower()

    def test_first_run_exactly_n_audio(self) -> None:
        seqs = [0, 1, 2, 3, 4, 100, 101]  # first run = 5
        all_frames = self._frames(seqs)
        result = select_golden(all_frames, n_audio=5)
        audio_results = [r for n, r in result if n == "AUDIO"]
        assert len(audio_results) == 5
        seqs_out = [decode_frame(r)["seq"] for r in audio_results]
        assert seqs_out == [0, 1, 2, 3, 4]

    def test_no_audio_frames(self) -> None:
        s_raw = _make_startup()
        e_raw = _make_env()
        all_frames = [(decode_frame(r), r) for r in [s_raw, e_raw]]
        result = select_golden(all_frames, n_audio=8)
        assert len(result) == 2
        assert result[0][0] == "STARTUP"
        assert result[1][0] == "ENV"

    def test_output_order_startup_env_audio(self) -> None:
        # ENV before STARTUP in stream — output order must still be STARTUP, ENV, AUDIO
        e_raw = _make_env(seq=0)
        s_raw = _make_startup(seq=0)
        a_raw = _make_audio(seq=0)
        # Deliberately put ENV first in stream
        all_frames = [(decode_frame(r), r) for r in [e_raw, s_raw, a_raw]]
        result = select_golden(all_frames, n_audio=1)
        names = [n for n, _ in result]
        assert names == ["STARTUP", "ENV", "AUDIO"]


# ---------------------------------------------------------------------------
# Tests: --force behaviour and CLI
# ---------------------------------------------------------------------------

class TestForce:
    def test_refuses_overwrite_without_force(self, tmp_path, monkeypatch) -> None:
        """CLI exits with error if output exists and --force not given."""
        import io
        out_file = tmp_path / "frames_fw.hex"
        out_file.write_text("STARTUP:aabb\n", encoding="ascii")

        from kaawayan import rawdump

        monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

        with pytest.raises(SystemExit) as exc_info:
            rawdump.main([
                "--serial", "FAKE",
                "--seconds", "0.01",
                "--out", str(out_file),
            ])
        assert exc_info.value.code != 0

    def test_force_overwrites_existing_file(self, tmp_path, monkeypatch) -> None:
        """--force overwrites without error."""
        out_file = tmp_path / "frames_fw.hex"
        out_file.write_text("STARTUP:aabb\n", encoding="ascii")

        # Build fake raw data: STARTUP + ENV + 8 AUDIO
        s_raw = _make_startup()
        e_raw = _make_env()
        a_raws = [_make_audio(seq=i) for i in range(8)]
        fake_data = s_raw + e_raw + b"".join(a_raws)

        _patch_serial_source(monkeypatch, fake_data)

        from kaawayan import rawdump
        rawdump.main([
            "--serial", "FAKESN",
            "--seconds", "0.01",
            "--out", str(out_file),
            "--force",
        ])
        content = out_file.read_text(encoding="ascii")
        assert "STARTUP:" in content
        assert "AUDIO:" in content


# ---------------------------------------------------------------------------
# Monkeypatch helper for CLI tests
# ---------------------------------------------------------------------------

def _patch_serial_source(monkeypatch, raw_data: bytes) -> None:
    """Replace SerialSource with a fake that returns raw_data in one read."""

    class _FakeSource:
        def __init__(self, *args, **kwargs):
            self._data = raw_data
            self._done = False

        def read(self, size: int = 4096) -> bytes:
            if self._done:
                return b""
            self._done = True
            return self._data

        @property
        def is_open(self) -> bool:
            return not self._done

        def close(self) -> None:
            self._done = True

    import kaawayan.rawdump as _rawdump
    import kaawayan.record as _record
    monkeypatch.setattr(_record, "SerialSource", _FakeSource)


class TestCLI:
    def test_cli_writes_correct_format(self, tmp_path, monkeypatch) -> None:
        """CLI writes TYPE:hex lines; written file parses correctly."""
        out_file = tmp_path / "frames_fw.hex"

        s_raw = _make_startup()
        e_raw = _make_env()
        a_raws = [_make_audio(seq=i) for i in range(8)]
        fake_data = s_raw + e_raw + b"".join(a_raws)

        _patch_serial_source(monkeypatch, fake_data)

        from kaawayan import rawdump
        rawdump.main([
            "--serial", "FAKESN",
            "--seconds", "0.01",
            "--out", str(out_file),
        ])

        assert out_file.exists()
        lines = [l.strip() for l in out_file.read_text(encoding="ascii").splitlines() if l.strip()]
        assert len(lines) == 10  # STARTUP + ENV + 8 AUDIO
        assert lines[0].startswith("STARTUP:")
        assert lines[1].startswith("ENV:")
        for line in lines[2:]:
            assert line.startswith("AUDIO:")

    def test_cli_raw_bytes_match_original(self, tmp_path, monkeypatch) -> None:
        """Hex in written file must be the original encoded bytes, not re-encoded."""
        out_file = tmp_path / "frames_fw.hex"

        s_raw = _make_startup(seq=42)
        e_raw = _make_env(seq=7)
        a_raws = [_make_audio(seq=i) for i in range(3)]
        fake_data = s_raw + e_raw + b"".join(a_raws)

        _patch_serial_source(monkeypatch, fake_data)

        from kaawayan import rawdump
        rawdump.main([
            "--serial", "FAKESN",
            "--seconds", "0.01",
            "--out", str(out_file),
            "--n-audio", "3",
        ])

        lines = [l.strip() for l in out_file.read_text(encoding="ascii").splitlines() if l.strip()]
        startup_hex = lines[0].split(":", 1)[1]
        assert bytes.fromhex(startup_hex) == s_raw  # exact original bytes


# ---------------------------------------------------------------------------
# Tests: written file passes the 4 checks from test_golden_fw.TestFramesFwHex
# ---------------------------------------------------------------------------

class TestWrittenFilePasses4Checks:
    """Replicate the 4 assertions of TestFramesFwHex on a file written by rawdump."""

    def _write_file(self, tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
        out_file = tmp_path / "frames_fw.hex"
        s_raw = _make_startup(seq=0)
        e_raw = _make_env(seq=0)
        a_raws = [_make_audio(seq=i) for i in range(8)]
        fake_data = s_raw + e_raw + b"".join(a_raws)
        _patch_serial_source(monkeypatch, fake_data)
        from kaawayan import rawdump
        rawdump.main([
            "--serial", "X",
            "--seconds", "0.01",
            "--out", str(out_file),
        ])
        return out_file

    def _parse(self, path: pathlib.Path) -> list[tuple[str, bytes]]:
        frames = []
        for line in path.read_text(encoding="ascii").splitlines():
            line = line.strip()
            if not line:
                continue
            label, hex_str = line.split(":", 1)
            frames.append((label.strip(), bytes.fromhex(hex_str.strip())))
        return frames

    def test_crc_valid_for_all_frames(self, tmp_path, monkeypatch) -> None:
        path = self._write_file(tmp_path, monkeypatch)
        for label, raw in self._parse(path):
            decode_frame(raw)  # raises on CRC mismatch

    def test_type_matches_label(self, tmp_path, monkeypatch) -> None:
        path = self._write_file(tmp_path, monkeypatch)
        label_to_type = {
            "STARTUP": FrameType.STARTUP,
            "AUDIO":   FrameType.AUDIO,
            "ENV":     FrameType.ENV,
        }
        for label, raw in self._parse(path):
            frame = decode_frame(raw)
            assert frame["type"] == label_to_type[label]

    def test_audio_payload_512_stereo_samples(self, tmp_path, monkeypatch) -> None:
        import numpy as np
        path = self._write_file(tmp_path, monkeypatch)
        for label, raw in self._parse(path):
            if label != "AUDIO":
                continue
            frame = decode_frame(raw)
            payload = frame["payload"]
            assert len(payload) == 512 * 2 * 2
            arr = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
            assert arr.shape == (512, 2)

    def test_seq_numbers_consecutive_per_type(self, tmp_path, monkeypatch) -> None:
        path = self._write_file(tmp_path, monkeypatch)
        last_seq: dict[FrameType, int] = {}
        for label, raw in self._parse(path):
            frame = decode_frame(raw)
            ftype = frame["type"]
            seq = frame["seq"]
            if ftype in last_seq:
                assert seq == last_seq[ftype] + 1
            last_seq[ftype] = seq
