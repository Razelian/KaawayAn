"""
tests/test_golden_fw.py — Golden frame tests for firmware frame validation.

Dump format (tests/golden/frames_fw.hex):
    One frame per line, text file, format "TYPE:hex"
    where TYPE is STARTUP, AUDIO, or ENV and hex is the complete
    raw serial bytes for that frame (including SYNC header and CRC-32).
    This file is produced by capturing raw serial bytes from the firmware
    and converting each frame to hex.

If tests/golden/frames_fw.hex does not exist, all tests in this module
are skipped with a clear message.

Two self-contained tests always run (no file required):

1. test_gen_frames_hex_matches_file:
   Regenerates the golden frames in memory using gen_frames.py's make_*
   functions and compares with the on-disk frames.hex byte-by-byte.
   This verifies that gen_frames.py changes (e.g. mic_flags unsigned B)
   do not silently corrupt the golden file.

2. test_frames_fw (conditional on frames_fw.hex existing):
   Parses each "TYPE:hex" line, decodes with protocol.decode_frame,
   asserts CRC valid (decode_frame raises on mismatch), frame type matches
   the label, AUDIO payloads have exactly 512 stereo int16 samples at 48 kHz,
   and per-type seq numbers in the file are strictly consecutive (no gaps).
"""

from __future__ import annotations

import pathlib
import struct
import sys

import pytest

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "golden"
FRAMES_HEX = GOLDEN_DIR / "frames.hex"
FRAMES_FW_HEX = GOLDEN_DIR / "frames_fw.hex"

from kaawayan.protocol import decode_frame, FrameType

_FS = 48_000
_SAMPLES_PER_AUDIO = 512
_AUDIO_PAYLOAD_BYTES = _SAMPLES_PER_AUDIO * 2 * 2  # 512 stereo int16 = 2048 bytes


# ---------------------------------------------------------------------------
# Helper: parse a frames.hex (or frames_fw.hex) file
# ---------------------------------------------------------------------------

def _parse_hex_file(path: pathlib.Path) -> list[tuple[str, bytes]]:
    """Parse a TYPE:hex text file into list of (type_name, raw_bytes)."""
    frames = []
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid line in {path}: {line!r}")
        label, hex_str = line.split(":", 1)
        frames.append((label.strip(), bytes.fromhex(hex_str.strip())))
    return frames


# ---------------------------------------------------------------------------
# Test 1: regenerate golden frames in memory and compare with frames.hex
# ---------------------------------------------------------------------------

class TestGenFramesHex:
    def test_gen_frames_hex_matches_file(self) -> None:
        """gen_frames.py (in-memory) must produce bytes identical to frames.hex."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gen_frames", GOLDEN_DIR / "gen_frames.py"
        )
        gen = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(gen)  # type: ignore[union-attr]

        # Regenerate frames
        in_memory: dict[str, bytes] = {
            "STARTUP": gen.make_startup(),
            "AUDIO":   gen.make_audio(),
            "ENV":     gen.make_env(),
        }

        # Load on-disk
        on_disk = {label: data for label, data in _parse_hex_file(FRAMES_HEX)}

        for name in ["STARTUP", "AUDIO", "ENV"]:
            assert name in on_disk, f"Missing {name} in frames.hex"
            assert in_memory[name] == on_disk[name], (
                f"{name} frame mismatch:\n"
                f"  in-memory: {in_memory[name].hex()}\n"
                f"  on-disk:   {on_disk[name].hex()}"
            )


# ---------------------------------------------------------------------------
# Test 2: firmware dump (skipped if frames_fw.hex absent)
# ---------------------------------------------------------------------------

_FW_REASON = (
    "tests/golden/frames_fw.hex not found; "
    "provide a firmware frame dump to enable this test. "
    "Format: one frame per line, 'TYPE:hex', raw serial bytes including SYNC and CRC."
)


@pytest.mark.skipif(not FRAMES_FW_HEX.exists(), reason=_FW_REASON)
class TestFramesFwHex:
    def _frames(self) -> list[tuple[str, bytes]]:
        return _parse_hex_file(FRAMES_FW_HEX)

    def test_crc_valid_for_all_frames(self) -> None:
        """decode_frame must succeed for every frame (raises on CRC mismatch)."""
        for label, raw in self._frames():
            try:
                decode_frame(raw)
            except Exception as exc:
                pytest.fail(f"{label} frame: decode_frame raised: {exc}")

    def test_type_matches_label(self) -> None:
        """Decoded frame type must match the TYPE label in the file."""
        label_to_type = {
            "STARTUP": FrameType.STARTUP,
            "AUDIO":   FrameType.AUDIO,
            "ENV":     FrameType.ENV,
        }
        for label, raw in self._frames():
            expected = label_to_type.get(label)
            if expected is None:
                pytest.fail(f"Unknown frame label {label!r}")
            frame = decode_frame(raw)
            assert frame["type"] == expected, (
                f"Label {label}: expected type {expected}, got {frame['type']}"
            )

    def test_audio_payload_512_stereo_samples(self) -> None:
        """Every AUDIO frame in the dump has exactly 512 stereo int16 samples."""
        for label, raw in self._frames():
            if label != "AUDIO":
                continue
            frame = decode_frame(raw)
            payload = frame["payload"]
            assert len(payload) == _AUDIO_PAYLOAD_BYTES, (
                f"AUDIO payload {len(payload)} bytes, expected {_AUDIO_PAYLOAD_BYTES}"
            )
            # Verify it can be reshaped to (512, 2) int16
            arr = __import__("numpy").frombuffer(payload, dtype="<i2").reshape(-1, 2)
            assert arr.shape == (_SAMPLES_PER_AUDIO, 2), (
                f"AUDIO array shape {arr.shape} != ({_SAMPLES_PER_AUDIO}, 2)"
            )

    def test_seq_numbers_consecutive_per_type(self) -> None:
        """Per-type seq numbers in the dump must be consecutive (no gaps)."""
        last_seq: dict[FrameType, int] = {}
        for label, raw in self._frames():
            frame = decode_frame(raw)
            ftype = frame["type"]
            seq = frame["seq"]
            if ftype in last_seq:
                expected = last_seq[ftype] + 1
                assert seq == expected, (
                    f"{label}: seq jumped from {last_seq[ftype]} to {seq} "
                    f"(expected {expected}); firmware dump has gaps"
                )
            last_seq[ftype] = seq
