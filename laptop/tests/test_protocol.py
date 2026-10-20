"""Tests for kaawayan.protocol – written BEFORE the implementation."""
import struct
import zlib
import pytest
import numpy as np
from kaawayan.protocol import (
    SYNC,
    HDR_FMT,
    HDR_SIZE,
    encode_frame,
    decode_frame,
    parse_stream,
    FrameType,
    AudioPayload,
    EnvPayload,
    StartupPayload,
    _STARTUP_FMT,
)

# ---------------------------------------------------------------------------
# 1. CRC-32 golden vector (CONTEXT.md §4)
# ---------------------------------------------------------------------------

def test_crc32_golden():
    assert zlib.crc32(b"123456789") & 0xFFFFFFFF == 0xCBF43926


# ---------------------------------------------------------------------------
# 2. Round-trip tests for each frame type
# ---------------------------------------------------------------------------

def _roundtrip(payload_bytes: bytes, frame_type: FrameType, node_id: int = 1, seq: int = 0, t_us: int = 0):
    raw = encode_frame(frame_type, node_id, seq, t_us, payload_bytes)
    frame = decode_frame(raw)
    assert frame["type"] == frame_type
    assert frame["node_id"] == node_id
    assert frame["seq"] == seq
    assert frame["t_us"] == t_us
    assert frame["payload"] == payload_bytes
    return frame


def test_audio_roundtrip():
    # 512 stereo int16 pairs = 2048 bytes
    samples = bytes(range(256)) * 8  # 2048 bytes
    _roundtrip(samples, FrameType.AUDIO, node_id=2, seq=7, t_us=123456)


def test_env_roundtrip():
    # <hHIIB  (13 bytes): temp_c_x100 i16, rh_x100 u16, overrun u32, usb_drop u32, bme_status u8
    payload = struct.pack("<hHIIB", -500, 6000, 0, 0, 1)
    _roundtrip(payload, FrameType.ENV, node_id=1, seq=3, t_us=999)


def test_startup_roundtrip():
    # <BBBBIIHBb -> now <BBBBIIHBB (mic_flags unsigned)
    payload = struct.pack(_STARTUP_FMT, 1, 0, 1, 0, 0xDEAD, 48000, 512, 2, 3)
    _roundtrip(payload, FrameType.STARTUP, node_id=1, seq=0, t_us=0)


def test_decode_returns_version_1():
    payload = b"\x00" * 13
    raw = encode_frame(FrameType.ENV, 1, 0, 0, payload)
    frame = decode_frame(raw)
    assert frame["version"] == 1


# ---------------------------------------------------------------------------
# 3. Reject len > 4096
# ---------------------------------------------------------------------------

def test_encode_rejects_oversized_payload():
    with pytest.raises(ValueError, match="payload.*4096"):
        encode_frame(FrameType.AUDIO, 1, 0, 0, bytes(4097))


def test_decode_rejects_oversized_len_field():
    """Craft a frame whose header len field claims > 4096; decode must reject it."""
    # Build a valid-looking header with len=4097 but tiny payload
    version = 1
    frame_type = FrameType.AUDIO.value
    node_id = 1
    seq = 0
    t_us = 0
    bad_len = 4097
    hdr = SYNC + struct.pack(HDR_FMT, version, frame_type, node_id, seq, t_us, bad_len)
    # CRC over version..end-of-payload (no actual payload bytes here)
    crc_input = hdr[2:]  # skip sync
    crc = struct.pack("<I", zlib.crc32(crc_input) & 0xFFFFFFFF)
    with pytest.raises(ValueError, match="4096"):
        decode_frame(hdr + crc)


# ---------------------------------------------------------------------------
# 4. CRC corruption is detected
# ---------------------------------------------------------------------------

def test_bad_crc_raises():
    payload = struct.pack("<hHIIB", 2500, 5000, 0, 0, 0)
    raw = bytearray(encode_frame(FrameType.ENV, 1, 0, 0, payload))
    raw[-1] ^= 0xFF  # corrupt last CRC byte
    with pytest.raises(ValueError, match="CRC"):
        decode_frame(bytes(raw))


# ---------------------------------------------------------------------------
# 5. Stream parser resyncs after garbage bytes
# ---------------------------------------------------------------------------

def test_stream_resync_after_garbage():
    payload = struct.pack("<hHIIB", 2500, 5000, 1, 0, 0)
    good_frame = encode_frame(FrameType.ENV, 1, 0, 0, payload)
    garbage = b"\xDE\xAD\xBE\xEF\xBA\x00\xFF" * 3  # contains partial sync bytes
    stream = garbage + good_frame
    frames = list(parse_stream(stream))
    assert len(frames) == 1
    assert frames[0]["type"] == FrameType.ENV


def test_stream_multiple_frames():
    payload = struct.pack("<hHIIB", 1000, 4000, 0, 0, 0)
    f1 = encode_frame(FrameType.ENV, 1, 0, 0, payload)
    f2 = encode_frame(FrameType.ENV, 1, 1, 0, payload)
    frames = list(parse_stream(f1 + f2))
    assert len(frames) == 2
    assert frames[0]["seq"] == 0
    assert frames[1]["seq"] == 1


def test_stream_skips_bad_crc_frame_and_finds_next():
    payload = struct.pack("<hHIIB", 1000, 4000, 0, 0, 0)
    good = encode_frame(FrameType.ENV, 1, 5, 0, payload)
    bad = bytearray(encode_frame(FrameType.ENV, 1, 4, 0, payload))
    bad[-1] ^= 0xFF
    stream = bytes(bad) + good
    frames = list(parse_stream(stream))
    # bad frame is skipped; the good frame must still be found
    assert len(frames) == 1
    assert frames[0]["seq"] == 5


def test_stream_empty():
    assert list(parse_stream(b"")) == []


def test_stream_truncated_returns_nothing():
    payload = struct.pack("<hHIIB", 1000, 4000, 0, 0, 0)
    f = encode_frame(FrameType.ENV, 1, 0, 0, payload)
    # feed only half the frame
    assert list(parse_stream(f[: len(f) // 2])) == []


# ---------------------------------------------------------------------------
# 6. Typed payload helpers
# ---------------------------------------------------------------------------

def test_audio_payload_struct():
    samples = b"\x01\x02" * 1024  # 2048 bytes
    ap = AudioPayload(raw=samples)
    assert len(ap.raw) == 2048
    pairs = ap.stereo_pairs()  # list of (left_i16, right_i16)
    assert len(pairs) == 512
    assert pairs[0] == (0x0201, 0x0201)


def test_env_payload_struct():
    raw = struct.pack("<hHIIB", -100, 5500, 2, 3, 1)
    ep = EnvPayload.from_bytes(raw)
    assert ep.temp_c_x100 == -100
    assert ep.rh_x100 == 5500
    assert ep.overrun_count == 2
    assert ep.usb_drop_count == 3
    assert ep.bme_status == 1


def test_startup_payload_struct():
    raw = struct.pack(_STARTUP_FMT, 2, 0, 2, 0, 0xABCD, 48000, 512, 2, 3)
    sp = StartupPayload.from_bytes(raw)
    assert sp.board_type == 2
    assert sp.sample_rate_hz == 48000
    assert sp.samples_per_ch == 512
    assert sp.channels == 2


# ---------------------------------------------------------------------------
# 7. mic_flags=0x80 round-trips as unsigned
# ---------------------------------------------------------------------------

def test_startup_mic_flags_high_bit():
    """mic_flags=0x80 must survive round-trip (unsigned byte)."""
    raw = struct.pack(_STARTUP_FMT, 1, 0, 1, 0, 0, 48000, 512, 2, 0x80)
    sp = StartupPayload.from_bytes(raw)
    assert sp.mic_flags == 0x80


# ---------------------------------------------------------------------------
# 8. decode_frame rejects version != 1
# ---------------------------------------------------------------------------

def test_decode_rejects_bad_version():
    """A frame with version=2 but valid CRC must raise ValueError."""
    payload = b"\x00" * 13
    raw = bytearray(encode_frame(FrameType.ENV, 1, 0, 0, payload))
    # version byte is at index 2 (after 2-byte SYNC)
    raw[2] = 2
    # Recompute CRC over header_body (post-sync) + payload so CRC is valid
    header_body = bytes(raw[2:HDR_SIZE])
    crc_input = header_body + payload
    new_crc = struct.pack("<I", zlib.crc32(crc_input) & 0xFFFFFFFF)
    raw[HDR_SIZE + len(payload):] = new_crc
    with pytest.raises(ValueError, match="version"):
        decode_frame(bytes(raw))


# ---------------------------------------------------------------------------
# 9. parse_stream skips a version-2 frame (advances one byte) and finds next
# ---------------------------------------------------------------------------

def test_stream_skips_bad_version_frame():
    """parse_stream must skip a version!=1 frame and yield the following good frame."""
    payload = b"\x00" * 13
    good = encode_frame(FrameType.ENV, 1, 7, 0, payload)
    bad = bytearray(encode_frame(FrameType.ENV, 1, 0, 0, payload))
    bad[2] = 2  # set version=2
    # recompute CRC so CRC is valid (parse_stream must still reject it via decode_frame)
    header_body = bytes(bad[2:HDR_SIZE])
    crc_input = header_body + payload
    new_crc = struct.pack("<I", zlib.crc32(crc_input) & 0xFFFFFFFF)
    bad[HDR_SIZE + len(payload):] = new_crc
    stream = bytes(bad) + good
    frames = list(parse_stream(stream))
    assert len(frames) == 1
    assert frames[0]["seq"] == 7


# ---------------------------------------------------------------------------
# 10. AudioPayload.to_numpy()
# ---------------------------------------------------------------------------

def test_audio_to_numpy_shape():
    """to_numpy() must return shape (512, 2) for a standard 2048-byte payload."""
    raw = bytes(range(256)) * 8  # 2048 bytes
    ap = AudioPayload(raw=raw)
    arr = ap.to_numpy()
    assert arr.shape == (512, 2)
    assert arr.dtype == np.dtype("<i2")


def test_audio_to_numpy_values_equal_stereo_pairs():
    """to_numpy()[:, 0] and [:, 1] must match stereo_pairs() left/right."""
    raw = bytes(range(256)) * 8
    ap = AudioPayload(raw=raw)
    arr = ap.to_numpy()
    pairs = ap.stereo_pairs()
    for i, (left, right) in enumerate(pairs):
        assert arr[i, 0] == left
        assert arr[i, 1] == right
