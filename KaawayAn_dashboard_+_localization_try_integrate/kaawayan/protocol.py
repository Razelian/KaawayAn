"""
kaawayan.protocol
=================
Encode / decode KaawayAn wire-protocol frames (version 1, little-endian).

Frame layout (CONTEXT.md §4):
  SYNC (2B) | header (18B) | payload (len B) | CRC-32 (4B)

Header struct format: '<2sBBHIQH'  (packed as HDR_FMT after SYNC)
  sync      2   0xBA 0xAB
  version   1   must be 1
  type      1   1=STARTUP, 2=AUDIO, 3=ENV
  node_id   2   uint16
  seq       4   uint32, per-node per-type counter
  t_us      8   uint64, node microsecond timestamp (diagnostic)
  len       2   payload byte count

CRC-32 (IEEE/zlib) over version..end-of-payload (sync excluded).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Generator

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNC: bytes = b"\xBA\xAB"
VERSION: int = 1
MAX_PAYLOAD: int = 4096

# Header = sync(2) + version(1) + type(1) + node_id(2) + seq(4) + t_us(8) + len(2)
# The struct format covers everything AFTER the 2-byte sync.
HDR_FMT = "BBHIQH"            # version, type, node_id, seq, t_us, len  – all little-endian
HDR_SIZE: int = 2 + struct.calcsize("<" + HDR_FMT)   # 2 sync + 18 = 20
CRC_SIZE: int = 4


# ---------------------------------------------------------------------------
# Frame type enum
# ---------------------------------------------------------------------------

class FrameType(IntEnum):
    STARTUP = 1
    AUDIO   = 2
    ENV     = 3


# ---------------------------------------------------------------------------
# CRC helper
# ---------------------------------------------------------------------------

def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode_frame(
    frame_type: FrameType,
    node_id: int,
    seq: int,
    t_us: int,
    payload: bytes,
) -> bytes:
    """Return a complete wire frame including SYNC and CRC-32."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload length {len(payload)} exceeds 4096")
    header_body = struct.pack(
        "<" + HDR_FMT,
        VERSION,
        int(frame_type),
        node_id,
        seq,
        t_us,
        len(payload),
    )
    crc_input = header_body + payload          # version..end-of-payload
    crc = struct.pack("<I", _crc32(crc_input))
    return SYNC + header_body + payload + crc


# ---------------------------------------------------------------------------
# Decode (single frame)
# ---------------------------------------------------------------------------

def decode_frame(data: bytes) -> dict:
    """
    Decode one complete frame from *data*.

    Returns a dict with keys:
        version, type (FrameType), node_id, seq, t_us, payload (bytes)

    Raises ValueError on CRC mismatch or len > 4096.
    Raises struct.error / ValueError if data is too short.
    """
    if len(data) < HDR_SIZE + CRC_SIZE:
        raise ValueError("frame too short")
    if data[:2] != SYNC:
        raise ValueError("missing SYNC")
    header_body = data[2 : HDR_SIZE]
    version, ftype, node_id, seq, t_us, plen = struct.unpack(
        "<" + HDR_FMT, header_body
    )
    if plen > MAX_PAYLOAD:
        raise ValueError(f"declared payload length {plen} exceeds 4096")
    expected_total = HDR_SIZE + plen + CRC_SIZE
    if len(data) < expected_total:
        raise ValueError("frame truncated")
    payload = data[HDR_SIZE : HDR_SIZE + plen]
    crc_stored = struct.unpack("<I", data[HDR_SIZE + plen : HDR_SIZE + plen + CRC_SIZE])[0]
    crc_input = header_body + payload
    crc_calc = _crc32(crc_input)
    if crc_calc != crc_stored:
        raise ValueError(f"CRC mismatch: got {crc_calc:#010x}, expected {crc_stored:#010x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}; expected {VERSION}")
    return {
        "version": version,
        "type": FrameType(ftype),
        "node_id": node_id,
        "seq": seq,
        "t_us": t_us,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Stream parser (advances one byte on failure, per CONTEXT.md §4)
# ---------------------------------------------------------------------------

def parse_stream(data: bytes) -> Generator[dict, None, None]:
    """
    Yield decoded frames from a raw byte stream.

    On CRC failure or invalid len field, advance ONE byte and rescan.
    Incomplete frames at the end of *data* are silently discarded
    (caller must buffer and retry with more data).
    """
    buf = data
    i = 0
    while i < len(buf):
        # Search for SYNC
        sync_pos = buf.find(SYNC, i)
        if sync_pos == -1:
            break
        i = sync_pos
        # Need at least header + CRC to proceed
        if i + HDR_SIZE + CRC_SIZE > len(buf):
            break
        # Peek at the len field
        header_body = buf[i + 2 : i + HDR_SIZE]
        try:
            _, _, _, _, _, plen = struct.unpack("<" + HDR_FMT, header_body)
        except struct.error:
            i += 1
            continue
        if plen > MAX_PAYLOAD:
            i += 1
            continue
        frame_end = i + HDR_SIZE + plen + CRC_SIZE
        if frame_end > len(buf):
            # Truncated – stop; caller should buffer
            break
        raw_frame = buf[i : frame_end]
        try:
            frame = decode_frame(raw_frame)
            yield frame
            i = frame_end
        except (ValueError, struct.error):
            # Advance one byte and rescan
            i += 1


# ---------------------------------------------------------------------------
# Typed payload helpers
# ---------------------------------------------------------------------------

_AUDIO_PAIR_FMT = "<hh"   # left int16, right int16


@dataclass
class AudioPayload:
    """Wrapper around a raw AUDIO payload (512 stereo int16 pairs = 2048 bytes)."""
    raw: bytes

    def stereo_pairs(self) -> list[tuple[int, int]]:
        """Return list of (left_i16, right_i16) tuples."""
        n = len(self.raw) // 4
        return [
            struct.unpack_from("<hh", self.raw, offset=i * 4)
            for i in range(n)
        ]

    def to_numpy(self) -> np.ndarray:
        """Return int16 array of shape (n, 2); column 0 = left, column 1 = right."""
        n = len(self.raw) // 4
        return np.frombuffer(self.raw, dtype="<i2").reshape(n, 2)


_ENV_FMT = "<hHIIB"   # temp_c_x100, rh_x100, overrun_count, usb_drop_count, bme_status


@dataclass
class EnvPayload:
    temp_c_x100: int
    rh_x100: int
    overrun_count: int
    usb_drop_count: int
    bme_status: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "EnvPayload":
        t, rh, ov, drop, bme = struct.unpack(_ENV_FMT, data)
        return cls(
            temp_c_x100=t,
            rh_x100=rh,
            overrun_count=ov,
            usb_drop_count=drop,
            bme_status=bme,
        )


_STARTUP_FMT = "<BBBBIIHBB"
# board_type, fw_major, fw_minor, fw_patch, git_short, sample_rate_hz,
# samples_per_ch, channels, mic_flags  (mic_flags is unsigned)


@dataclass
class StartupPayload:
    board_type: int
    fw_major: int
    fw_minor: int
    fw_patch: int
    git_short: int
    sample_rate_hz: int
    samples_per_ch: int
    channels: int
    mic_flags: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "StartupPayload":
        (board, fmaj, fmin, fpatch, git, sr, spc, ch, mf) = struct.unpack(_STARTUP_FMT, data)
        return cls(
            board_type=board,
            fw_major=fmaj,
            fw_minor=fmin,
            fw_patch=fpatch,
            git_short=git,
            sample_rate_hz=sr,
            samples_per_ch=spc,
            channels=ch,
            mic_flags=mf,
        )
