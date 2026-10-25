#!/usr/bin/env python3
"""
Generate tests/golden/frames.hex – one frame of each type.

Run from the repo root:
    python laptop/tests/golden/gen_frames.py

The hex file is used by both the Python tests and the C firmware
to verify identical parsing.  Each line is a complete frame in
lowercase hex, prefixed with the frame type name:
    STARTUP:<hex>
    AUDIO:<hex>
    ENV:<hex>
"""
import struct
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # laptop/

from kaawayan.protocol import encode_frame, FrameType

OUT = Path(__file__).parent / "frames.hex"


def make_startup() -> bytes:
    # board_type=1(XIAO), fw 0.1.0, git_short=0xCAFEBABE,
    # sample_rate=48000, samples_per_ch=512, channels=2, mic_flags=3 (unsigned B)
    payload = struct.pack("<BBBBIIHBB", 1, 0, 1, 0, 0xCAFEBABE, 48000, 512, 2, 3)
    return encode_frame(FrameType.STARTUP, node_id=1, seq=0, t_us=0, payload=payload)


def make_audio() -> bytes:
    # 512 stereo pairs, interleaved int16: left=0x0100, right=0x0200 repeated
    pair = struct.pack("<hh", 0x0100, 0x0200)
    payload = pair * 512   # 2048 bytes
    return encode_frame(FrameType.AUDIO, node_id=1, seq=0, t_us=10_670, payload=payload)


def make_env() -> bytes:
    # temp=25.00°C (x100=2500), rh=60.00% (x100=6000), overrun=0, drop=0, bme_status=1
    payload = struct.pack("<hHIIB", 2500, 6000, 0, 0, 1)
    return encode_frame(FrameType.ENV, node_id=1, seq=0, t_us=1_000_000, payload=payload)


def main() -> None:
    frames = [
        ("STARTUP", make_startup()),
        ("AUDIO",   make_audio()),
        ("ENV",     make_env()),
    ]
    lines = []
    for name, frame in frames:
        lines.append(f"{name}:{frame.hex()}")
    OUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"Wrote {OUT}")
    for name, frame in frames:
        print(f"  {name}: {len(frame)} bytes")


if __name__ == "__main__":
    main()
