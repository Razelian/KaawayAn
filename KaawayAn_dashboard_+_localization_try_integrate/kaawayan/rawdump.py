"""
kaawayan.rawdump — Dump raw firmware frames to a TYPE:hex golden file.

API
---
split_raw_frames(data: bytes) -> list[tuple[dict, bytes]]
    Scan *data* for valid frames. Return (frame_dict, raw_bytes) pairs.
    raw_bytes are the exact bytes from the stream (never re-encoded).
    Garbage and CRC-corrupt frames between good frames are skipped
    (advance ONE byte on failure, same policy as FrameAssembler).

select_golden(frames, n_audio=8) -> list[tuple[str, bytes]]
    From the decoded list pick:
      - first STARTUP frame
      - first ENV frame
      - the first n_audio AUDIO frames with consecutive seq numbers
        (if that run is shorter, use the longest consecutive run and
        print a notice)
    Output order: STARTUP, ENV, AUDIO...
    Missing STARTUP or ENV: omitted silently.

CLI
---
python -m kaawayan.rawdump \\
    --serial <usb_serial> --seconds 3 \\
    --out laptop/tests/golden/frames_fw.hex \\
    [--n-audio 8] [--force]

Reads raw bytes with SerialSource, splits, selects, writes one
"TYPE:hex" line per frame (same format as frames.hex), refuses to
overwrite unless --force, prints what it wrote.
"""

from __future__ import annotations

import struct
import sys
from typing import TYPE_CHECKING

from kaawayan.protocol import (
    SYNC,
    HDR_FMT,
    HDR_SIZE,
    CRC_SIZE,
    MAX_PAYLOAD,
    FrameType,
    decode_frame,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# split_raw_frames
# ---------------------------------------------------------------------------

def split_raw_frames(data: bytes) -> list[tuple[dict, bytes]]:
    """Scan *data* for valid frames; return (frame_dict, raw_bytes) pairs.

    raw_bytes are the exact bytes from *data* (sliced, never re-encoded).
    Garbage and corrupt frames are skipped: advance ONE byte on failure.
    """
    buf = memoryview(data)
    pos = 0
    results: list[tuple[dict, bytes]] = []
    n = len(data)

    while pos < n:
        # Find next SYNC from pos
        sync_pos = data.find(SYNC, pos)
        if sync_pos == -1:
            break
        pos = sync_pos

        # Need at least HDR_SIZE + CRC_SIZE to read header
        if pos + HDR_SIZE + CRC_SIZE > n:
            break

        # Parse len field
        header_body = bytes(buf[pos + 2 : pos + HDR_SIZE])
        try:
            _, _, _, _, _, plen = struct.unpack("<" + HDR_FMT, header_body)
        except struct.error:
            pos += 1
            continue

        if plen > MAX_PAYLOAD:
            pos += 1
            continue

        frame_end = pos + HDR_SIZE + plen + CRC_SIZE
        if frame_end > n:
            break  # truncated at end of data

        raw = bytes(buf[pos:frame_end])
        try:
            frame = decode_frame(raw)
            results.append((frame, raw))
            pos = frame_end
        except (ValueError, struct.error):
            pos += 1

    return results


# ---------------------------------------------------------------------------
# select_golden
# ---------------------------------------------------------------------------

_TYPE_NAME = {
    FrameType.STARTUP: "STARTUP",
    FrameType.AUDIO:   "AUDIO",
    FrameType.ENV:     "ENV",
}


def select_golden(
    frames: list[tuple[dict, bytes]],
    n_audio: int = 8,
) -> list[tuple[str, bytes]]:
    """Select STARTUP, ENV, and up to n_audio consecutive AUDIO frames.

    Returns list of (type_name, raw_bytes) in order: STARTUP, ENV, AUDIO...
    Missing STARTUP or ENV are omitted silently.
    If the first consecutive AUDIO run is shorter than n_audio, the longest
    run is used and a notice is printed to stderr.
    """
    startup: tuple[str, bytes] | None = None
    env: tuple[str, bytes] | None = None
    audio_frames: list[tuple[dict, bytes]] = []

    for frame, raw in frames:
        ftype = frame["type"]
        if ftype == FrameType.STARTUP and startup is None:
            startup = ("STARTUP", raw)
        elif ftype == FrameType.ENV and env is None:
            env = ("ENV", raw)
        elif ftype == FrameType.AUDIO:
            audio_frames.append((frame, raw))

    # Find longest consecutive-seq run
    selected_audio: list[bytes] = []
    if audio_frames:
        # Try first-run approach
        first_run: list[bytes] = [audio_frames[0][1]]
        for i in range(1, len(audio_frames)):
            if audio_frames[i][0]["seq"] == audio_frames[i - 1][0]["seq"] + 1:
                first_run.append(audio_frames[i][1])
            else:
                break

        if len(first_run) >= n_audio:
            selected_audio = first_run[:n_audio]
        else:
            # Find the longest run
            best_run: list[bytes] = list(first_run)
            cur_run: list[bytes] = [audio_frames[0][1]]
            for i in range(1, len(audio_frames)):
                if audio_frames[i][0]["seq"] == audio_frames[i - 1][0]["seq"] + 1:
                    cur_run.append(audio_frames[i][1])
                else:
                    if len(cur_run) > len(best_run):
                        best_run = cur_run
                    cur_run = [audio_frames[i][1]]
            if len(cur_run) > len(best_run):
                best_run = cur_run

            if len(best_run) < len(first_run):
                best_run = first_run  # shouldn't happen; safety

            print(
                f"rawdump: notice: first consecutive AUDIO run has {len(first_run)} "
                f"frames (< {n_audio}); using longest run ({len(best_run)} frames).",
                file=sys.stderr,
            )
            selected_audio = best_run[:n_audio]

    result: list[tuple[str, bytes]] = []
    if startup is not None:
        result.append(startup)
    if env is not None:
        result.append(env)
    for raw in selected_audio:
        result.append(("AUDIO", raw))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_hex_file(path: "pathlib.Path", frames: list[tuple[str, bytes]]) -> None:
    import pathlib  # noqa: F811
    lines = [f"{name}:{raw.hex()}" for name, raw in frames]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import pathlib
    import time

    parser = argparse.ArgumentParser(
        description="Dump raw firmware frames to a TYPE:hex golden file."
    )
    parser.add_argument("--serial", required=True,
                        help="USB serial number or port name (e.g. COM3).")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="Seconds to capture (default: 3).")
    parser.add_argument("--out", required=True,
                        help="Output file path (e.g. laptop/tests/golden/frames_fw.hex).")
    parser.add_argument("--n-audio", type=int, default=8,
                        help="Number of consecutive AUDIO frames to select (default: 8).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file.")
    args = parser.parse_args(argv)

    out_path = pathlib.Path(args.out)
    if out_path.exists() and not args.force:
        print(
            f"rawdump: {out_path} already exists; use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    from kaawayan.record import SerialSource

    print(f"rawdump: opening serial {args.serial!r} ...", file=sys.stderr)
    src = SerialSource(args.serial)

    raw_data = bytearray()
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline:
            chunk = src.read()
            if chunk:
                raw_data.extend(chunk)
    finally:
        src.close()

    print(f"rawdump: read {len(raw_data)} bytes.", file=sys.stderr)

    frames = split_raw_frames(bytes(raw_data))
    print(f"rawdump: found {len(frames)} valid frames.", file=sys.stderr)

    selected = select_golden(frames, n_audio=args.n_audio)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_hex_file(out_path, selected)

    for name, raw in selected:
        print(f"  wrote {name} ({len(raw)} bytes)")

    print(f"rawdump: wrote {len(selected)} frames to {out_path}.", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
