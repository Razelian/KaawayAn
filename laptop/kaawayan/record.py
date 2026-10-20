"""
kaawayan.record — Ingest raw USB CDC byte stream into WAV + logs.

Classes
-------
FrameAssembler
    Stateful byte-stream reassembler. Call feed(bytes, t_laptop_s) to get
    a list of (frame_dict, t_laptop_s) tuples. Buffers across calls.
    Advances ONE byte on failure; counts crc_failures and resync_bytes_skipped.
    All frames completed in one feed() call get that call's t_laptop_s
    (approximation documented here: the true arrival time of each byte is
    not tracked; only the call time is available).

NodeRecorder
    Per-node sink that accepts feed_frame(frame_dict, t_laptop_s).
    Writes:
      node{id}.wav  — stereo 48 kHz int16 WAV (crash-safe: header patched every
                      100 AUDIO frames and at close)
      node{id}_env.csv  — ENV frame rows (appended/flushed immediately)
      gaps.log      — shared via TrialRecorder; NodeRecorder logs to a file
                      directly so bytes survive a crash
      meta.json     — written only by TrialRecorder.close()
    A seq < last_seq means node reboot: RESET logged, resets counter
    incremented, flagged, stream continues (gap unknown, no zero-fill).
    A seq == last_seq is a duplicate: skipped, DUP logged.
    NodeRecorder.close() is still usable stand-alone; it writes its own
    minimal meta.json in that case.

TrialRecorder
    Owns one NodeRecorder per node_id.
    feed_frame(frame, t) routes by frame["node_id"].
    close(trial_id, kind, cfg_path=None, extra_meta=None, assemblers=None)
    writes ONE shared meta.json (extra_meta merged verbatim, core keys
    protected) and one shared gaps.log.

SerialSource
    Wraps a pyserial port; matches list_ports serial_number to config
    usb_serial; read() returns bytes.
    UNTESTED ON HARDWARE — use a fake implementing the same interface.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import pathlib
import struct
import subprocess
import wave
from typing import Any

import numpy as np

from kaawayan.protocol import (
    SYNC,
    HDR_FMT,
    HDR_SIZE,
    CRC_SIZE,
    MAX_PAYLOAD,
    FrameType,
    decode_frame,
    EnvPayload,
    StartupPayload,
)

log = logging.getLogger(__name__)

_FS = 48_000
_SAMPLES_PER_AUDIO = 512
_WAV_PATCH_EVERY = 100  # patch WAV header every N AUDIO frames

# ---------------------------------------------------------------------------
# Git / config helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def _config_hash(cfg_path: pathlib.Path | None = None) -> str:
    """SHA-256 of site.toml, or 'n/a' if not found."""
    if cfg_path is None:
        here = pathlib.Path(__file__).resolve().parent
        for candidate in [here / ".." / ".." / "config" / "site.toml",
                          here / ".." / "config" / "site.toml"]:
            candidate = candidate.resolve()
            if candidate.exists():
                cfg_path = candidate
                break
    if cfg_path is None or not pathlib.Path(cfg_path).exists():
        return "n/a"
    try:
        raw = pathlib.Path(cfg_path).read_bytes()
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Minimal WAV writer (crash-safe: patches header on demand)
# ---------------------------------------------------------------------------

_WAV_HEADER_SIZE = 44  # RIFF/fmt/data header, no extras


def _wav_header(n_frames: int, n_channels: int = 2, sample_width: int = 2,
                framerate: int = _FS) -> bytes:
    """Build a 44-byte PCM WAV header."""
    data_size = n_frames * n_channels * sample_width
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16, 1, n_channels, framerate,
        framerate * n_channels * sample_width,
        n_channels * sample_width, sample_width * 8,
        b"data", data_size,
    )


class _CrashSafeWav:
    """Minimal WAV writer that patches its own header periodically."""

    def __init__(self, path: pathlib.Path, n_channels: int = 2,
                 sample_width: int = 2, framerate: int = _FS) -> None:
        self._path = path
        self._n_channels = n_channels
        self._sample_width = sample_width
        self._framerate = framerate
        self._n_frames: int = 0
        self._audio_frame_count: int = 0  # AUDIO frames since last patch

        # Open/create file and write initial header
        self._fh = open(path, "wb")
        self._fh.write(_wav_header(0, n_channels, sample_width, framerate))
        self._fh.flush()

    def write_frames(self, data: bytes) -> None:
        self._fh.write(data)
        samples = len(data) // (self._n_channels * self._sample_width)
        self._n_frames += samples
        self._audio_frame_count += 1
        if self._audio_frame_count >= _WAV_PATCH_EVERY:
            self._patch_header()
            self._audio_frame_count = 0

    def _patch_header(self) -> None:
        pos = self._fh.tell()
        self._fh.seek(0)
        self._fh.write(_wav_header(self._n_frames, self._n_channels,
                                   self._sample_width, self._framerate))
        self._fh.seek(pos)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._patch_header()
            self._fh.close()

    @property
    def n_frames(self) -> int:
        return self._n_frames


# ---------------------------------------------------------------------------
# FrameAssembler
# ---------------------------------------------------------------------------

class FrameAssembler:
    """Stateful byte-stream reassembler.

    feed(data, t_laptop_s) -> list[(frame_dict, t_laptop_s)]

    All frames completed in a single call get that call's t_laptop_s
    (documented approximation: per-byte arrival time is not tracked).
    On CRC or parse failure: advance ONE byte, increment counters.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self.crc_failures: int = 0
        self.resync_bytes_skipped: int = 0

    def feed(self, data: bytes, t_laptop_s: float) -> list[tuple[dict, float]]:
        self._buf.extend(data)
        results: list[tuple[dict, float]] = []

        while True:
            # Find next SYNC
            sync_pos = self._buf.find(SYNC)
            if sync_pos == -1:
                # No SYNC found. Keep the last byte in case it is the first
                # byte of a SYNC split across two feed() calls (0xBA alone).
                keep = 1 if len(self._buf) >= 1 and self._buf[-1] == SYNC[0] else 0
                skipped = len(self._buf) - keep
                if skipped > 0:
                    self.resync_bytes_skipped += skipped
                if keep:
                    self._buf = self._buf[-1:]
                else:
                    self._buf.clear()
                break

            if sync_pos > 0:
                # Bytes before SYNC are garbage
                self.resync_bytes_skipped += sync_pos
                del self._buf[:sync_pos]

            # Need at least HDR_SIZE + CRC_SIZE bytes to parse header
            if len(self._buf) < HDR_SIZE + CRC_SIZE:
                break  # wait for more data

            # Parse len field from header
            header_body = bytes(self._buf[2:HDR_SIZE])
            try:
                _, _, _, _, _, plen = struct.unpack("<" + HDR_FMT, header_body)
            except struct.error:
                self.resync_bytes_skipped += 1
                del self._buf[:1]
                continue

            if plen > MAX_PAYLOAD:
                self.resync_bytes_skipped += 1
                del self._buf[:1]
                continue

            frame_end = HDR_SIZE + plen + CRC_SIZE
            if len(self._buf) < frame_end:
                break  # wait for rest of frame

            raw_frame = bytes(self._buf[:frame_end])
            try:
                frame = decode_frame(raw_frame)
                results.append((frame, t_laptop_s))
                del self._buf[:frame_end]
            except (ValueError, struct.error) as exc:
                # CRC mismatch or version error
                if "CRC" in str(exc):
                    self.crc_failures += 1
                self.resync_bytes_skipped += 1
                del self._buf[:1]

        return results


# ---------------------------------------------------------------------------
# NodeRecorder
# ---------------------------------------------------------------------------

class NodeRecorder:
    """Per-node sink that writes WAV + env.csv + (optionally shared) gaps.log.

    Parameters
    ----------
    node_id : int
        Expected node_id for frames fed to this recorder.
    out_dir : pathlib.Path | str
        Directory where output files are written (created if absent).
    gaps_log_path : pathlib.Path | None
        If provided, gap/reset/dup lines are appended to this path directly
        (used by TrialRecorder for a shared gaps.log). Otherwise a per-node
        gaps.log in out_dir is used.
    """

    def __init__(self, node_id: int, out_dir: pathlib.Path | str,
                 gaps_log_path: pathlib.Path | None = None) -> None:
        self.node_id = node_id
        self._out = pathlib.Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)

        # Crash-safe WAV writer (opened on first AUDIO frame)
        self._wav: _CrashSafeWav | None = None
        self._wav_path = self._out / f"node{node_id}.wav"

        # Gap/reset tracking
        self._last_audio_seq: int | None = None
        self._total_gap_samples: int = 0
        self._resets: int = 0
        self._flagged: bool = False

        # Seq tracking per type (for all types)
        self._last_seq: dict[FrameType, int] = {}

        # STARTUP info
        self.fw_version: str | None = None
        self.mic_flags: int | None = None

        # t_first_audio_laptop_s
        self.t_first_audio_laptop_s: float | None = None

        # n_audio_frames
        self._n_audio_frames: int = 0

        # Gaps log (shared path or local)
        if gaps_log_path is not None:
            self._gaps_log_path = gaps_log_path
        else:
            self._gaps_log_path = self._out / "gaps.log"

        # ENV CSV — append+flush immediately (crash-safe)
        self._env_path = self._out / f"node{node_id}_env.csv"
        self._env_fh: Any = None
        self._env_writer: Any = None
        self._env_fieldnames = ["t_laptop_s", "temp_c", "rh_pct",
                                 "overrun_count", "usb_drop_count", "bme_status"]

        # Counters (set from assembler at close)
        self.crc_failures: int = 0
        self.resync_bytes_skipped: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_wav(self) -> None:
        if self._wav is not None:
            return
        self._wav = _CrashSafeWav(self._wav_path)

    def _write_zero_gap(self, n_samples: int) -> None:
        """Write n_samples of zero stereo pairs to WAV."""
        self._open_wav()
        zeros = np.zeros((n_samples, 2), dtype=np.int16)
        self._wav.write_frames(zeros.tobytes())  # type: ignore[union-attr]

    def _write_audio_bytes(self, payload: bytes) -> None:
        self._open_wav()
        self._wav.write_frames(payload)  # type: ignore[union-attr]

    def _log_gap(self, line: str) -> None:
        """Append a single line to the gaps log immediately."""
        with open(self._gaps_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _open_env(self) -> None:
        if self._env_fh is not None:
            return
        self._env_fh = open(self._env_path, "w", newline="", encoding="utf-8")
        self._env_writer = csv.DictWriter(self._env_fh, fieldnames=self._env_fieldnames)
        self._env_writer.writeheader()
        self._env_fh.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_frame(self, frame: dict, t_laptop_s: float) -> None:
        """Dispatch one decoded frame to the appropriate handler."""
        nid = frame.get("node_id")
        if nid != self.node_id:
            log.warning(
                "NodeRecorder(%d): received frame from node_id=%s; skipping",
                self.node_id, nid,
            )
            return

        ftype = frame["type"]
        seq = frame["seq"]

        if ftype == FrameType.AUDIO:
            self._handle_audio(frame, t_laptop_s, seq)
        elif ftype == FrameType.ENV:
            self._handle_env(frame, t_laptop_s)
        elif ftype == FrameType.STARTUP:
            self._handle_startup(frame)

    def _handle_audio(self, frame: dict, t_laptop_s: float, seq: int) -> None:
        if self._last_audio_seq is None:
            # First frame — defines baseline; no gap before it
            self._last_audio_seq = seq
            self.t_first_audio_laptop_s = t_laptop_s
        else:
            if seq < self._last_audio_seq:
                # Node rebooted
                self._resets += 1
                self._flagged = True
                self._log_gap(
                    f"node_id={self.node_id} type=AUDIO seq={self._last_audio_seq}->0 "
                    f"RESET (gap unknown, no zero-fill)"
                )
                self._last_audio_seq = seq
            elif seq == self._last_audio_seq:
                # Duplicate frame — skip silently with log
                self._log_gap(
                    f"node_id={self.node_id} type=AUDIO seq={seq} DUP (skipped)"
                )
                return
            else:
                expected_seq = self._last_audio_seq + 1
                if seq > expected_seq:
                    gap_frames = seq - expected_seq
                    gap_samples = gap_frames * _SAMPLES_PER_AUDIO
                    duration_ms = gap_samples / _FS * 1000.0
                    self._write_zero_gap(gap_samples)
                    self._total_gap_samples += gap_samples
                    self._log_gap(
                        f"node_id={self.node_id} type=AUDIO "
                        f"seq={expected_seq}..{seq - 1} duration_ms={duration_ms:.2f}"
                    )
                    if (self._total_gap_samples / _FS * 1000.0) > 100.0:
                        self._flagged = True
                self._last_audio_seq = seq

        self._write_audio_bytes(frame["payload"])
        self._n_audio_frames += 1

    def _handle_env(self, frame: dict, t_laptop_s: float) -> None:
        try:
            ep = EnvPayload.from_bytes(frame["payload"])
        except Exception as exc:
            log.warning("NodeRecorder(%d): bad ENV payload: %s", self.node_id, exc)
            return
        self._open_env()
        row = {
            "t_laptop_s": t_laptop_s,
            "temp_c": ep.temp_c_x100 / 100.0,
            "rh_pct": ep.rh_x100 / 100.0,
            "overrun_count": ep.overrun_count,
            "usb_drop_count": ep.usb_drop_count,
            "bme_status": ep.bme_status,
        }
        self._env_writer.writerow(row)
        self._env_fh.flush()

    def _handle_startup(self, frame: dict) -> None:
        try:
            sp = StartupPayload.from_bytes(frame["payload"])
            self.fw_version = f"{sp.fw_major}.{sp.fw_minor}.{sp.fw_patch}"
            self.mic_flags = sp.mic_flags
        except Exception as exc:
            log.warning("NodeRecorder(%d): bad STARTUP payload: %s", self.node_id, exc)

    def close(self, trial_id: str, kind: str,
               cfg_path: pathlib.Path | None = None,
               assembler: "FrameAssembler | None" = None,
               extra_meta: dict | None = None) -> dict[str, Any]:
        """Finalise all files and return summary dict.

        When called stand-alone (not via TrialRecorder), also writes meta.json.
        """
        if assembler is not None:
            self.crc_failures = assembler.crc_failures
            self.resync_bytes_skipped = assembler.resync_bytes_skipped

        # Close WAV (or create empty WAV)
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        else:
            # Create empty WAV
            w = _CrashSafeWav(self._wav_path)
            w.close()

        # Close ENV CSV
        if self._env_fh is not None:
            self._env_fh.close()
            self._env_fh = None
        else:
            # Write an empty env CSV (header only)
            with open(self._env_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._env_fieldnames).writeheader()

        total_gap_ms = self._total_gap_samples / _FS * 1000.0
        flagged = self._flagged or (total_gap_ms > 100.0)

        summary = {
            "fw_version": self.fw_version,
            "mic_flags": self.mic_flags,
            "total_gap_ms": total_gap_ms,
            "flagged": flagged,
            "resets": self._resets,
            "crc_failures": self.crc_failures,
            "resync_bytes_skipped": self.resync_bytes_skipped,
            "t_first_audio_laptop_s": self.t_first_audio_laptop_s,
            "n_audio_frames": self._n_audio_frames,
        }

        # Stand-alone use: write a minimal meta.json
        meta_path = self._out / "meta.json"
        if not meta_path.exists():
            _core: dict[str, Any] = {
                "trial_id": trial_id,
                "kind": kind,
                "config_hash": _config_hash(cfg_path),
                "git_commit": _git_commit(),
                "flagged": flagged,
                "nodes": {str(self.node_id): summary},
            }
            if extra_meta:
                for k, v in extra_meta.items():
                    if k not in _core:
                        _core[k] = v
            meta_path.write_text(json.dumps(_core, indent=2, sort_keys=True))

        return {
            "total_gap_ms": total_gap_ms,
            "flagged": flagged,
            "crc_failures": self.crc_failures,
            "resync_bytes_skipped": self.resync_bytes_skipped,
        }

    def node_summary(self) -> dict[str, Any]:
        """Return the per-node summary dict (for TrialRecorder.close)."""
        total_gap_ms = self._total_gap_samples / _FS * 1000.0
        return {
            "fw_version": self.fw_version,
            "mic_flags": self.mic_flags,
            "total_gap_ms": total_gap_ms,
            "flagged": self._flagged or (total_gap_ms > 100.0),
            "resets": self._resets,
            "crc_failures": self.crc_failures,
            "resync_bytes_skipped": self.resync_bytes_skipped,
            "t_first_audio_laptop_s": self.t_first_audio_laptop_s,
            "n_audio_frames": self._n_audio_frames,
        }


# ---------------------------------------------------------------------------
# TrialRecorder
# ---------------------------------------------------------------------------

class TrialRecorder:
    """Owns one NodeRecorder per node_id; routes frames by node_id.

    Parameters
    ----------
    out_dir : pathlib.Path | str
        Trial output directory.
    node_ids : list[int]
        IDs of the nodes expected in this trial.
    """

    def __init__(self, out_dir: pathlib.Path | str, node_ids: list[int]) -> None:
        self._out = pathlib.Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._gaps_log_path = self._out / "gaps.log"
        self._recorders: dict[int, NodeRecorder] = {
            nid: NodeRecorder(nid, out_dir, gaps_log_path=self._gaps_log_path)
            for nid in node_ids
        }

    def feed_frame(self, frame: dict, t_laptop_s: float) -> None:
        """Route a decoded frame to the appropriate NodeRecorder."""
        nid = frame.get("node_id")
        rec = self._recorders.get(nid)
        if rec is None:
            log.warning("TrialRecorder: unknown node_id=%s; skipping", nid)
            return
        rec.feed_frame(frame, t_laptop_s)

    def close(
        self,
        trial_id: str,
        kind: str,
        cfg_path: pathlib.Path | None = None,
        extra_meta: dict | None = None,
        assemblers: dict[int, "FrameAssembler"] | None = None,
    ) -> dict[str, Any]:
        """Close all node recorders and write ONE shared meta.json.

        Parameters
        ----------
        extra_meta:
            Merged verbatim into meta.json; core keys (trial_id, kind,
            config_hash, git_commit, flagged, nodes) are never overridden.
        assemblers:
            Optional {node_id: FrameAssembler} to echo CRC/resync counts.
        """
        if assemblers is None:
            assemblers = {}

        node_summaries: dict[str, Any] = {}
        any_flagged = False

        for nid, rec in self._recorders.items():
            asm = assemblers.get(nid)
            if asm is not None:
                rec.crc_failures = asm.crc_failures
                rec.resync_bytes_skipped = asm.resync_bytes_skipped

            # Close WAV + env for each node
            if rec._wav is not None:
                rec._wav.close()
                rec._wav = None
            else:
                w = _CrashSafeWav(rec._wav_path)
                w.close()

            if rec._env_fh is not None:
                rec._env_fh.close()
                rec._env_fh = None
            else:
                # Create empty env CSV
                ep = self._out / f"node{nid}_env.csv"
                if not ep.exists():
                    with open(ep, "w", newline="", encoding="utf-8") as f:
                        csv.DictWriter(
                            f, fieldnames=rec._env_fieldnames
                        ).writeheader()

            summary = rec.node_summary()
            if summary["flagged"]:
                any_flagged = True
            node_summaries[str(nid)] = summary

        # Build meta.json — core keys cannot be overridden by extra_meta
        core: dict[str, Any] = {
            "trial_id": trial_id,
            "kind": kind,
            "config_hash": _config_hash(cfg_path),
            "git_commit": _git_commit(),
            "flagged": any_flagged,
            "nodes": node_summaries,
        }

        if extra_meta:
            for k, v in extra_meta.items():
                if k not in core:
                    core[k] = v

        meta_path = self._out / "meta.json"
        meta_path.write_text(json.dumps(core, indent=2, sort_keys=True))

        return core


# ---------------------------------------------------------------------------
# SerialSource  — UNTESTED ON HARDWARE
# ---------------------------------------------------------------------------

class SerialSource:
    """Wraps a pyserial port; matched by USB serial number.

    UNTESTED ON HARDWARE. Use a fake implementing the same interface:
        fake.read(n: int) -> bytes
        fake.is_open -> bool
        fake.close() -> None

    Parameters
    ----------
    port_or_serial : str
        Device path (e.g. "COM3") or USB serial number to match via
        serial.tools.list_ports.
    baudrate : int
        Serial baud rate (ignored for CDC ACM but kept for API symmetry).

    Notes
    -----
    The Serial object is created WITHOUT a port so the OS does not open
    the device immediately (which would assert DTR/RTS and can reset the
    ESP32-S3 into download mode).  DTR and RTS are forced False before
    open() is called.  After open, the receive buffer is enlarged to 1 MiB
    on Windows (set_buffer_size, ignored if unavailable).
    """

    def __init__(self, port_or_serial: str, baudrate: int = 115200) -> None:
        import serial  # type: ignore[import]
        import serial.tools.list_ports  # type: ignore[import]

        # Try to match by usb_serial first
        matched_port: str | None = None
        self.serial_number: str | None = None
        for info in serial.tools.list_ports.comports():
            if info.serial_number == port_or_serial:
                matched_port = info.device
                self.serial_number = info.serial_number
                break

        if matched_port is None:
            # Fall back to treating it as a direct port name
            matched_port = port_or_serial

        # Create UNOPENED Serial, set port/baud/timeout, suppress DTR+RTS,
        # then open.  This avoids the reset-on-connect problem on ESP32-S3.
        ser = serial.Serial()
        ser.port = matched_port
        ser.baudrate = baudrate
        ser.timeout = 0.05
        ser.dtr = False
        ser.rts = False
        ser.open()

        # Enlarge the OS receive buffer (Windows only; ignore errors).
        if hasattr(ser, "set_buffer_size"):
            try:
                ser.set_buffer_size(rx_size=1 << 20)
            except Exception:
                pass

        self._serial = ser

    def read(self, size: int = 4096) -> bytes:
        """Read up to size bytes; uses in_waiting for efficiency."""
        waiting = self._serial.in_waiting
        n = max(1, min(waiting, 65536))
        return self._serial.read(n)

    @property
    def is_open(self) -> bool:
        return self._serial.is_open

    def close(self) -> None:
        self._serial.close()
