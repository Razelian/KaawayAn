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
      node{id}.wav  — stereo 48 kHz int16 WAV
      env.csv       — ENV frame rows per CONTEXT §7
      gaps.log      — AUDIO seq gaps (node, type, seq range, duration ms)
      meta.json     — trial metadata per CONTEXT §7 (no wall-clock timestamps)
    An AUDIO seq gap of k frames inserts k*512 zero stereo samples so that
    sample_index == seq * 512 holds.
    close(trial_id, kind) -> dict(total_gap_ms, flagged, crc_failures,
                                   resync_bytes_skipped)

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

# ---------------------------------------------------------------------------
# Git / config helpers (same pattern as band_diagnostic.py)
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
        # Walk up from this file looking for config/site.toml
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
    """Per-node sink that writes WAV + env.csv + gaps.log + meta.json.

    Parameters
    ----------
    node_id : int
        Expected node_id for frames fed to this recorder.
    out_dir : pathlib.Path | str
        Directory where output files are written (created if absent).
    """

    def __init__(self, node_id: int, out_dir: pathlib.Path | str) -> None:
        self.node_id = node_id
        self._out = pathlib.Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)

        # WAV writer (opened on first AUDIO frame)
        self._wav: wave.Wave_write | None = None
        self._wav_path = self._out / f"node{node_id}.wav"

        # Gap tracking per (type)
        self._last_audio_seq: int | None = None
        self._total_gap_samples: int = 0  # in audio samples (stereo pairs)

        # Seq tracking per type for all types
        self._last_seq: dict[FrameType, int] = {}

        # STARTUP info
        self.fw_version: str | None = None
        self.mic_flags: int | None = None

        # Gaps log
        self._gaps_log_path = self._out / "gaps.log"
        self._gaps_log_lines: list[str] = []

        # ENV CSV
        self._env_rows: list[dict] = []

        # Counters (set from assembler at close)
        self.crc_failures: int = 0
        self.resync_bytes_skipped: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_wav(self) -> None:
        if self._wav is not None:
            return
        self._wav = wave.open(str(self._wav_path), "wb")
        self._wav.setnchannels(2)
        self._wav.setsampwidth(2)   # int16
        self._wav.setframerate(_FS)

    def _write_zero_gap(self, n_samples: int) -> None:
        """Write n_samples of zero stereo pairs to WAV."""
        self._open_wav()
        zeros = np.zeros((n_samples, 2), dtype=np.int16)
        self._wav.writeframes(zeros.tobytes())  # type: ignore[union-attr]

    def _write_audio(self, payload: bytes) -> None:
        self._open_wav()
        self._wav.writeframes(payload)  # type: ignore[union-attr]

    def _log_gap(self, frame_type: str, seq_from: int, seq_to: int,
                  duration_ms: float) -> None:
        line = (
            f"node_id={self.node_id} type={frame_type} "
            f"seq={seq_from}..{seq_to} duration_ms={duration_ms:.2f}"
        )
        self._gaps_log_lines.append(line)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_frame(self, frame: dict, t_laptop_s: float) -> None:
        """Dispatch one decoded frame to the appropriate handler."""
        nid = frame.get("node_id")
        if nid != self.node_id:
            log.warning(
                "NodeRecorder(%d): received frame from unknown node_id=%s; skipping",
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
            # First frame seen — defines baseline; no gap before it
            self._last_audio_seq = seq
        else:
            expected_seq = self._last_audio_seq + 1
            if seq > expected_seq:
                gap_frames = seq - expected_seq
                gap_samples = gap_frames * _SAMPLES_PER_AUDIO
                duration_ms = gap_frames * _SAMPLES_PER_AUDIO / _FS * 1000.0
                self._write_zero_gap(gap_samples)
                self._total_gap_samples += gap_samples
                self._log_gap("AUDIO", expected_seq, seq - 1, duration_ms)
            self._last_audio_seq = seq

        self._write_audio(frame["payload"])

    def _handle_env(self, frame: dict, t_laptop_s: float) -> None:
        try:
            ep = EnvPayload.from_bytes(frame["payload"])
        except Exception as exc:
            log.warning("NodeRecorder(%d): bad ENV payload: %s", self.node_id, exc)
            return
        self._env_rows.append({
            "t_laptop_s": t_laptop_s,
            "temp_c": ep.temp_c_x100 / 100.0,
            "rh_pct": ep.rh_x100 / 100.0,
            "overrun_count": ep.overrun_count,
            "usb_drop_count": ep.usb_drop_count,
            "bme_status": ep.bme_status,
        })

    def _handle_startup(self, frame: dict) -> None:
        try:
            sp = StartupPayload.from_bytes(frame["payload"])
            self.fw_version = f"{sp.fw_major}.{sp.fw_minor}.{sp.fw_patch}"
            self.mic_flags = sp.mic_flags
        except Exception as exc:
            log.warning("NodeRecorder(%d): bad STARTUP payload: %s", self.node_id, exc)

    def close(self, trial_id: str, kind: str,
               cfg_path: pathlib.Path | None = None,
               assembler: "FrameAssembler | None" = None) -> dict[str, Any]:
        """Finalise all files and return summary dict.

        Parameters
        ----------
        trial_id, kind:
            Stored verbatim in meta.json.
        cfg_path:
            Path to site.toml for hashing; auto-detected if None.
        assembler:
            If supplied, its crc_failures and resync_bytes_skipped are
            echoed into the result dict and meta.json.
        """
        if assembler is not None:
            self.crc_failures = assembler.crc_failures
            self.resync_bytes_skipped = assembler.resync_bytes_skipped
        # Close WAV
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        elif not self._wav_path.exists():
            # Create empty WAV so the file always exists
            self._open_wav()
            self._wav.close()  # type: ignore[union-attr]
            self._wav = None

        # Write gaps.log
        if self._gaps_log_lines:
            with open(self._gaps_log_path, "w") as f:
                for line in self._gaps_log_lines:
                    f.write(line + "\n")

        # Write env.csv
        env_path = self._out / "env.csv"
        fieldnames = ["t_laptop_s", "temp_c", "rh_pct", "overrun_count",
                       "usb_drop_count", "bme_status"]
        with open(env_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._env_rows:
                writer.writerow(row)

        # Compute summary
        total_gap_ms = self._total_gap_samples / _FS * 1000.0
        flagged = total_gap_ms > 100.0

        # Write meta.json (no wall-clock timestamps)
        meta: dict[str, Any] = {
            "trial_id": trial_id,
            "kind": kind,
            "config_hash": _config_hash(cfg_path),
            "git_commit": _git_commit(),
            "fw_version": self.fw_version,
            "mic_flags": self.mic_flags,
            "total_gap_ms": total_gap_ms,
            "flagged": flagged,
            "crc_failures": self.crc_failures,
            "resync_bytes_skipped": self.resync_bytes_skipped,
        }
        meta_path = self._out / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

        return {
            "total_gap_ms": total_gap_ms,
            "flagged": flagged,
            "crc_failures": self.crc_failures,
            "resync_bytes_skipped": self.resync_bytes_skipped,
        }


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
    """

    def __init__(self, port_or_serial: str, baudrate: int = 115200) -> None:  # pragma: no cover
        import serial
        import serial.tools.list_ports

        # Try to match by usb_serial first
        matched_port = None
        for info in serial.tools.list_ports.comports():
            if info.serial_number == port_or_serial:
                matched_port = info.device
                break

        if matched_port is None:
            # Fall back to treating it as a direct port name
            matched_port = port_or_serial

        self._serial = serial.Serial(matched_port, baudrate=baudrate, timeout=0.05)

    def read(self, size: int = 256) -> bytes:  # pragma: no cover
        return self._serial.read(size)

    @property
    def is_open(self) -> bool:  # pragma: no cover
        return self._serial.is_open

    def close(self) -> None:  # pragma: no cover
        self._serial.close()
