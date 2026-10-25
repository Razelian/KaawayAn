"""Fake SerialSource that emits encoded v1 frames (same ingest path as hardware)."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from kaawayan.fusion import NodePose
from kaawayan.geosim import simulate_node
from kaawayan.protocol import FrameType, encode_frame

FS = 48_000
FRAME_N = 512


def _to_int16(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 1.0
    scale = 10000.0 / peak if peak > 0 else 1.0
    return np.clip(x * scale, -32767, 32767).astype(np.int16)


def harmonic_signal(
    duration_s: float,
    fs: int = FS,
    f0_hz: float = 180.0,
    n_harmonics: int = 12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    sig = np.zeros(n, dtype=np.float64)
    r = rng if rng is not None else np.random.default_rng(0)
    for k in range(1, n_harmonics + 1):
        phase = r.uniform(0, 2 * np.pi)
        sig += (1.0 / k) * np.sin(2 * np.pi * k * f0_hz * t + phase)
    return sig


def encode_stereo_session(
    stereo: np.ndarray,
    node_id: int,
    temp_c: float = 20.0,
    humidity_pct: float = 50.0,
    bme_status: int = 0,
) -> bytes:
    """Turn (N, 2) int16 audio into AUDIO+ENV v1 frames."""
    n = (len(stereo) // FRAME_N) * FRAME_N
    stereo = np.asarray(stereo[:n], dtype=np.int16)
    out = bytearray()
    audio_seq = 0
    env_seq = 0
    temp_x100 = int(round(temp_c * 100))
    rh_x100 = int(round(humidity_pct * 100))
    env_payload = struct.pack("<hHIIB", temp_x100, rh_x100, 0, 0, bme_status)
    # Emit initial boot ENV frame
    out.extend(
        encode_frame(FrameType.ENV, node_id, env_seq, 0, env_payload)
    )
    env_seq += 1

    for i in range(0, n, FRAME_N):
        chunk = stereo[i : i + FRAME_N]
        payload = np.ascontiguousarray(chunk, dtype="<i2").tobytes()
        t_us = int(i / FS * 1_000_000)
        out.extend(
            encode_frame(FrameType.AUDIO, node_id, audio_seq, t_us, payload)
        )
        audio_seq += 1
        if audio_seq % 94 == 0:
            out.extend(
                encode_frame(FrameType.ENV, node_id, env_seq, t_us, env_payload)
            )
            env_seq += 1
    return bytes(out)


def build_node_stereo(
    source_xy: tuple[float, float],
    pose: NodePose,
    signal: np.ndarray,
    cfg: dict[str, Any],
    snr_db: float,
    rng: np.random.Generator,
    temp_c: float = 20.0,
) -> np.ndarray:
    left, right = simulate_node(
        source_xy,
        pose,
        signal,
        fs=float(FS),
        d_m=float(cfg["mic_spacing_m"]),
        temp_c=temp_c,
        snr_db=snr_db,
        rng=rng,
    )
    return np.column_stack((_to_int16(left), _to_int16(right)))


class ByteSource:
    """SerialSource-compatible: read() / is_open / close()."""

    def __init__(self, payload: bytes, chunk: int = 4096, loop: bool = False):
        self._payload = payload
        self._chunk = chunk
        self._loop = loop
        self._off = 0
        self._open = True

    def read(self, size: int = 4096) -> bytes:
        if not self._open:
            return b""
        n = size if size else self._chunk
        if self._off >= len(self._payload):
            if not self._loop or not self._payload:
                return b""
            self._off = 0
        end = min(self._off + n, len(self._payload))
        data = self._payload[self._off : end]
        self._off = end
        return data

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def exhausted(self) -> bool:
        if self._loop or not self._open:
            return False
        return self._off >= len(self._payload)

    def close(self) -> None:
        self._open = False


def build_demo_sources(
    cfg: dict[str, Any],
    poses: dict[int, NodePose],
    *,
    duration_s: float = 8.0,
    source_xy: tuple[float, float] = (1.5, 3.0),
    snr_db: float = 20.0,
    f0_hz: float = 180.0,
    noise_only: bool = False,
    seed: int = 0,
    loop: bool = False,
    temp_c: float = 20.0,
    bme_status: int = 0,
) -> dict[int, ByteSource]:
    rng = np.random.default_rng(seed)
    if noise_only:
        n = int(round(duration_s * FS))
        signal = np.zeros(n, dtype=np.float64)
        snr_db = -120.0
    else:
        signal = harmonic_signal(duration_s, FS, f0_hz, rng=rng)
    sources: dict[int, ByteSource] = {}
    for nid, pose in poses.items():
        stereo = build_node_stereo(
            source_xy, pose, signal, cfg, snr_db, rng, temp_c=temp_c
        )
        raw = encode_stereo_session(
            stereo, nid, temp_c=temp_c, bme_status=bme_status
        )
        sources[nid] = ByteSource(raw, loop=loop)
    return sources
