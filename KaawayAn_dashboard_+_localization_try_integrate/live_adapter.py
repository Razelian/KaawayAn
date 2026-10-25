"""Ingest adapter: SerialSource/FakeSource -> FrameAssembler -> LivePipeline + classifier.

Does not reimplement gating, pairing, or smoothing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import resample_poly

from config import Config
from database.database import Database
from kaawayan.live import LivePipeline
from kaawayan.protocol import AudioPayload, EnvPayload, FrameType, StartupPayload
from kaawayan.record import FrameAssembler, TrialRecorder

log = logging.getLogger("kaawayan.dash")

CLASSIFIER_IN_N = int(Config.AUDIO_FS)  # 48000 left-channel samples -> 16000 out


class ClassifierWorker:
    """Bounded queue; drop when full. Never blocks the caller."""

    def __init__(
        self,
        predict: Callable[[np.ndarray], float],
        database: Database,
        threshold: float,
        maxsize: int = Config.CLASSIFIER_QUEUE_MAX,
        status: str = "ok",
    ) -> None:
        self.predict = predict
        self.database = database
        self.threshold = threshold
        self.queue: Queue = Queue(maxsize=maxsize)
        self.dropped = 0
        self.last_len: int | None = None
        self.status = status
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        self._drone_lock = threading.Lock()
        self._last_drone_t: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        self.database.set_system_status("classifier", self.status, time.time())
        self.database.set_system_status("dropped_windows", "0", time.time())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def check_watchdog(self) -> None:
        """Restart worker thread if it died unexpectedly."""
        if not self._stop.is_set() and not self._thread.is_alive():
            log.warning("Classifier worker thread was dead; restarting via watchdog")
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def submit(self, node_id: int, t_wall: float, left: np.ndarray) -> None:
        try:
            self.queue.put_nowait((int(node_id), float(t_wall), np.asarray(left).copy()))
        except Full:
            self.dropped += 1
            self.database.set_system_status("dropped_windows", str(self.dropped), t_wall)

    def drone_recent(self, now: float, window_s: float = Config.DRONE_RECENT_S) -> bool:
        with self._drone_lock:
            if self._last_drone_t is None:
                return False
            return (now - self._last_drone_t) <= window_s

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                try:
                    node_id, t_wall, left = self.queue.get(timeout=0.1)
                except Empty:
                    continue

                x = np.asarray(left, dtype=np.float64).ravel()
                x = x - np.mean(x)

                # Store input level: RMS dBFS (peak full-scale for 16-bit audio is 32768.0)
                norm_audio = x / 32768.0
                rms = float(np.sqrt(np.mean(norm_audio ** 2)))
                rms_dbfs = float(20.0 * np.log10(rms)) if rms > 1e-12 else -120.0

                # Optional normalization (default is None = current behavior preserved)
                norm_target = getattr(Config, "CLASSIFIER_NORMALIZE_RMS_DBFS", None)
                if norm_target is not None and rms > 1e-12:
                    target_rms = 10.0 ** (norm_target / 20.0)
                    x = x * (target_rms / rms)

                samples = resample_poly(x, 1, 3)
                self.last_len = int(len(samples))

                prob = float(self.predict(samples))
                predicted = 1 if prob >= self.threshold else 0
                if predicted:
                    with self._drone_lock:
                        self._last_drone_t = t_wall

                # Fill nearest bearing within 1.0 s for this node
                theta_deg = self.database.get_nearest_bearing_theta(node_id, t_wall, max_dt=1.0)

                self.database.log_detection(
                    node_id=node_id,
                    predicted_drone=predicted,
                    model_confidence=prob,
                    timestamp=f"{t_wall:.6f}",
                    theta_deg=theta_deg,
                    t_wall=t_wall,
                    rms_dbfs=rms_dbfs,
                )

                self.consecutive_errors = 0
                if self.status != "disabled":
                    self.status = "ok"
                    self.database.set_system_status("classifier", "ok", t_wall)
                    self.database.set_system_status("classifier_last_ok_t", f"{t_wall:.6f}", t_wall)

            except Exception as exc:
                self.consecutive_errors += 1
                log.error("ClassifierWorker error (%d consecutive): %s", self.consecutive_errors, exc)
                if self.consecutive_errors >= self.max_consecutive_errors:
                    self.status = "failed"
                    self.database.set_system_status("classifier", "failed", time.time())


class _NodeAudio:
    """Classifier-only windowing (does not affect LivePipeline)."""

    def __init__(self, window_n: int, hop_n: int) -> None:
        self.window_n = window_n
        self.hop_n = hop_n
        self.chunks: list[np.ndarray] = []
        self.n = 0

    def push(self, mono: np.ndarray) -> list[np.ndarray]:
        self.chunks.append(mono)
        self.n += len(mono)
        out: list[np.ndarray] = []
        while self.n >= self.window_n:
            full = np.concatenate(self.chunks, axis=0)
            out.append(full[: self.window_n])
            rem = full[self.hop_n :]
            self.chunks = [rem] if len(rem) else []
            self.n = len(rem)
        return out


def store_pipeline_result(
    db: Database,
    result: dict[str, Any],
    drone_recent: bool,
) -> None:
    """Write LivePipeline result fields. Gate-fail groups become status 'none'."""
    status = str(result.get("status") or "uncertain")
    reason = str(result.get("reason") or "")
    bearings = result.get("bearings") or {}
    any_gate = any(bool(b.get("gate_ok")) for b in bearings.values())
    if (not any_gate) or reason == "too_few_gated_bearings":
        status = "none"

    t_wall = float(result["t_laptop_s"])
    db.log_localization(
        timestamp=f"{t_wall:.6f}",
        x_m=result.get("x_m"),
        y_m=result.get("y_m"),
        crossing_deg=result.get("crossing_deg"),
        n_nodes=int(result.get("n_nodes") or 0),
        status=status,
        reason=reason,
        t_wall=t_wall,
        drone_recent=drone_recent,
        dead_nodes=result.get("dead_node_ids") or [],
    )
    for nid_s, br in bearings.items():
        db.log_bearing(
            t_wall=t_wall,
            node_id=int(nid_s),
            window_idx=br.get("window_idx"),
            theta_deg=br.get("theta_deg"),
            confidence=br.get("confidence"),
            peak_median_ratio=br.get("peak_median_ratio"),
            gate_ok=bool(br.get("gate_ok")),
        )


def handle_frame(
    frame: dict[str, Any],
    t_wall: float,
    pipeline: LivePipeline,
    db: Database,
    classifier: Optional[ClassifierWorker],
    audio_bufs: dict[int, _NodeAudio],
    recorder: Optional[TrialRecorder] = None,
    seq_trackers: Optional[dict[tuple[int, int], int]] = None,
    node_drops: Optional[dict[int, int]] = None,
) -> None:
    if recorder is not None:
        recorder.feed_frame(frame, t_wall)

    ftype = frame["type"]
    node_id = int(frame["node_id"])

    # Host-side sequence gap detection per node and per frame type
    if seq_trackers is not None and node_drops is not None:
        key = (node_id, int(ftype))
        seq = int(frame.get("seq", 0))
        if key in seq_trackers:
            diff = (seq - (seq_trackers[key] + 1)) & 0xFFFFFFFF
            if 0 < diff < 100000:
                node_drops[node_id] += diff
                db.set_system_status(f"node_{node_id}_usb_drops", str(node_drops[node_id]), t_wall)
        seq_trackers[key] = seq

    if ftype == FrameType.ENV:
        env = EnvPayload.from_bytes(frame["payload"])
        temp_c = env.temp_c_x100 / 100.0
        rh = env.rh_x100 / 100.0
        drop_cnt = env.usb_drop_count
        if drop_cnt == 0 and node_drops and node_drops.get(node_id, 0) > 0:
            drop_cnt = node_drops[node_id]

        db.log_env(
            node_id=node_id,
            timestamp=f"{t_wall:.6f}",
            temp_c=temp_c,
            humidity_pct=rh,
            overrun_count=env.overrun_count,
            usb_drop_count=drop_cnt,
            bme_status=env.bme_status,
            t_wall=t_wall,
        )
        if env.bme_status == 0:
            pipeline.push_env(node_id, temp_c)
        return

    if ftype == FrameType.STARTUP:
        startup = StartupPayload.from_bytes(frame["payload"])
        version = f"{startup.fw_major}.{startup.fw_minor}.{startup.fw_patch}"
        valid = (
            startup.fw_major == 1
            and startup.sample_rate_hz == Config.AUDIO_FS
            and startup.samples_per_ch == 512
            and startup.channels == 2
        )
        db.set_system_status(f"node_{node_id}_firmware_version", version, t_wall)
        db.set_system_status(
            f"node_{node_id}_firmware_status", "ok" if valid else "failed", t_wall
        )
        if not valid:
            log.error(
                "node %s reports incompatible STARTUP firmware=%s sample_rate=%s samples_per_ch=%s channels=%s",
                node_id,
                version,
                startup.sample_rate_hz,
                startup.samples_per_ch,
                startup.channels,
            )
        return

    if ftype != FrameType.AUDIO:
        return

    stereo = AudioPayload(frame["payload"]).to_numpy()
    pipeline.push_audio(node_id, t_wall, stereo)

    if classifier is not None:
        buf = audio_bufs[node_id]
        left = stereo[:, 0].astype(np.float64)
        for window in buf.push(left):
            classifier.submit(node_id, t_wall, window)

    for result in pipeline.poll():
        recent = classifier.drone_recent(float(result["t_laptop_s"])) if classifier else False
        store_pipeline_result(db, result, recent)


def run_pipeline(
    cfg: dict[str, Any],
    pipeline: LivePipeline,
    sources: dict[int, Any],
    db: Database,
    *,
    classifier: Optional[ClassifierWorker] = None,
    recorder: Optional[TrialRecorder] = None,
    source_factory: Optional[Callable[[int], Any]] = None,
    duration_s: Optional[float] = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    stop_event: Optional[threading.Event] = None,
    read_size: int = 4096,
) -> dict[str, Any]:
    """Read sources, assemble frames, push through LivePipeline, write DB.

    Lost nodes are retried with exponential backoff and recorded in system_status.
    """
    assemblers = {nid: FrameAssembler() for nid in sources}
    win_n = int(round(float(cfg["window_s"]) * Config.AUDIO_FS))
    hop_n = int(round(float(cfg["hop_s"]) * Config.AUDIO_FS))
    audio_bufs: dict[int, _NodeAudio] = defaultdict(lambda: _NodeAudio(win_n, hop_n))

    seq_trackers: dict[tuple[int, int], int] = {}
    node_drops: dict[int, int] = defaultdict(int)
    node_frame_errors: dict[int, int] = defaultdict(int)
    node_crc_seen: dict[int, int] = defaultdict(int)

    lost: set[int] = set()
    node_state: dict[int, str] = {nid: "ok" for nid in sources}
    backoff_s: dict[int, float] = {nid: 1.0 for nid in sources}
    next_retry_t: dict[int, float] = {nid: 0.0 for nid in sources}

    t0 = clock()
    last_heartbeat = t0
    last_prune = t0
    idle_rounds = 0
    backlog_warnings = 0

    # Initial state reporting
    for nid in sources:
        db.set_system_status(f"node_{nid}_state", "ok", t0)
    db.set_system_status("ingest_heartbeat_t", f"{t0:.6f}", t0)

    while True:
        now = clock()
        if stop_event is not None and stop_event.is_set():
            break
        if duration_s is not None and (now - t0) >= duration_s:
            break

        # Heartbeat update every 1 second
        if (now - last_heartbeat) >= 1.0:
            last_heartbeat = now
            db.set_system_status("ingest_heartbeat_t", f"{now:.6f}", now)

        # Retention pruning every 60 seconds
        if (now - last_prune) >= 60.0:
            last_prune = now
            cutoff = now - (Config.RETENTION_HOURS * 3600.0)
            db.prune_older_than(cutoff)

        # Check classifier watchdog
        if classifier is not None:
            classifier.check_watchdog()

        # Retry lost nodes if backoff elapsed
        for nid in list(lost):
            if now >= next_retry_t[nid]:
                node_state[nid] = "reconnecting"
                db.set_system_status(f"node_{nid}_state", "reconnecting", now)
                if source_factory is not None:
                    try:
                        new_src = source_factory(nid)
                        sources[nid] = new_src
                        assemblers[nid] = FrameAssembler()
                        lost.discard(nid)
                        node_state[nid] = "ok"
                        backoff_s[nid] = 1.0
                        db.set_system_status(f"node_{nid}_state", "ok", now)
                        log.info("Node %s reconnected successfully", nid)
                    except Exception as exc:
                        log.warning("Reconnect node %s failed: %s", nid, exc)
                        next_retry_t[nid] = now + backoff_s[nid]
                        backoff_s[nid] = min(16.0, backoff_s[nid] * 2.0)
                        db.set_system_status(f"node_{nid}_state", "lost", now)

        any_bytes = False
        loop_start_t = clock()

        for nid, src in list(sources.items()):
            if nid in lost:
                continue
            try:
                data = src.read(read_size)
            except Exception as exc:
                log.error("node %s source lost: %s", nid, exc)
                lost.add(nid)
                node_state[nid] = "lost"
                db.set_system_status(f"node_{nid}_state", "lost", clock())
                next_retry_t[nid] = clock() + backoff_s[nid]
                backoff_s[nid] = min(16.0, backoff_s[nid] * 2.0)
                try:
                    src.close()
                except Exception:
                    pass
                continue

            if not data:
                continue

            any_bytes = True
            t_wall = clock()
            frames = assemblers[nid].feed(data, t_wall)
            assembler = assemblers[nid]
            if assembler.crc_failures > node_crc_seen[nid]:
                node_crc_seen[nid] = assembler.crc_failures
                db.set_system_status(
                    f"node_{nid}_crc_errors", str(node_crc_seen[nid]), t_wall
                )
            for frame, ts in frames:
                try:
                    handle_frame(
                        frame,
                        ts,
                        pipeline,
                        db,
                        classifier,
                        audio_bufs,
                        recorder,
                        seq_trackers=seq_trackers,
                        node_drops=node_drops,
                    )
                except Exception:
                    node_frame_errors[nid] += 1
                    log.exception(
                        "node %s frame processing failed (count=%d)",
                        nid,
                        node_frame_errors[nid],
                    )
                    db.set_system_status(
                        f"node_{nid}_frame_errors",
                        str(node_frame_errors[nid]),
                        clock(),
                    )

        loop_dur = clock() - loop_start_t
        if any_bytes and loop_dur > 0.5:
            backlog_warnings += 1
            log.warning("Processing backlog growing (loop duration=%.3fs, counter=%d)", loop_dur, backlog_warnings)

        if not any_bytes:
            idle_rounds += 1
            # ByteSources used in tests/demo go idle when drained.
            all_exhausted_or_closed = all(
                (nid in lost) or getattr(src, "exhausted", False) or not getattr(src, "is_open", True)
                for nid, src in sources.items()
            )
            if all_exhausted_or_closed:
                # If we have finite exhausted test ByteSources, break out
                if any(getattr(src, "exhausted", False) for src in sources.values()):
                    break
                if duration_s is not None and (clock() - t0) >= duration_s:
                    break
                # Never exit with code 0 when all nodes are lost in live mode
                if duration_s is None and len(lost) == len(sources):
                    sleep(0.05)
                    continue

            if idle_rounds > 50 and duration_s is None:
                idle_rounds = 0
            sleep(0.002)
        else:
            idle_rounds = 0

    pipeline.flush()
    for result in pipeline.poll():
        recent = classifier.drone_recent(float(result["t_laptop_s"])) if classifier else False
        store_pipeline_result(db, result, recent)

    return {"lost_nodes": sorted(lost), "idle_rounds": idle_rounds}


def default_cfg_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "site.toml"
