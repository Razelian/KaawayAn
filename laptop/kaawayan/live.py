"""
kaawayan.live — Real-time bearing/position pipeline.

LivePipeline(cfg, poses, beta=None)
    push_audio(node_id, t_laptop_s, frame_int16)  — add (n,2) int16 audio
    push_env(node_id, temp_c)                     — update temperature
    poll() -> list[dict]                          — drain pending results

replay_trial(folder) -> list[dict]
    Feed node{id}.wav and node{id}_env.csv with t_laptop_s derived from
    t_first_audio_laptop_s in meta.json. Writes bearings.csv and positions.csv.

Each result dict is JSON-serialisable and contains:
    t_laptop_s, x_m (or null), y_m (or null), crossing_deg, n_nodes,
    status, bearings {id: {theta_deg, confidence, gate_ok}},
    config_hash, convention_version.

x_m/y_m are None unless status == "ok" (never stale/held).
A node with no audio for > 2*window_s -> status "degraded" when used.

UNTESTED ON HARDWARE:
  - __main__ runner from SerialSource
  - Reconnect handling (plan L2) is OUT OF SCOPE
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import pathlib
import wave
from collections import deque
from typing import Any

import numpy as np

from kaawayan.bearing import estimate_bearing, speed_of_sound
from kaawayan.fusion import fuse, pair_windows, RollingMedianXY, NodePose
from kaawayan.windows import iter_windows, band_energy_db


# ---------------------------------------------------------------------------
# _NodeBuffer — per-node ring buffer for audio
# ---------------------------------------------------------------------------

class _NodeBuffer:
    """Per-node ring buffer accumulating int16 stereo audio."""

    def __init__(self, node_id: int, window_s: float, hop_s: float, fs: int) -> None:
        self.node_id = node_id
        self._window_samples = int(round(window_s * fs))
        self._hop_samples = int(round(hop_s * fs))
        self._fs = fs
        self._buf: list[np.ndarray] = []  # list of (n, 2) int16 arrays
        self._total_samples: int = 0
        self._emitted_samples: int = 0  # how many samples consumed so far
        self.temp_c: float = 20.0
        self.t_base: float | None = None   # t_laptop_s for sample 0
        self.last_push_t: float | None = None

    def push(self, t_laptop_s: float, frame: np.ndarray) -> None:
        """Add an (n, 2) int16 frame to the buffer."""
        if self.t_base is None:
            self.t_base = t_laptop_s
        self.last_push_t = t_laptop_s
        self._buf.append(frame.astype(np.int16))
        self._total_samples += len(frame)

    def _get_flat(self) -> np.ndarray:
        """Return all buffered audio as (N, 2) array."""
        if not self._buf:
            return np.zeros((0, 2), dtype=np.int16)
        return np.concatenate(self._buf, axis=0)

    def drain_windows(self) -> list[tuple[float, np.ndarray]]:
        """Return list of (t_window_end_laptop_s, window) ready for processing.

        t_window_end = t_base + (emitted + window_samples) / fs
        """
        flat = self._get_flat()
        n_total = self._total_samples
        results: list[tuple[float, np.ndarray]] = []

        while (self._emitted_samples + self._window_samples) <= n_total:
            start = self._emitted_samples
            end = start + self._window_samples
            win = flat[start:end]
            if self.t_base is not None:
                t_end = self.t_base + end / self._fs
            else:
                t_end = end / self._fs
            results.append((t_end, win))
            self._emitted_samples += self._hop_samples

        # Trim consumed audio from the front to save memory
        keep_from = self._emitted_samples - self._hop_samples  # keep last hop
        if keep_from > 0 and self._buf:
            # Rebuild buf from flat array starting at max(0, keep_from - window)
            keep = max(0, self._emitted_samples - self._window_samples)
            self._buf = [flat[keep:]]
            self._total_samples = len(flat) - keep
            self._emitted_samples -= keep

        return results


# ---------------------------------------------------------------------------
# LivePipeline
# ---------------------------------------------------------------------------

class LivePipeline:
    """Real-time bearing/position pipeline.

    Parameters
    ----------
    cfg : dict
        Parsed config dict (from config.load_config).
    poses : dict[int, NodePose]
        Surveyed node positions/headings.
    beta : float | None
        GCC-PHAT exponent; defaults to cfg["phat_beta"] or 1.0.
    """

    CONVENTION_VERSION = 1

    def __init__(self, cfg: dict, poses: dict[int, NodePose],
                 beta: float | None = None) -> None:
        self._cfg = cfg
        self._poses = poses
        self._fs: int = 48_000

        self._window_s: float = cfg["window_s"]
        self._hop_s: float = cfg["hop_s"]
        self._d_m: float = cfg["mic_spacing_m"]
        self._band_hz: tuple[float, float] = tuple(cfg["band_hz"])  # type: ignore
        self._pair_tol: float = cfg.get("pair_tolerance_s", 0.1)
        self._gate_min_conf: float = cfg.get("gate_min_confidence", 0.0)
        self._gate_min_energy_db: float = cfg.get("gate_min_band_energy_db", -120.0)
        self._min_cross_deg: float = cfg.get("min_crossing_deg", 30.0)
        self._beta: float = beta if beta is not None else cfg.get("phat_beta", 1.0)

        # Per-node config
        self._tau0: dict[int, float] = {}
        self._swap: dict[int, bool] = {}
        for node in cfg.get("nodes", []):
            nid = node["node_id"]
            self._tau0[nid] = node.get("tau0_s", 0.0)
            self._swap[nid] = node.get("channel_swap", False)

        # Per-node buffers
        self._bufs: dict[int, _NodeBuffer] = {
            nid: _NodeBuffer(nid, self._window_s, self._hop_s, self._fs)
            for nid in poses
        }

        # Rolling median smoother (last 5 ok fixes)
        self._smoother = RollingMedianXY(5)

        # Pending bearing records: (node_id, t_end, theta_deg, confidence)
        self._pending: list[tuple[int, float, float, float]] = []
        # (also store full bearing info for reporting)
        self._pending_full: list[dict] = []

        # Config hash
        self._config_hash: str = "n/a"

        # Results queue
        self._results: list[dict] = []

    def set_config_hash(self, h: str) -> None:
        self._config_hash = h

    def push_audio(self, node_id: int, t_laptop_s: float,
                   frame_int16: np.ndarray) -> None:
        """Add an (n, 2) int16 stereo audio chunk for one node."""
        buf = self._bufs.get(node_id)
        if buf is None:
            return
        buf.push(t_laptop_s, frame_int16)
        self._process_node(node_id, buf)

    def push_env(self, node_id: int, temp_c: float) -> None:
        """Update the temperature estimate for a node."""
        buf = self._bufs.get(node_id)
        if buf is None:
            return
        buf.temp_c = temp_c

    def _process_node(self, node_id: int, buf: _NodeBuffer) -> None:
        """Drain any complete windows from the node buffer, compute bearings."""
        pose = self._poses.get(node_id)
        if pose is None:
            return
        tau0 = self._tau0.get(node_id, 0.0)
        swap = self._swap.get(node_id, False)

        for t_end, win in buf.drain_windows():
            left = win[:, 0].astype(np.float64)
            right = win[:, 1].astype(np.float64)
            c = speed_of_sound(buf.temp_c)

            res = estimate_bearing(
                left, right, self._fs, self._d_m, buf.temp_c,
                band_hz=self._band_hz,
                tau0_s=tau0,
                channel_swap=swap,
                beta=self._beta,
            )

            energy_db = band_energy_db(win, self._fs, self._band_hz)
            gate_ok = (res["confidence"] >= self._gate_min_conf and
                       energy_db >= self._gate_min_energy_db)

            # Store pending bearing record
            self._pending.append(
                (node_id, t_end, res["theta_deg"], res["confidence"])
            )
            self._pending_full.append({
                "node_id": node_id,
                "t_end": t_end,
                "theta_deg": res["theta_deg"],
                "confidence": res["confidence"],
                "gate_ok": gate_ok,
            })

        # Try to form fused results
        self._try_fuse()

    def _try_fuse(self) -> None:
        """Pair pending bearing windows and fuse to positions."""
        if len(self._pending) < 2:
            return

        # pair_windows expects (node_id, t_laptop_s, theta_deg, confidence)
        groups = pair_windows(self._pending, self._pair_tol)

        # Find how far we consumed
        used_records: set[int] = set()
        results_to_emit: list[dict] = []

        for group in groups:
            # Only process groups that include at least 2 nodes
            if len(group) < 2:
                continue

            # Check if we have any fully paired group not yet consumed
            # (group records are tuples from self._pending)
            node_ids_in_group = {r[0] for r in group}
            t_anchor = max(r[1] for r in group)

            # Build bearings list — only gated bearings go to fuse
            bearings_report: dict[int, dict] = {}
            bearings_for_fuse: list[tuple[int, float, float]] = []

            for record in group:
                nid, t_end, theta_deg, conf = record
                # Find full info
                full = next(
                    (f for f in self._pending_full
                     if f["node_id"] == nid and abs(f["t_end"] - t_end) < 1e-9
                     and abs(f["theta_deg"] - theta_deg) < 1e-9),
                    None,
                )
                gate_ok = full["gate_ok"] if full else True
                bearings_report[nid] = {
                    "theta_deg": theta_deg,
                    "confidence": conf,
                    "gate_ok": gate_ok,
                }
                if gate_ok:
                    bearings_for_fuse.append((nid, theta_deg, conf))

            # Fuse
            if not bearings_for_fuse or len(bearings_for_fuse) < 2:
                fuse_result = {
                    "x_m": None, "y_m": None,
                    "crossing_deg": None, "n_lines": 0,
                    "used_node_ids": [], "status": "uncertain",
                    "reason": "insufficient_gated_bearings",
                }
            else:
                try:
                    fuse_result = fuse(
                        bearings_for_fuse,
                        self._poses,
                        min_crossing_deg=self._min_cross_deg,
                        gate_min_confidence=0.0,  # already gated above
                    )
                except ValueError:
                    fuse_result = {
                        "x_m": None, "y_m": None,
                        "crossing_deg": None, "n_lines": 0,
                        "used_node_ids": [], "status": "uncertain",
                        "reason": "fuse_error",
                    }

            status = fuse_result.get("status", "uncertain")

            # Check for degraded: any node has no audio for > 2*window_s
            now_t = t_anchor
            any_degraded = False
            for nid, buf in self._bufs.items():
                if buf.last_push_t is not None and (now_t - buf.last_push_t) > 2 * self._window_s:
                    any_degraded = True
                    break
            if any_degraded and status == "ok":
                status = "degraded"

            # Only emit x_m/y_m when status == "ok"
            if status == "ok":
                x_m: float | None = fuse_result.get("x_m")
                y_m: float | None = fuse_result.get("y_m")
            else:
                x_m = None
                y_m = None

            crossing_deg = fuse_result.get("crossing_deg")

            result: dict[str, Any] = {
                "t_laptop_s": t_anchor,
                "x_m": x_m,
                "y_m": y_m,
                "crossing_deg": crossing_deg,
                "n_nodes": len(group),
                "status": status,
                "bearings": {
                    str(nid): v for nid, v in bearings_report.items()
                },
                "config_hash": self._config_hash,
                "convention_version": self.CONVENTION_VERSION,
            }
            results_to_emit.append(result)

        # Emit all fully-processed results
        self._results.extend(results_to_emit)

        # Remove consumed records from pending (groups with >= 2 nodes)
        paired_tuples: set[tuple] = set()
        for group in groups:
            if len(group) >= 2:
                for rec in group:
                    paired_tuples.add(rec)

        self._pending = [r for r in self._pending if r not in paired_tuples]
        self._pending_full = [
            f for f in self._pending_full
            if not any(
                f["node_id"] == r[0] and abs(f["t_end"] - r[1]) < 1e-9
                for r in paired_tuples
            )
        ]

    def poll(self) -> list[dict]:
        """Drain and return all pending results."""
        out = self._results
        self._results = []
        return out


# ---------------------------------------------------------------------------
# replay_trial
# ---------------------------------------------------------------------------

def replay_trial(folder: pathlib.Path | str) -> list[dict]:
    """Replay a recorded trial through LivePipeline and write CSVs.

    Reads node{id}.wav and node{id}_env.csv for each node listed in meta.json.
    t_laptop_s = meta["nodes"][id]["t_first_audio_laptop_s"] (0.0 if missing)
                 + sample_index / fs

    Writes:
        bearings.csv:  t_laptop_s, node_id, theta_deg, confidence, gate_ok
        positions.csv: t_laptop_s, x_m, y_m, crossing_deg, n_nodes, status

    Returns list of result dicts.
    """
    folder = pathlib.Path(folder)
    meta_path = folder / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    # Load config — try to find site.toml from meta config_hash or fallback
    # We just use a minimal config-like dict extracted from meta where possible.
    # For replay, we need the full config; callers should ensure the right cfg
    # is passed. Here we look for a config path relative to the trial.
    # Since evaluate.py just reads positions.csv, we write based on pipeline.
    from kaawayan.config import load_config
    from kaawayan.fusion import poses_from_config

    # Try to locate site.toml
    cfg_path_candidates = [
        folder.parent.parent / "config" / "site.toml",
        folder.parent / "site.toml",
        pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "site.toml",
    ]
    cfg_path: pathlib.Path | None = None
    for c in cfg_path_candidates:
        if c.exists():
            cfg_path = c
            break

    if cfg_path is None:
        raise FileNotFoundError("Could not locate site.toml for replay_trial")

    cfg, _ = load_config(cfg_path)
    poses = poses_from_config(cfg)
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(config_hash)

    fs: int = 48_000
    nodes_meta: dict = meta.get("nodes", {})

    # Feed env (temperature) first for each node
    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        env_path = folder / f"node{nid}_env.csv"
        if env_path.exists():
            with open(env_path) as f:
                for row in csv.DictReader(f):
                    try:
                        pipeline.push_env(nid, float(row["temp_c"]))
                    except (KeyError, ValueError):
                        pass

    # Feed audio
    all_results: list[dict] = []
    _CHUNK = 512

    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        wav_path = folder / f"node{nid}.wav"
        if not wav_path.exists():
            continue

        node_meta = nodes_meta.get(str(nid), {})
        t0 = float(node_meta.get("t_first_audio_laptop_s", 0.0))

        with wave.open(str(wav_path)) as wf:
            n_frames = wf.getnframes()
            sample_idx = 0
            while sample_idx < n_frames:
                chunk_frames = min(_CHUNK, n_frames - sample_idx)
                raw = wf.readframes(chunk_frames)
                arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
                t_laptop_s = t0 + sample_idx / fs
                pipeline.push_audio(nid, t_laptop_s, arr)
                sample_idx += chunk_frames

    all_results = pipeline.poll()

    # Write bearings.csv
    bearings_path = folder / "bearings.csv"
    with open(bearings_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "t_laptop_s", "node_id", "theta_deg", "confidence", "gate_ok"
        ])
        w.writeheader()
        for res in all_results:
            for nid_str, binfo in res.get("bearings", {}).items():
                w.writerow({
                    "t_laptop_s": res["t_laptop_s"],
                    "node_id": nid_str,
                    "theta_deg": binfo["theta_deg"],
                    "confidence": binfo["confidence"],
                    "gate_ok": binfo["gate_ok"],
                })

    # Write positions.csv
    positions_path = folder / "positions.csv"
    with open(positions_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "t_laptop_s", "x_m", "y_m", "crossing_deg", "n_nodes", "status"
        ])
        w.writeheader()
        for res in all_results:
            # x_m/y_m blank unless ok
            status = res["status"]
            w.writerow({
                "t_laptop_s": res["t_laptop_s"],
                "x_m": res["x_m"] if status == "ok" else "",
                "y_m": res["y_m"] if status == "ok" else "",
                "crossing_deg": res.get("crossing_deg", ""),
                "n_nodes": res["n_nodes"],
                "status": status,
            })

    return all_results


# ---------------------------------------------------------------------------
# __main__ — UNTESTED ON HARDWARE
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    """Live runner printing JSON lines from SerialSource. UNTESTED ON HARDWARE.
    Reconnect handling (plan L2) is OUT OF SCOPE.
    """
    import sys
    import time
    from kaawayan.config import load_config
    from kaawayan.fusion import poses_from_config
    from kaawayan.record import FrameAssembler, SerialSource

    if len(sys.argv) < 2:
        print("Usage: python -m kaawayan.live <site.toml>", file=sys.stderr)
        sys.exit(1)

    cfg_path = pathlib.Path(sys.argv[1])
    cfg, _ = load_config(cfg_path)
    poses = poses_from_config(cfg)
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(config_hash)

    assemblers: dict = {}
    sources: dict = {}
    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        src = SerialSource(node["usb_serial"])
        sources[nid] = src
        assemblers[nid] = FrameAssembler()

    t_start = time.monotonic()
    try:
        while True:
            for nid, src in sources.items():
                data = src.read()
                if data:
                    for frame, _ in assemblers[nid].feed(data, time.monotonic() - t_start):
                        if frame["type"].name == "AUDIO":
                            arr = np.frombuffer(frame["payload"], dtype=np.int16).reshape(-1, 2)
                            pipeline.push_audio(nid, time.monotonic() - t_start, arr)
                        elif frame["type"].name == "ENV":
                            from kaawayan.protocol import EnvPayload
                            ep = EnvPayload.from_bytes(frame["payload"])
                            pipeline.push_env(nid, ep.temp_c_x100 / 100.0)
            for res in pipeline.poll():
                print(json.dumps(res, sort_keys=True))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
