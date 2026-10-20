"""
kaawayan.live — Real-time bearing/position pipeline.

LivePipeline(cfg, poses, beta=None)
    push_audio(node_id, t_laptop_s, frame_int16)  — add (n,2) int16 audio
    push_env(node_id, temp_c)                     — update temperature
    flush()                                       — emit all pending groups
    poll() -> list[dict]                          — drain pending results
    set_config_hash(h)

replay_trial(folder, cfg_path=None) -> list[dict]
    Feed node{id}.wav and node{id}_env.csv interleaved in order of
    (t_laptop_s, node_id), chunks of 512 samples. Calls flush() at the end.
    Writes bearings.csv (with window_idx, peak_median_ratio) and positions.csv.
    Returns result list.

Each result dict is JSON-serialisable (json.dumps(sort_keys=True,
allow_nan=False) always works) and contains:
    t_laptop_s, x_m, y_m, x_raw_m, y_raw_m, crossing_deg,
    n_nodes, dead_node_ids, status, reason,
    bearings {str(id): {window_idx, theta_deg, confidence,
                        peak_median_ratio, gate_ok}},
    config_hash, convention_version.

Bug fixes vs original:
  (a) t_end tracks an ABSOLUTE sample index; t_base is t_laptop_s of first
      push for this node (later arrivals ignored). Buffer keeps only needed
      samples.
  (b) flush() emits all pending groups. Liveness:
      - A node is "alive" if it has produced a window with
        t_end >= latest_t - 2*window_s (latest_t = newest window end across
        all nodes).
      - A node never seen is dead once latest_t - first_window_t > 2*window_s,
        where first_window_t is the t_end of the first-ever window across
        all nodes.
      - A group is FINAL when it holds a record from every alive node, OR
        anchor_t + pair_tolerance_s < min(alive nodes' latest window end).
      - Pending stays bounded.
  (c) x_m/y_m from RollingMedianXY(5); x_raw_m/y_raw_m unsmoothed.
      status: fuse status; if fuse not "ok" and dead_node_ids non-empty
      -> "degraded". Dead nodes with rest fusing "ok" -> status stays "ok".
  (d) All numpy scalars cast to Python float/bool/int.

UNTESTED ON HARDWARE:
  - __main__ runner from SerialSource
  - Reconnect handling (plan L2) is OUT OF SCOPE
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import sys
import wave
from typing import Any

import numpy as np

from kaawayan.bearing import estimate_bearing
from kaawayan.fusion import fuse, RollingMedianXY, NodePose
from kaawayan.windows import band_energy_db


# ---------------------------------------------------------------------------
# _NodeBuffer — per-node ring buffer using absolute sample index
# ---------------------------------------------------------------------------

class _NodeBuffer:
    """Per-node buffer accumulating int16 stereo audio.

    t_base is set on the FIRST push and never changed. Subsequent pushes are
    appended; their t_laptop_s argument is ignored for timing (documented:
    only the first arrival sets the clock origin).

    Window k covers absolute samples [k*hop, k*hop+win). After draining,
    samples before the oldest needed hop are discarded so memory stays bounded.
    """

    def __init__(self, node_id: int, window_s: float, hop_s: float,
                 fs: int) -> None:
        self.node_id = node_id
        self._win = int(round(window_s * fs))
        self._hop = int(round(hop_s * fs))
        self._fs = fs
        # Raw sample storage (may be trimmed at front)
        self._buf: list[np.ndarray] = []   # list of (n,2) int16
        self._buf_offset: int = 0          # absolute index of buf[0][0]
        self._total: int = 0               # absolute count of samples pushed
        # Next window index to emit
        self._next_k: int = 0
        # Latest emitted window t_end
        self.latest_t_end: float | None = None
        # Timing anchor: t_laptop_s for absolute sample 0
        self.t_base: float | None = None
        # Temperature (updated by push_env)
        self.temp_c: float = 20.0

    def push(self, t_laptop_s: float, frame: np.ndarray) -> None:
        """Add an (n, 2) int16 frame. First push sets t_base."""
        if self.t_base is None:
            self.t_base = t_laptop_s
        self._buf.append(frame.astype(np.int16))
        self._total += len(frame)

    def drain_windows(self) -> list[tuple[int, float, np.ndarray]]:
        """Return list of (window_idx, t_end, window_array) for complete windows.

        t_end = t_base + (k*hop + win) / fs
        After draining, trims buffer to keep only samples still needed.
        """
        if self.t_base is None:
            return []

        results: list[tuple[int, float, np.ndarray]] = []

        while True:
            k = self._next_k
            abs_start = k * self._hop
            abs_end = abs_start + self._win
            if abs_end > self._total:
                break  # not enough samples yet

            # Fetch window from buffer
            rel_start = abs_start - self._buf_offset
            if not self._buf:
                break
            flat = np.concatenate(self._buf, axis=0)
            if rel_start < 0:
                rel_start = 0
            win_arr = flat[rel_start:rel_start + self._win]
            if len(win_arr) < self._win:
                break

            t_end = self.t_base + abs_end / self._fs
            results.append((k, t_end, win_arr))
            self.latest_t_end = t_end
            self._next_k += 1

        # Trim: keep only samples from start of next window onward
        # (window starts at next_k * hop, so we can trim everything before that)
        keep_from_abs = self._next_k * self._hop
        trim = keep_from_abs - self._buf_offset
        if trim > 0 and self._buf:
            flat = np.concatenate(self._buf, axis=0)
            actual_trim = min(trim, len(flat))
            new_flat = flat[actual_trim:]
            self._buf = [new_flat] if len(new_flat) > 0 else []
            self._buf_offset += actual_trim

        return results

    def buffered_samples(self) -> int:
        """Number of samples currently stored in the buffer."""
        return sum(len(b) for b in self._buf)


# ---------------------------------------------------------------------------
# _BearingRecord
# ---------------------------------------------------------------------------

class _BearingRecord:
    """Full bearing info for one node at one window."""
    __slots__ = ("node_id", "window_idx", "t_end", "theta_deg", "confidence",
                 "peak_median_ratio", "gate_ok")

    def __init__(self, node_id: int, window_idx: int, t_end: float,
                 theta_deg: float, confidence: float,
                 peak_median_ratio: float, gate_ok: bool) -> None:
        self.node_id = node_id
        self.window_idx = window_idx
        self.t_end = t_end
        self.theta_deg = theta_deg
        self.confidence = confidence
        self.peak_median_ratio = peak_median_ratio
        self.gate_ok = gate_ok


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
        GCC-PHAT exponent; defaults to cfg[\"phat_beta\"] or 1.0.
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

        # Per-node buffers (all nodes from config that appear in poses)
        self._bufs: dict[int, _NodeBuffer] = {
            nid: _NodeBuffer(nid, self._window_s, self._hop_s, self._fs)
            for nid in poses
        }

        # All bearing records not yet consumed by a FINAL group
        # Keyed by (node_id, window_idx) -> _BearingRecord
        self._pending_records: dict[tuple[int, int], _BearingRecord] = {}

        # Track emitted group keys so we never emit twice
        # Key: (anchor_t_rounded, frozenset(node_ids))
        self._emitted_group_keys: set[tuple[float, frozenset]] = set()

        # Rolling median smoother (last 5 ok fixes)
        self._smoother = RollingMedianXY(5)

        # Config hash
        self._config_hash: str = "n/a"

        # Results queue
        self._results: list[dict] = []

        # Tracking: t_end of first-ever window produced by any node
        self._first_window_t: float | None = None

    def set_config_hash(self, h: str) -> None:
        self._config_hash = h

    def push_audio(self, node_id: int, t_laptop_s: float,
                   frame_int16: np.ndarray) -> None:
        """Add an (n, 2) int16 stereo audio chunk for one node."""
        buf = self._bufs.get(node_id)
        if buf is None:
            return
        buf.push(t_laptop_s, frame_int16)
        self._drain_node(node_id, buf)
        self._try_emit()

    def push_env(self, node_id: int, temp_c: float) -> None:
        """Update the temperature estimate for a node."""
        buf = self._bufs.get(node_id)
        if buf is None:
            return
        buf.temp_c = float(temp_c)

    def _drain_node(self, node_id: int, buf: _NodeBuffer) -> None:
        """Drain complete windows from buf, compute bearings, store records."""
        tau0 = self._tau0.get(node_id, 0.0)
        swap = self._swap.get(node_id, False)

        for k, t_end, win in buf.drain_windows():
            # Track first-ever window time
            if self._first_window_t is None or t_end < self._first_window_t:
                self._first_window_t = t_end

            left = win[:, 0].astype(np.float64)
            right = win[:, 1].astype(np.float64)

            res = estimate_bearing(
                left, right, self._fs, self._d_m, buf.temp_c,
                band_hz=self._band_hz,
                tau0_s=tau0,
                channel_swap=swap,
                beta=self._beta,
            )

            energy_db = band_energy_db(win, self._fs, self._band_hz)
            gate_ok = (float(res["confidence"]) >= self._gate_min_conf and
                       energy_db >= self._gate_min_energy_db)

            key = (node_id, k)
            self._pending_records[key] = _BearingRecord(
                node_id=node_id,
                window_idx=k,
                t_end=float(t_end),
                theta_deg=float(res["theta_deg"]),
                confidence=float(res["confidence"]),
                peak_median_ratio=float(res["peak_median_ratio"]),
                gate_ok=bool(gate_ok),
            )

    def _alive_nodes(self) -> tuple[set[int], float | None]:
        """Return (alive_node_ids, latest_t_end).

        Liveness rules:
        - latest_t = max t_end seen across all nodes (from buf.latest_t_end)
        - A node is alive if latest_t_end >= latest_t - 2*window_s
        - A node never seen is dead once latest_t - first_window_t > 2*window_s
          (i.e. a node that has produced no windows while another node has
          been running for > 2 window lengths)
        """
        # Collect latest t_end per node from bufs (includes emitted windows)
        node_latest: dict[int, float] = {}
        for nid, buf in self._bufs.items():
            if buf.latest_t_end is not None:
                node_latest[nid] = buf.latest_t_end

        if not node_latest:
            # No windows produced by any node yet
            return set(self._bufs.keys()), None

        latest_t = max(node_latest.values())
        threshold = latest_t - 2.0 * self._window_s
        first_t = self._first_window_t  # t_end of first-ever window

        alive: set[int] = set()
        for nid in self._bufs:
            lt = node_latest.get(nid)
            if lt is not None:
                # Has produced at least one window
                if lt >= threshold:
                    alive.add(nid)
                # else: behind by > 2*window_s → dead
            else:
                # Never produced a window
                # Dead if another node has been going for > 2*window_s
                if first_t is None or (latest_t - first_t) <= 2.0 * self._window_s:
                    # Too early to declare dead
                    alive.add(nid)
                # else: first_t was a while ago and this node never responded → dead

        return alive, latest_t

    def _group_pending(self) -> list[list[_BearingRecord]]:
        """Group pending records into time-aligned windows.

        Returns list of groups sorted by anchor t_end.
        Each group has at most one record per node.
        Uses pair_tolerance_s to match windows across nodes.
        """
        all_recs = list(self._pending_records.values())
        if not all_recs:
            return []

        # Sort by (t_end, node_id) for determinism
        all_recs.sort(key=lambda r: (r.t_end, r.node_id))
        used: set[tuple[int, int]] = set()
        groups: list[list[_BearingRecord]] = []

        for anchor_rec in all_recs:
            akey = (anchor_rec.node_id, anchor_rec.window_idx)
            if akey in used:
                continue
            anchor_t = anchor_rec.t_end
            group: list[_BearingRecord] = [anchor_rec]
            used.add(akey)
            nodes_in: set[int] = {anchor_rec.node_id}

            # Find best match for each other node within tolerance
            best: dict[int, _BearingRecord] = {}
            for rec in all_recs:
                key = (rec.node_id, rec.window_idx)
                if key in used:
                    continue
                if rec.node_id in nodes_in:
                    continue
                if abs(rec.t_end - anchor_t) <= self._pair_tol:
                    prev = best.get(rec.node_id)
                    if prev is None or abs(rec.t_end - anchor_t) < abs(prev.t_end - anchor_t):
                        best[rec.node_id] = rec

            for rec in best.values():
                group.append(rec)
                used.add((rec.node_id, rec.window_idx))
                nodes_in.add(rec.node_id)

            group.sort(key=lambda r: r.node_id)
            groups.append(group)

        # Sort groups by anchor t (max t_end in group)
        groups.sort(key=lambda g: max(r.t_end for r in g))
        return groups

    def _try_emit(self) -> None:
        """Check all pending groups; emit FINAL ones."""
        alive_nodes, latest_t = self._alive_nodes()
        if latest_t is None:
            return

        # Collect alive nodes' latest t_end (only those with data)
        alive_latest: list[float] = []
        for nid in alive_nodes:
            buf = self._bufs.get(nid)
            if buf and buf.latest_t_end is not None:
                alive_latest.append(buf.latest_t_end)

        groups = self._group_pending()
        to_remove: set[tuple[int, int]] = set()

        for group in groups:
            anchor_t = max(r.t_end for r in group)
            group_nids = frozenset(r.node_id for r in group)
            # Round anchor_t to nanosecond for key stability
            group_key = (round(anchor_t, 9), group_nids)

            if group_key in self._emitted_group_keys:
                for r in group:
                    to_remove.add((r.node_id, r.window_idx))
                continue

            # Is every alive node represented?
            has_all_alive = alive_nodes <= group_nids

            # Timeout: anchor_t + pair_tol < min(alive nodes' latest t_end)
            timeout_final = bool(alive_latest and
                                 anchor_t + self._pair_tol < min(alive_latest))

            is_final = has_all_alive or timeout_final

            if not is_final:
                continue

            dead_nids = sorted(nid for nid in self._bufs if nid not in alive_nodes)
            self._emit_group(group, dead_nids)
            self._emitted_group_keys.add(group_key)
            for r in group:
                to_remove.add((r.node_id, r.window_idx))

        for key in to_remove:
            self._pending_records.pop(key, None)

    def flush(self) -> None:
        """Emit all pending groups regardless of finality."""
        alive_nodes, _ = self._alive_nodes()
        dead_nids = sorted(nid for nid in self._bufs if nid not in alive_nodes)

        groups = self._group_pending()
        to_remove: set[tuple[int, int]] = set()

        for group in groups:
            anchor_t = max(r.t_end for r in group)
            group_nids = frozenset(r.node_id for r in group)
            group_key = (round(anchor_t, 9), group_nids)

            if group_key in self._emitted_group_keys:
                for r in group:
                    to_remove.add((r.node_id, r.window_idx))
                continue

            self._emit_group(group, dead_nids)
            self._emitted_group_keys.add(group_key)
            for r in group:
                to_remove.add((r.node_id, r.window_idx))

        for key in to_remove:
            self._pending_records.pop(key, None)

    def _emit_group(self, group: list[_BearingRecord],
                    dead_nids: list[int]) -> None:
        """Fuse a group and append result to _results. Emits even if 1 node."""
        anchor_t = max(r.t_end for r in group)

        # Build bearings report and fuse list
        bearings_report: dict[str, dict] = {}
        bearings_for_fuse: list[tuple[int, float, float]] = []

        for rec in group:
            bearings_report[str(rec.node_id)] = {
                "window_idx": int(rec.window_idx),
                "theta_deg": float(rec.theta_deg),
                "confidence": float(rec.confidence),
                "peak_median_ratio": float(rec.peak_median_ratio),
                "gate_ok": bool(rec.gate_ok),
            }
            if rec.gate_ok:
                bearings_for_fuse.append((rec.node_id, float(rec.theta_deg),
                                          float(rec.confidence)))

        # Fuse (gate_min_confidence=0.0 because gating is done above)
        if len(bearings_for_fuse) >= 2:
            try:
                fuse_result = fuse(
                    bearings_for_fuse,
                    self._poses,
                    min_crossing_deg=self._min_cross_deg,
                    gate_min_confidence=0.0,
                )
            except ValueError:
                fuse_result = {
                    "x_m": None, "y_m": None,
                    "crossing_deg": None, "n_lines": 0,
                    "used_node_ids": [], "status": "uncertain",
                    "reason": "fuse_error",
                }
        else:
            fuse_result = {
                "x_m": None, "y_m": None,
                "crossing_deg": None, "n_lines": len(bearings_for_fuse),
                "used_node_ids": [b[0] for b in bearings_for_fuse],
                "status": "uncertain",
                "reason": "too_few_gated_bearings",
            }

        status: str = fuse_result.get("status", "uncertain")
        reason: str = fuse_result.get("reason", "")

        # Dead-node status override:
        # If fuse is not "ok" AND there are dead nodes -> "degraded"
        # A dead node with the rest fusing "ok" stays "ok"
        if status != "ok" and dead_nids:
            status = "degraded"

        # Smoother: update only on ok
        x_raw_m: float | None = (float(fuse_result["x_m"])
                                  if fuse_result.get("x_m") is not None else None)
        y_raw_m: float | None = (float(fuse_result["y_m"])
                                  if fuse_result.get("y_m") is not None else None)

        if status == "ok" and x_raw_m is not None and y_raw_m is not None:
            sm = self._smoother.update({"status": "ok",
                                        "x_m": x_raw_m, "y_m": y_raw_m})
            x_m: float | None = float(sm[0]) if sm else x_raw_m
            y_m: float | None = float(sm[1]) if sm else y_raw_m
        else:
            self._smoother.update({"status": status})
            x_m = None
            y_m = None

        crossing_deg = fuse_result.get("crossing_deg")

        result: dict[str, Any] = {
            "t_laptop_s": float(anchor_t),
            "x_m": x_m,
            "y_m": y_m,
            "x_raw_m": x_raw_m,
            "y_raw_m": y_raw_m,
            "crossing_deg": (float(crossing_deg)
                             if crossing_deg is not None else None),
            "n_nodes": int(len(group)),
            "dead_node_ids": [int(x) for x in dead_nids],
            "status": status,
            "reason": reason,
            "bearings": bearings_report,
            "config_hash": self._config_hash,
            "convention_version": int(self.CONVENTION_VERSION),
        }
        self._results.append(result)

    def poll(self) -> list[dict]:
        """Drain and return all pending results."""
        out = self._results
        self._results = []
        return out


# ---------------------------------------------------------------------------
# replay_trial
# ---------------------------------------------------------------------------

def replay_trial(folder: pathlib.Path | str,
                 cfg_path: pathlib.Path | str | None = None) -> list[dict]:
    """Replay a recorded trial through LivePipeline and write CSVs.

    Config resolution order:
      1. cfg_path argument
      2. folder/site.toml
      3. repo config/site.toml

    If meta.json config_hash differs from the loaded config's hash, a warning
    is printed to stderr (the replay continues).

    Feeds all nodes INTERLEAVED in order of (t_laptop_s, node_id), 512-sample
    chunks. Calls flush() at the end.

    Writes:
        bearings.csv: t_laptop_s, node_id, window_idx, theta_deg, confidence,
                      peak_median_ratio, gate_ok
        positions.csv: t_laptop_s, x_m, y_m, crossing_deg, n_nodes, status

    Returns list of result dicts.
    """
    from kaawayan.config import load_config
    from kaawayan.fusion import poses_from_config

    folder = pathlib.Path(folder)
    meta_path = folder / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    # Resolve config path
    if cfg_path is None:
        candidates = [
            folder / "site.toml",
            pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "site.toml",
        ]
        for c in candidates:
            if c.exists():
                cfg_path = c
                break
    if cfg_path is None:
        raise FileNotFoundError("Could not locate site.toml for replay_trial")

    cfg_path = pathlib.Path(cfg_path)
    cfg, _ = load_config(cfg_path)
    poses = poses_from_config(cfg)
    config_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()

    # Warn if config_hash mismatch
    meta_hash = meta.get("config_hash", "")
    if meta_hash and meta_hash != config_hash:
        print(
            f"WARNING replay_trial: meta.json config_hash={meta_hash!r} differs from "
            f"loaded config hash={config_hash!r}; proceeding anyway.",
            file=sys.stderr,
        )

    pipeline = LivePipeline(cfg, poses)
    pipeline.set_config_hash(config_hash)

    fs: int = 48_000
    nodes_meta: dict = meta.get("nodes", {})
    _CHUNK = 512

    # Feed env temperatures first (all nodes)
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

    # Build interleaved event list: (t_laptop_s, node_id, chunk_array)
    # Enumerate chunks for each node, compute t_laptop_s, then sort.
    events: list[tuple[float, int, np.ndarray]] = []

    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        wav_path = folder / f"node{nid}.wav"
        if not wav_path.exists():
            continue
        node_meta_entry = nodes_meta.get(str(nid), {})
        t0 = float(node_meta_entry.get("t_first_audio_laptop_s", 0.0))

        with wave.open(str(wav_path)) as wf:
            n_frames = wf.getnframes()
            sample_idx = 0
            while sample_idx < n_frames:
                chunk_n = min(_CHUNK, n_frames - sample_idx)
                raw = wf.readframes(chunk_n)
                arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
                t_push = t0 + sample_idx / fs
                events.append((t_push, nid, arr))
                sample_idx += chunk_n

    # Sort interleaved by (t_laptop_s, node_id)
    events.sort(key=lambda e: (e[0], e[1]))

    for t_push, nid, arr in events:
        pipeline.push_audio(nid, t_push, arr)

    pipeline.flush()
    all_results = pipeline.poll()

    # Write bearings.csv (deterministic: sort by t_laptop_s, node_id)
    bearings_path = folder / "bearings.csv"
    bearing_rows: list[dict] = []
    for res in all_results:
        for nid_str, binfo in res.get("bearings", {}).items():
            bearing_rows.append({
                "t_laptop_s": res["t_laptop_s"],
                "node_id": nid_str,
                "window_idx": binfo["window_idx"],
                "theta_deg": binfo["theta_deg"],
                "confidence": binfo["confidence"],
                "peak_median_ratio": binfo["peak_median_ratio"],
                "gate_ok": binfo["gate_ok"],
            })
    bearing_rows.sort(key=lambda r: (r["t_laptop_s"], str(r["node_id"])))
    with open(bearings_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "t_laptop_s", "node_id", "window_idx", "theta_deg",
            "confidence", "peak_median_ratio", "gate_ok",
        ])
        w.writeheader()
        w.writerows(bearing_rows)

    # Write positions.csv
    positions_path = folder / "positions.csv"
    with open(positions_path, "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=[
            "t_laptop_s", "x_m", "y_m", "crossing_deg", "n_nodes", "status"
        ])
        w2.writeheader()
        for res in all_results:
            w2.writerow({
                "t_laptop_s": res["t_laptop_s"],
                "x_m": res["x_m"] if res["x_m"] is not None else "",
                "y_m": res["y_m"] if res["y_m"] is not None else "",
                "crossing_deg": (res["crossing_deg"]
                                 if res["crossing_deg"] is not None else ""),
                "n_nodes": res["n_nodes"],
                "status": res["status"],
            })

    return all_results


# ---------------------------------------------------------------------------
# __main__ — UNTESTED ON HARDWARE
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    """Live runner printing JSON lines from SerialSource. UNTESTED ON HARDWARE.
    Reconnect handling (plan L2) is OUT OF SCOPE.
    """
    import time
    from kaawayan.config import load_config
    from kaawayan.fusion import poses_from_config
    from kaawayan.record import FrameAssembler, SerialSource
    from kaawayan.protocol import EnvPayload

    if len(sys.argv) < 2:
        print("Usage: python -m kaawayan.live <site.toml>", file=sys.stderr)
        sys.exit(1)

    _cfg_path = pathlib.Path(sys.argv[1])
    _cfg, _ = load_config(_cfg_path)
    _poses = poses_from_config(_cfg)
    _config_hash = hashlib.sha256(_cfg_path.read_bytes()).hexdigest()

    _pipeline = LivePipeline(_cfg, _poses)
    _pipeline.set_config_hash(_config_hash)

    _assemblers: dict = {}
    _sources: dict = {}
    for _node in _cfg.get("nodes", []):
        _nid = _node["node_id"]
        _src = SerialSource(_node["usb_serial"])
        _sources[_nid] = _src
        _assemblers[_nid] = FrameAssembler()

    _t_start = time.monotonic()
    try:
        while True:
            for _nid, _src in _sources.items():
                _data = _src.read()
                if _data:
                    for _frame, _ in _assemblers[_nid].feed(_data, time.monotonic() - _t_start):
                        if _frame["type"].name == "AUDIO":
                            _arr = np.frombuffer(_frame["payload"], dtype=np.int16).reshape(-1, 2)
                            _pipeline.push_audio(_nid, time.monotonic() - _t_start, _arr)
                        elif _frame["type"].name == "ENV":
                            _ep = EnvPayload.from_bytes(_frame["payload"])
                            _pipeline.push_env(_nid, _ep.temp_c_x100 / 100.0)
            for _res in _pipeline.poll():
                print(json.dumps(_res, sort_keys=True, allow_nan=False))
                sys.stdout.flush()
    except KeyboardInterrupt:
        _pipeline.flush()
        for _res in _pipeline.poll():
            print(json.dumps(_res, sort_keys=True, allow_nan=False))
