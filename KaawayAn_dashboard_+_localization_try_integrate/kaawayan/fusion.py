"""
kaawayan.fusion — Multi-node bearing fusion to a 2-D position.

Convention (CONTEXT §5, convention_version = 1):
  World frame: right-handed 2D, metres, CCW positive from +x.
  Node heading psi: direction of plate normal (out toward area).
  Bearing theta: source angle relative to node front, CCW positive.
  World azimuth of source = psi + theta.
  Fusion line: p + t*(cos(psi+theta), sin(psi+theta)), t<0 means behind node.

Algorithm (CONTEXT §8):
  Confidence-weighted least squares of perpendicular distances to lines.
  A = sum_i w_i (I - u_i u_i^T)
  b = sum_i w_i (I - u_i u_i^T) p_i
  x = A^-1 b
  w_i = max(confidence_i, 1e-6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# NodePose frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodePose:
    """Surveyed position and heading for one node."""
    node_id: int
    x_m: float
    y_m: float
    heading_deg: float


# ---------------------------------------------------------------------------
# poses_from_config
# ---------------------------------------------------------------------------

def poses_from_config(cfg: dict) -> dict[int, NodePose]:
    """Build a node_id -> NodePose mapping from a parsed config dict.

    Parameters
    ----------
    cfg:
        Dict as returned by config.load_config() (the first element of the tuple).

    Returns
    -------
    dict mapping node_id (int) to NodePose.
    """
    poses: dict[int, NodePose] = {}
    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        poses[nid] = NodePose(
            node_id=nid,
            x_m=float(node["x_m"]),
            y_m=float(node["y_m"]),
            heading_deg=float(node["heading_deg"]),
        )
    return poses


# ---------------------------------------------------------------------------
# bearing_line
# ---------------------------------------------------------------------------

def bearing_line(pose: NodePose, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (p, u) for a bearing line.

    Parameters
    ----------
    pose:
        NodePose with node position and heading.
    theta_deg:
        Measured bearing angle in degrees (relative to node front, CCW positive).

    Returns
    -------
    p: ndarray shape (2,) — node position [x, y] in metres.
    u: ndarray shape (2,) — unit direction vector at azimuth heading_deg + theta_deg.
    """
    azimuth_deg = pose.heading_deg + theta_deg
    azimuth_rad = math.radians(azimuth_deg)
    u = np.array([math.cos(azimuth_rad), math.sin(azimuth_rad)], dtype=np.float64)
    p = np.array([pose.x_m, pose.y_m], dtype=np.float64)
    return p, u


# ---------------------------------------------------------------------------
# _wrap180: wrap an angle to [-180, 180)
# ---------------------------------------------------------------------------

def _wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------

def fuse(
    bearings: list[tuple[int, float, float]],
    poses: dict[int, NodePose],
    min_crossing_deg: float = 30.0,
    gate_min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Fuse multiple bearing estimates to a 2-D position.

    Parameters
    ----------
    bearings:
        List of (node_id, theta_deg, confidence) tuples.
    poses:
        Mapping from node_id to NodePose.
    min_crossing_deg:
        Minimum largest pairwise acute angle in degrees; below this -> uncertain.
    gate_min_confidence:
        Drop bearings with confidence strictly below this threshold.

    Returns
    -------
    dict with keys:
        x_m, y_m: float or None — position (None unless status == "ok").
        crossing_deg: float — largest pairwise acute angle between kept lines.
        n_lines: int — number of bearing lines used.
        used_node_ids: list[int] — sorted node_ids of used lines.
        status: "ok" | "uncertain"
        reason: str — human-readable explanation when uncertain.

    Raises
    ------
    ValueError:
        If duplicate node_ids appear in bearings, or if a node_id is not in poses.
    """
    # --- validate node_ids ---------------------------------------------------
    seen_ids: set[int] = set()
    for node_id, theta_deg, confidence in bearings:
        if node_id in seen_ids:
            raise ValueError(f"duplicate node_id {node_id} in bearings")
        seen_ids.add(node_id)
        if node_id not in poses:
            raise ValueError(f"node_id {node_id} not found in poses")

    # --- gate: drop low-confidence bearings ----------------------------------
    kept = [
        (node_id, theta_deg, confidence)
        for node_id, theta_deg, confidence in bearings
        if confidence >= gate_min_confidence
    ]

    # --- build lines ---------------------------------------------------------
    lines: list[tuple[np.ndarray, np.ndarray, float]] = []  # (p, u, w)
    for node_id, theta_deg, confidence in kept:
        pose = poses[node_id]
        p, u = bearing_line(pose, theta_deg)
        w = max(confidence, 1e-6)
        lines.append((p, u, w))

    used_node_ids = sorted(node_id for node_id, _, _ in kept)

    def _uncertain(reason: str, crossing_deg: float | None = None) -> dict:
        return {
            "x_m": None,
            "y_m": None,
            "crossing_deg": crossing_deg,
            "n_lines": len(lines),
            "used_node_ids": used_node_ids,
            "status": "uncertain",
            "reason": reason,
        }

    # --- too few lines -------------------------------------------------------
    if len(lines) < 2:
        return _uncertain("too_few_lines")  # crossing_deg=None: not computable

    # --- crossing angle: largest pairwise acute angle ------------------------
    max_angle = 0.0
    ps = [line[0] for line in lines]
    us = [line[1] for line in lines]
    ws_arr = [line[2] for line in lines]

    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            dot = float(np.clip(np.abs(np.dot(us[i], us[j])), 0.0, 1.0))
            angle = math.degrees(math.acos(dot))
            if angle > max_angle:
                max_angle = angle

    crossing_deg = max_angle

    if crossing_deg < min_crossing_deg:
        return _uncertain("shallow_crossing", crossing_deg)

    # --- weighted least squares ----------------------------------------------
    # A = sum_i w_i (I - u_i u_i^T)
    # b = sum_i w_i (I - u_i u_i^T) p_i
    I2 = np.eye(2, dtype=np.float64)
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)

    for p_i, u_i, w_i in zip(ps, us, ws_arr):
        M = I2 - np.outer(u_i, u_i)
        A += w_i * M
        b += w_i * M @ p_i

    # check singularity
    cond = np.linalg.cond(A)
    if cond > 1e12 or not np.all(np.isfinite(A)):
        return _uncertain("parallel", crossing_deg)

    try:
        x_est = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return _uncertain("parallel", crossing_deg)

    if not np.all(np.isfinite(x_est)):
        return _uncertain("parallel", crossing_deg)

    # --- behind-node check: any t_i = u_i.(x - p_i) < 0 --------------------
    for p_i, u_i, _ in zip(ps, us, ws_arr):
        t_i = float(np.dot(u_i, x_est - p_i))
        if t_i < 0.0:
            return _uncertain("behind_node", crossing_deg)

    return {
        "x_m": float(x_est[0]),
        "y_m": float(x_est[1]),
        "crossing_deg": crossing_deg,
        "n_lines": len(lines),
        "used_node_ids": used_node_ids,
        "status": "ok",
        "reason": "",
    }


# ---------------------------------------------------------------------------
# pair_windows (additive — added in slice 6b)
# ---------------------------------------------------------------------------

def pair_windows(
    records: list[tuple[int, float, float, float]],
    tolerance_s: float,
) -> list[list[tuple[int, float, float, float]]]:
    """Group bearing records from multiple nodes into time-aligned windows.

    Parameters
    ----------
    records:
        List of (node_id, t_laptop_s, theta_deg, confidence) tuples.
    tolerance_s:
        Maximum |t - t_anchor| for a record to join a group.

    Returns
    -------
    List of groups; each group is a list of records sorted by node_id.
    Groups are produced in chronological order (anchor time ascending).
    Each group has at most one record per node.
    Leftover records (not paired) each become their own single-record group.
    """
    # Sort by (t_laptop_s, node_id) for deterministic behaviour
    sorted_records = sorted(records, key=lambda r: (r[1], r[0]))
    used = [False] * len(sorted_records)

    groups: list[list[tuple[int, float, float, float]]] = []

    for anchor_idx, anchor in enumerate(sorted_records):
        if used[anchor_idx]:
            continue
        # This record becomes the anchor
        anchor_node_id = anchor[0]
        anchor_t = anchor[1]
        group: list[tuple[int, float, float, float]] = [anchor]
        used[anchor_idx] = True
        # Nodes already in group
        nodes_in_group: set[int] = {anchor_node_id}

        # For each other node not yet in the group, pick the closest unused record
        # within tolerance. We do one pass over all remaining nodes.
        # Collect candidates: records of other nodes within tolerance
        candidate_by_node: dict[int, tuple[int, float]] = {}  # node_id -> (idx, |dt|)
        for idx, rec in enumerate(sorted_records):
            if used[idx]:
                continue
            nid = rec[0]
            dt = abs(rec[1] - anchor_t)
            if dt <= tolerance_s and nid not in nodes_in_group:
                if nid not in candidate_by_node or dt < candidate_by_node[nid][1]:
                    candidate_by_node[nid] = (idx, dt)

        for nid, (idx, _) in candidate_by_node.items():
            group.append(sorted_records[idx])
            used[idx] = True
            nodes_in_group.add(nid)

        group.sort(key=lambda r: r[0])
        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# RollingMedianXY (additive — added in slice 6b)
# ---------------------------------------------------------------------------

class RollingMedianXY:
    """Rolling median of the last *n* accepted position fixes.

    Only results with status == "ok" are added.
    Returns per-axis median of the last n ok fixes.
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._xs: list[float] = []
        self._ys: list[float] = []

    def update(self, fuse_result: dict) -> tuple[float, float] | None:
        """Add a fuse result (if ok) and return the rolling median (x, y) or None."""
        if fuse_result.get("status") != "ok":
            return None if not self._xs else (
                float(np.median(self._xs[-self._n:])),
                float(np.median(self._ys[-self._n:])),
            )
        self._xs.append(fuse_result["x_m"])
        self._ys.append(fuse_result["y_m"])
        # Trim to last n
        if len(self._xs) > self._n:
            self._xs = self._xs[-self._n:]
            self._ys = self._ys[-self._n:]
        return (
            float(np.median(self._xs)),
            float(np.median(self._ys)),
        )

    def reset(self) -> None:
        """Clear the rolling buffer."""
        self._xs = []
        self._ys = []
