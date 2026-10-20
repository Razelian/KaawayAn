"""
kaawayan.evaluate — Position error metrics for a recorded trial.

Functions
---------
position_errors(est_xy_list, truth_xy) -> np.ndarray
    Returns array of Euclidean errors in metres.

summarize_errors(errors) -> dict(median_m, p95_m, max_m, n)
    Linear percentile summary (np.percentile).

evaluate_trial(folder) -> dict
    Reads positions.csv and meta.json; filters rows where status == "ok";
    computes errors against truth_xy_m from meta.json;
    echoes config_hash and git_commit.

CLI: python -m kaawayan.evaluate <folder> prints JSON.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# position_errors
# ---------------------------------------------------------------------------

def position_errors(
    est_xy_list: list[tuple[float, float]],
    truth_xy: tuple[float, float],
) -> np.ndarray:
    """Return array of Euclidean errors in metres.

    Parameters
    ----------
    est_xy_list:
        List of (x_m, y_m) estimated positions.
    truth_xy:
        (x_m, y_m) ground-truth position.

    Returns
    -------
    np.ndarray of shape (n,) with non-negative errors in metres.
    """
    if not est_xy_list:
        return np.array([], dtype=np.float64)
    tx, ty = truth_xy
    arr = np.array(est_xy_list, dtype=np.float64)  # shape (n, 2)
    errors = np.sqrt((arr[:, 0] - tx) ** 2 + (arr[:, 1] - ty) ** 2)
    return errors


# ---------------------------------------------------------------------------
# summarize_errors
# ---------------------------------------------------------------------------

def summarize_errors(errors: np.ndarray) -> dict[str, Any]:
    """Summarise position errors.

    Parameters
    ----------
    errors:
        1-D array of Euclidean errors in metres.

    Returns
    -------
    dict with keys:
        median_m: float
        p95_m: float
        max_m: float
        n: int
    """
    n = len(errors)
    if n == 0:
        return {"median_m": float("nan"), "p95_m": float("nan"),
                "max_m": float("nan"), "n": 0}
    return {
        "median_m": float(np.percentile(errors, 50)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "n": n,
    }


# ---------------------------------------------------------------------------
# evaluate_trial
# ---------------------------------------------------------------------------

def evaluate_trial(folder: pathlib.Path | str) -> dict[str, Any]:
    """Evaluate position errors for a trial folder.

    Reads:
      positions.csv  — rows with t_laptop_s, x_m, y_m, crossing_deg, n_nodes, status
      meta.json      — for true_xy_m, config_hash, git_commit

    Parameters
    ----------
    folder:
        Trial directory path.

    Returns
    -------
    dict with keys:
        median_m, p95_m, max_m, n: summary statistics.
        ok_fraction: fraction of rows with status == "ok".
        config_hash, git_commit: echoed from meta.json.
        truth_xy_m: (x_m, y_m) from meta.json.
    """
    import csv

    folder = pathlib.Path(folder)
    meta_path = folder / "meta.json"
    positions_path = folder / "positions.csv"

    # Load meta
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    truth_xy_raw = meta.get("true_xy_m", None)
    if truth_xy_raw is not None:
        truth_xy: tuple[float, float] | None = (
            float(truth_xy_raw[0]), float(truth_xy_raw[1])
        )
    else:
        truth_xy = None

    config_hash = meta.get("config_hash", "n/a")
    git_commit = meta.get("git_commit", "n/a")

    # Load positions.csv
    ok_rows: list[tuple[float, float]] = []
    total_rows = 0

    if positions_path.exists():
        with open(positions_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                if row.get("status", "") != "ok":
                    continue
                try:
                    x_m = row.get("x_m", "")
                    y_m = row.get("y_m", "")
                    if x_m == "" or y_m == "":
                        continue
                    ok_rows.append((float(x_m), float(y_m)))
                except (ValueError, KeyError):
                    pass

    ok_fraction = len(ok_rows) / total_rows if total_rows > 0 else 0.0

    if truth_xy is not None and ok_rows:
        errors = position_errors(ok_rows, truth_xy)
        summary = summarize_errors(errors)
    else:
        summary = summarize_errors(np.array([]))

    result: dict[str, Any] = {
        **summary,
        "ok_fraction": ok_fraction,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "truth_xy_m": list(truth_xy) if truth_xy else None,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trial position errors")
    parser.add_argument("folder", type=pathlib.Path)
    args = parser.parse_args(argv)
    result = evaluate_trial(args.folder)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
