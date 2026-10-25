"""
bench_table.py <trial_dir>... [--config <site.toml>] [--raw]

Offline benchmark tool: reads node{id}.wav files and meta.json from each
trial folder, runs bearing estimation across a grid of beta × band, and
prints per-angle and summary tables.

Columns per table cell: median |err|, p95 |err|, signed median error (bias),
mean confidence, % windows with confidence >= gate_min_confidence, n windows.

Summary table: WORST median over angles per cell.

Header prints config SHA-256 and git commit (CONTEXT Rule 7).

Usage:
    python -m kaawayan.scripts.bench_table trial1/ trial2/ [--config config/site.toml]

If --config is omitted, auto-detects config/site.toml relative to this script.
--raw skips applying tau0_s and channel_swap from config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import wave
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

from kaawayan.bearing import estimate_bearing, speed_of_sound
from kaawayan.config import load_config
from kaawayan.windows import iter_windows, band_energy_db

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BANDS = [
    (300, 1500),
    (300, 1700),
    (300, 4000),
    (300, 6000),
]
BAND_LABELS = [f"{lo}-{hi}" for lo, hi in BANDS]

# ---------------------------------------------------------------------------
# Rule 7 header
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_REPO, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def _config_sha256(cfg_path: pathlib.Path) -> str:
    try:
        raw = cfg_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()
    except Exception as exc:
        return f"ERROR({exc})"


def _print_header(cfg_path: pathlib.Path) -> None:
    sha = _config_sha256(cfg_path)
    git = _git_commit()
    print(f"config SHA-256 : {sha}")
    print(f"git commit     : {git}")
    print()


# ---------------------------------------------------------------------------
# Load a WAV file
# ---------------------------------------------------------------------------

def _load_wav(path: pathlib.Path) -> np.ndarray:
    """Return int16 stereo array (n, 2)."""
    with wave.open(str(path)) as wf:
        raw = wf.readframes(wf.getnframes())
        n = len(raw) // (wf.getsampwidth() * wf.getnchannels())
        arr = np.frombuffer(raw, dtype=np.int16).reshape(n, wf.getnchannels())
    return arr


# ---------------------------------------------------------------------------
# compute_table — pure, testable function
# ---------------------------------------------------------------------------

def compute_table(
    trial_dirs: list[pathlib.Path],
    cfg: dict,
    raw: bool = False,
) -> dict:
    """Compute per-(beta,band) per-angle statistics over all trial dirs.

    Returns
    -------
    dict with keys:
      cells[(beta, band)][true_angle] -> dict(median_abs_err, p95_abs_err,
          signed_median, mean_conf, pct_gated, n)
      summary[(beta, band)] -> worst median_abs_err over true angles
    """
    fs = 48_000
    gate_min_conf = cfg.get("gate_min_confidence", 0.0)
    window_s = cfg["window_s"]
    hop_s = cfg["hop_s"]
    d_m = cfg["mic_spacing_m"]

    # Build node calibration maps
    node_tau0: dict[int, float] = {}
    node_swap: dict[int, bool] = {}
    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        node_tau0[nid] = node.get("tau0_s", 0.0)
        node_swap[nid] = node.get("channel_swap", False)

    # Collect per-angle entries across trials
    # entries: list of dict(true_theta, audio, temp_c, tau0_s, channel_swap)
    all_entries: list[dict] = []

    for trial_dir in trial_dirs:
        trial_dir = pathlib.Path(trial_dir).resolve()
        meta_path = trial_dir / "meta.json"
        if not meta_path.exists():
            print(f"[SKIP] No meta.json in {trial_dir}")
            continue

        meta = json.loads(meta_path.read_text())

        # true_theta_deg: may be dict {node_id: angle} or flat
        raw_thetas = meta.get("true_theta_deg", {})
        if not raw_thetas:
            print(f"[SKIP] No true_theta_deg in {trial_dir}/meta.json")
            continue

        # Read temperature from node{id}_env.csv (new format) or env.csv (legacy)
        temp_by_node: dict[int, float] = {}
        for node in cfg.get("nodes", []):
            nid = node["node_id"]
            # Try per-node env first
            env_path = trial_dir / f"node{nid}_env.csv"
            if not env_path.exists():
                env_path = trial_dir / "env.csv"
            temps: list[float] = []
            if env_path.exists():
                import csv as _csv
                with open(env_path) as f:
                    for row in _csv.DictReader(f):
                        try:
                            temps.append(float(row["temp_c"]))
                        except (KeyError, ValueError):
                            pass
            temp_by_node[nid] = float(np.median(temps)) if temps else 20.0

        for node in cfg.get("nodes", []):
            nid = node["node_id"]
            wav_path = trial_dir / f"node{nid}.wav"
            if not wav_path.exists():
                print(f"[SKIP] No node{nid}.wav in {trial_dir}")
                continue

            theta_str = str(nid)
            if theta_str not in raw_thetas:
                print(f"[SKIP] No true_theta_deg for node {nid} in {trial_dir}/meta.json")
                continue

            audio = _load_wav(wav_path)
            tau0 = 0.0 if raw else node_tau0.get(nid, 0.0)
            swap = False if raw else node_swap.get(nid, False)
            all_entries.append({
                "true_theta": float(raw_thetas[theta_str]),
                "audio": audio,
                "temp_c": temp_by_node[nid],
                "tau0_s": tau0,
                "channel_swap": swap,
                "node_id": nid,
            })

    if not all_entries:
        return {"cells": {}, "summary": {}}

    true_angles = sorted({e["true_theta"] for e in all_entries})

    cells: dict = {}
    summary: dict = {}

    for beta in BETAS:
        for band in BANDS:
            key = (beta, band)
            cells[key] = {}

            worst_median = 0.0
            for theta in true_angles:
                entries = [e for e in all_entries if e["true_theta"] == theta]
                errs: list[float] = []
                confs: list[float] = []
                gated: int = 0

                for entry in entries:
                    audio = entry["audio"]
                    temp_c = entry["temp_c"]
                    tau0 = entry["tau0_s"]
                    swap = entry["channel_swap"]
                    for _start, win in iter_windows(audio, fs, window_s, hop_s):
                        left = win[:, 0].astype(np.float64)
                        right = win[:, 1].astype(np.float64)
                        res = estimate_bearing(
                            left, right, fs, d_m, temp_c,
                            band_hz=band, tau0_s=tau0,
                            channel_swap=swap, beta=beta,
                        )
                        err = res["theta_deg"] - theta
                        err = ((err + 180.0) % 360.0) - 180.0
                        errs.append(err)
                        confs.append(res["confidence"])
                        if res["confidence"] >= gate_min_conf:
                            gated += 1

                if not errs:
                    cells[key][theta] = {
                        "median_abs_err": float("nan"),
                        "p95_abs_err": float("nan"),
                        "signed_median": float("nan"),
                        "mean_conf": float("nan"),
                        "pct_gated": float("nan"),
                        "n": 0,
                    }
                    continue

                abs_errs = np.abs(errs)
                med = float(np.median(abs_errs))
                p95 = float(np.percentile(abs_errs, 95))
                bias = float(np.median(errs))
                mc = float(np.mean(confs))
                pct_g = 100.0 * gated / len(errs)
                n = len(errs)
                cells[key][theta] = {
                    "median_abs_err": med,
                    "p95_abs_err": p95,
                    "signed_median": bias,
                    "mean_conf": mc,
                    "pct_gated": pct_g,
                    "n": n,
                }
                if med > worst_median:
                    worst_median = med

            summary[key] = worst_median

    return {"cells": cells, "summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bench table: beta x band sweep")
    parser.add_argument("trial_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="Path to site.toml (default: auto-detect)")
    parser.add_argument("--raw", action="store_true",
                        help="Skip tau0_s and channel_swap from config")
    args = parser.parse_args(argv)

    # Resolve config
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = _REPO / "config" / "site.toml"
    cfg_path = cfg_path.resolve()

    cfg, _ = load_config(cfg_path)

    _print_header(cfg_path)

    result = compute_table(
        [pathlib.Path(d).resolve() for d in args.trial_dirs],
        cfg,
        raw=args.raw,
    )

    cells = result["cells"]
    summary = result["summary"]

    if not cells:
        print("No data found.")
        return

    true_angles = sorted({theta for angle_dict in cells.values() for theta in angle_dict})

    for beta in BETAS:
        for band in BANDS:
            key = (beta, band)
            band_label = f"{band[0]}-{band[1]}"
            print(f"\n--- beta={beta:.2f}  band=[{band_label}] ---")
            print(f"{'true_theta':>12} {'med|err|':>10} {'p95|err|':>10} "
                  f"{'bias':>8} {'mean_conf':>10} {'%gated':>8} {'n':>6}")
            angle_dict = cells.get(key, {})
            for theta in true_angles:
                st = angle_dict.get(theta)
                if st is None or st["n"] == 0:
                    continue
                print(f"{theta:>12.1f} {st['median_abs_err']:>10.3f} {st['p95_abs_err']:>10.3f} "
                      f"{st['signed_median']:>8.3f} {st['mean_conf']:>10.4f} "
                      f"{st['pct_gated']:>8.1f} {st['n']:>6d}")

    # Summary table
    print("\n\n=== SUMMARY: WORST median |err| (deg) over angles ===")
    header = f"{'beta\\\\band':>10}"
    for bl in BAND_LABELS:
        header += f"  {bl:>12}"
    print(header)
    for beta in BETAS:
        row = f"{beta:>10.2f}"
        for band in BANDS:
            val = summary.get((beta, band), float("nan"))
            row += f"  {val:>12.3f}"
        print(row)
    print()


if __name__ == "__main__":
    main()
