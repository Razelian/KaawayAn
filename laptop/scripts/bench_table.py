"""
bench_table.py <trial_dir>... [--config <site.toml>]

Offline benchmark tool: reads node{id}.wav files and meta.json from each
trial folder, runs bearing estimation across a grid of beta × band, and
prints per-angle and summary tables.

Columns per table cell: median |err|, p95 |err|, signed median error (bias),
mean confidence, % windows with confidence >= gate_min_confidence, n windows.

Summary table: WORST median over angles per cell.

Header prints config SHA-256 and git commit (CONTEXT Rule 7).

Usage:
    python laptop/scripts/bench_table.py trial1/ trial2/ [--config config/site.toml]

If --config is omitted, auto-detects config/site.toml relative to this script.
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
_LAPTOP = _HERE.parent
_REPO = _LAPTOP.parent
if str(_LAPTOP) not in sys.path:
    sys.path.insert(0, str(_LAPTOP))

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
# Collect windows for one trial / one node / one (beta, band) cell
# ---------------------------------------------------------------------------

def _run_cell(
    audio: np.ndarray,
    true_theta: float,
    cfg: dict,
    beta: float,
    band: tuple[int, int],
    gate_min_conf: float,
) -> dict[str, Any]:
    """Return per-window statistics for one (true_theta, beta, band) cell."""
    fs = cfg["sample_rate_hz"] if "sample_rate_hz" in cfg else 48_000
    window_s = cfg["window_s"]
    hop_s = cfg["hop_s"]
    d_m = cfg["mic_spacing_m"]
    temp_c = 20.0  # default if no env.csv; overridden by caller

    errs: list[float] = []
    confs: list[float] = []
    gated: int = 0

    for _start, win in iter_windows(audio, fs, window_s, hop_s):
        left = win[:, 0].astype(np.float64)
        right = win[:, 1].astype(np.float64)
        res = estimate_bearing(
            left, right, fs, d_m, temp_c,
            band_hz=band,
            tau0_s=0.0,
            channel_swap=False,
            beta=beta,
        )
        err = res["theta_deg"] - true_theta
        # Wrap to [-180, 180]
        err = ((err + 180.0) % 360.0) - 180.0
        errs.append(err)
        confs.append(res["confidence"])
        if res["confidence"] >= gate_min_conf:
            gated += 1

    if not errs:
        return {"median_abs_err": float("nan"), "p95_abs_err": float("nan"),
                "signed_median": float("nan"), "mean_conf": float("nan"),
                "pct_gated": float("nan"), "n": 0}

    abs_errs = np.abs(errs)
    return {
        "median_abs_err": float(np.median(abs_errs)),
        "p95_abs_err": float(np.percentile(abs_errs, 95)),
        "signed_median": float(np.median(errs)),
        "mean_conf": float(np.mean(confs)),
        "pct_gated": 100.0 * gated / len(errs),
        "n": len(errs),
    }


# ---------------------------------------------------------------------------
# Load trial
# ---------------------------------------------------------------------------

def _load_trial(trial_dir: pathlib.Path, cfg: dict) -> dict[int, dict]:
    """Return {node_id: {'audio': ndarray, 'true_theta': float, 'temp_c': float}}."""
    meta_path = trial_dir / "meta.json"
    if not meta_path.exists():
        print(f"[WARN] No meta.json in {trial_dir}; skipping")
        return {}

    meta = json.loads(meta_path.read_text())
    true_thetas: dict[str, float] = meta.get("true_theta_deg", {})

    # Load optional env.csv for temperature
    env_path = trial_dir / "env.csv"
    temp_c_default = 20.0
    if env_path.exists():
        import csv
        temps = []
        with open(env_path) as f:
            for row in csv.DictReader(f):
                try:
                    temps.append(float(row["temp_c"]))
                except (KeyError, ValueError):
                    pass
        if temps:
            temp_c_default = float(np.median(temps))

    nodes: dict[int, dict] = {}
    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        wav_path = trial_dir / f"node{nid}.wav"
        if not wav_path.exists():
            continue
        theta_str = str(nid)
        if theta_str not in true_thetas:
            continue
        audio = _load_wav(wav_path)
        nodes[nid] = {
            "audio": audio,
            "true_theta": float(true_thetas[theta_str]),
            "temp_c": temp_c_default,
        }
    return nodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bench table: beta × band sweep")
    parser.add_argument("trial_dirs", nargs="+", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=None,
                        help="Path to site.toml (default: auto-detect)")
    args = parser.parse_args(argv)

    # Resolve config
    cfg_path = args.config
    if cfg_path is None:
        cfg_path = _REPO / "config" / "site.toml"
    cfg_path = cfg_path.resolve()

    cfg, _ = load_config(cfg_path)
    gate_min_conf = cfg.get("gate_min_confidence", 0.0)
    fs = 48_000  # always 48 kHz

    _print_header(cfg_path)

    # Collect all (true_theta, audio, temp_c) pairs across all trials
    all_entries: list[dict] = []
    for trial_dir in args.trial_dirs:
        trial_dir = pathlib.Path(trial_dir).resolve()
        nodes = _load_trial(trial_dir, cfg)
        for nid, data in nodes.items():
            all_entries.append(data)

    if not all_entries:
        print("No data found.")
        return

    # Unique true_theta values
    true_thetas = sorted({e["true_theta"] for e in all_entries})

    # For each (beta, band): per-angle statistics
    # summary_table[beta_idx][band_idx] = worst median over angles
    summary: list[list[float]] = [[float("nan")] * len(BANDS) for _ in BETAS]

    for bi, beta in enumerate(BETAS):
        for bj, band in enumerate(BANDS):
            band_label = f"{band[0]}-{band[1]}"
            print(f"\n--- beta={beta:.2f}  band=[{band_label}] ---")
            print(f"{'true_theta':>12} {'med|err|':>10} {'p95|err|':>10} "
                  f"{'bias':>8} {'mean_conf':>10} {'%gated':>8} {'n':>6}")
            worst_median = 0.0
            for theta in true_thetas:
                entries = [e for e in all_entries if e["true_theta"] == theta]
                # Combine windows across all trials at this theta
                cell_errs: list[float] = []
                cell_confs: list[float] = []
                cell_gated = 0
                for entry in entries:
                    audio = entry["audio"]
                    temp_c = entry["temp_c"]
                    for _start, win in iter_windows(audio, fs, cfg["window_s"], cfg["hop_s"]):
                        left = win[:, 0].astype(np.float64)
                        right = win[:, 1].astype(np.float64)
                        res = estimate_bearing(
                            left, right, fs, cfg["mic_spacing_m"], temp_c,
                            band_hz=band, tau0_s=0.0, channel_swap=False, beta=beta,
                        )
                        err = res["theta_deg"] - theta
                        err = ((err + 180.0) % 360.0) - 180.0
                        cell_errs.append(err)
                        cell_confs.append(res["confidence"])
                        if res["confidence"] >= gate_min_conf:
                            cell_gated += 1

                if not cell_errs:
                    continue
                abs_errs = np.abs(cell_errs)
                med = float(np.median(abs_errs))
                p95 = float(np.percentile(abs_errs, 95))
                bias = float(np.median(cell_errs))
                mc = float(np.mean(cell_confs))
                pct_g = 100.0 * cell_gated / len(cell_errs)
                n = len(cell_errs)
                print(f"{theta:>12.1f} {med:>10.3f} {p95:>10.3f} "
                      f"{bias:>8.3f} {mc:>10.4f} {pct_g:>8.1f} {n:>6d}")
                if med > worst_median:
                    worst_median = med
            summary[bi][bj] = worst_median

    # Summary table
    print("\n\n=== SUMMARY: WORST median |err| (deg) over angles ===")
    print(f"{'beta\\band':>10}", end="")
    for bl in BAND_LABELS:
        print(f"  {bl:>12}", end="")
    print()
    for bi, beta in enumerate(BETAS):
        print(f"{beta:>10.2f}", end="")
        for bj in range(len(BANDS)):
            val = summary[bi][bj]
            print(f"  {val:>12.3f}", end="")
        print()
    print()


if __name__ == "__main__":
    main()
