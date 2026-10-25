"""
band_diagnostic.py — offline diagnostic tables for bearing.py / sim.py.

Run from any directory; this script adds the repo's laptop folder to sys.path.

Table (a): Bearing error vs frequency for pure tones at 500, 1000, 1500, 1700,
           2000, 3000, 4000, 6000 Hz; theta = 20 and 40 deg.
Table (b): Bearing error for white noise at band [300, 1500] vs [300, 6000],
           at 0 dB and 10 dB SNR, with and without the echo (5 ms, 0.5).
Table (c): Harmonic source (f0=180 Hz, 12 harmonics, amplitude 1/k, random phases,
           1 s @ 48 kHz, theta=+30, d=0.10 m) — beta x band x snr sweep,
           20 seeds.  Columns: median |err|, p95 |err|, mean confidence.
Table (d): White noise with echo (0.005 s, 0.5) — same grid as (c).

Not a pytest test — run directly: python laptop/scripts/band_diagnostic.py

Rule 7: every script that produces results must write the config hash and git
commit with them.
"""

from __future__ import annotations

import sys
import os
import hashlib
import subprocess

# Add the repo's laptop folder to sys.path so kaawayan is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LAPTOP = os.path.join(_HERE, "..")
_REPO = os.path.join(_LAPTOP, "..")
if _LAPTOP not in sys.path:
    sys.path.insert(0, os.path.normpath(_LAPTOP))

import numpy as np
from kaawayan.bearing import speed_of_sound, estimate_bearing
from kaawayan.sim import make_stereo

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
FS = 48_000
D = 0.10      # metres
T_C = 20.0
C = speed_of_sound(T_C)
N = FS        # 1 s
ECHO = (0.005, 0.5)

SITE_TOML = os.path.join(_REPO, "config", "site.toml")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _pure_tone(freq_hz: float, n: int, fs: float) -> np.ndarray:
    t = np.arange(n) / fs
    return np.sin(2.0 * np.pi * freq_hz * t)


def _harmonic_source(f0: float, n_harmonics: int, n: int, fs: float, rng) -> np.ndarray:
    """Sum of harmonics at f0*k with amplitude 1/k and random phases."""
    t = np.arange(n) / fs
    sig = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        sig += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + phase)
    return sig


def _run_bearing(
    sig: np.ndarray,
    theta_deg: float,
    band: tuple[float, float],
    snr_db: float,
    echo: tuple[float, float] | None,
    seed: int = 0,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Return (theta_recovered, confidence)."""
    rng = _rng(seed)
    left, right = make_stereo(sig, FS, theta_deg, D, C, snr_db, rng, echo=echo)
    result = estimate_bearing(left, right, FS, D, T_C, band, beta=beta)
    return result["theta_deg"], result["confidence"]


# ---------------------------------------------------------------------------
# Rule 7 header: config SHA-256 and git commit
# ---------------------------------------------------------------------------
def _print_header() -> None:
    # Config hash
    try:
        raw = open(SITE_TOML, "rb").read()
        sha256 = hashlib.sha256(raw).hexdigest()
    except Exception as e:
        sha256 = f"ERROR({e})"

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_REPO, timeout=5
        )
        git_commit = result.stdout.strip() if result.returncode == 0 else "n/a"
    except Exception:
        git_commit = "n/a"

    print(f"config/site.toml SHA-256 : {sha256}")
    print(f"git commit               : {git_commit}")
    print()


# ---------------------------------------------------------------------------
# Table (a): error vs frequency for pure tones
# ---------------------------------------------------------------------------
def table_a() -> None:
    print("=" * 72)
    print("Table (a) — Bearing error (deg) vs pure-tone frequency")
    print("           band_hz = (300, 6000), SNR = 20 dB, d = 0.10 m, T = 20 °C")
    print("=" * 72)
    freqs = [500, 1000, 1500, 1700, 2000, 3000, 4000, 6000]
    thetas = [20, 40]
    band = (300, 6000)
    snr = 20.0

    # Header
    header = f"{'freq (Hz)':>10}" + "".join(f"  theta={t:+d} err" for t in thetas)
    print(header)
    print("-" * len(header))

    for freq in freqs:
        sig = _pure_tone(freq, N, FS)
        row = f"{freq:>10d}"
        for theta in thetas:
            try:
                recovered, _ = _run_bearing(sig, theta, band, snr, echo=None, seed=freq)
                err = recovered - theta
                row += f"  {err:+8.3f} deg"
            except Exception as e:
                row += f"  ERROR({e})"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Table (b): white noise, band and SNR sweep, with and without echo
# ---------------------------------------------------------------------------
def table_b() -> None:
    print("=" * 72)
    print("Table (b) — Bearing error (deg) for white noise")
    print("           theta = 30 deg, d = 0.10 m, T = 20 °C")
    print("=" * 72)

    bands = [(300, 1500), (300, 6000)]
    snrs = [0, 10]
    echoes: list[tuple[str, tuple | None]] = [("no echo", None), ("echo(5ms,0.5)", ECHO)]
    theta_true = 30.0

    # Column widths
    band_labels = [f"[{lo},{hi}]" for lo, hi in bands]
    col_w = 14

    header = f"{'SNR':>6}  {'echo':>14}" + "".join(f"  {b:>{col_w}}" for b in band_labels)
    print(header)
    print("-" * len(header))

    for snr in snrs:
        for echo_label, echo_val in echoes:
            row = f"{snr:>3} dB  {echo_label:>14}"
            for band in bands:
                sig = _rng(snr).standard_normal(N)
                try:
                    recovered, _ = _run_bearing(sig, theta_true, band, snr, echo_val,
                                             seed=snr * 100 + (1 if echo_val else 0))
                    err = recovered - theta_true
                    row += f"  {err:>+{col_w}.3f} deg"
                except Exception as e:
                    row += f"  ERROR({e})"
            print(row)

    print()


# ---------------------------------------------------------------------------
# Tables (c) and (d): harmonic-source and white-noise+echo sweeps
# ---------------------------------------------------------------------------

# Grid definition
_BETAS = [0, 0.25, 0.5, 0.75, 1]
_BANDS = [(300, 1500), (300, 1700), (300, 4000), (300, 6000)]
_SNRS = [10, 0]
_N_SEEDS = 20
_THETA_TRUE = 30.0

# Harmonic source parameters
_F0 = 180.0
_N_HARM = 12


def _sweep(source_fn, echo=None) -> dict:
    """
    Run the full beta x band x snr grid.

    source_fn(rng) -> np.ndarray of length N.
    Returns a dict keyed by (beta, band, snr) -> (median_err, p95_err, mean_conf).
    """
    results = {}
    for beta in _BETAS:
        for band in _BANDS:
            for snr in _SNRS:
                errors = []
                confs = []
                for seed in range(_N_SEEDS):
                    rng = _rng(seed + 2000)
                    sig = source_fn(rng)
                    try:
                        recovered, conf = _run_bearing(
                            sig, _THETA_TRUE, band, snr, echo,
                            seed=seed + 2000, beta=beta
                        )
                        errors.append(abs(recovered - _THETA_TRUE))
                        confs.append(conf)
                    except Exception:
                        pass
                if errors:
                    results[(beta, band, snr)] = (
                        float(np.median(errors)),
                        float(np.percentile(errors, 95)),
                        float(np.mean(confs)),
                    )
                else:
                    results[(beta, band, snr)] = (float("nan"), float("nan"), float("nan"))
    return results


def _print_sweep_table(title: str, subtitle: str, results: dict) -> None:
    print("=" * 90)
    print(title)
    print(subtitle)
    print("=" * 90)

    # Column: band x snr combinations
    col_labels = [f"[{b[0]},{b[1]}] {s}dB" for b in _BANDS for s in _SNRS]
    col_w = 28  # enough for "med|err| p95|err| mean_conf"

    # Header row
    beta_w = 6
    header = f"{'beta':>{beta_w}}"
    for label in col_labels:
        header += f"  {label:^{col_w}}"
    print(header)

    # Sub-header
    sub = " " * beta_w
    for _ in col_labels:
        sub += f"  {'med|e|':>7}  {'p95|e|':>7}  {'conf':>6}   "
    print(sub)
    print("-" * len(header))

    for beta in _BETAS:
        row = f"{beta:>{beta_w}.2f}"
        for band in _BANDS:
            for snr in _SNRS:
                med, p95, conf = results.get((beta, band, snr), (float("nan"),) * 3)
                row += f"  {med:>7.2f}  {p95:>7.2f}  {conf:>6.3f}   "
        print(row)

    print()

    # Also print best cell
    best_key = min(
        ((k, v) for k, v in results.items()),
        key=lambda x: x[1][0]
    )
    bk, bv = best_key
    beta_b, band_b, snr_b = bk
    print(
        f"  Best cell: beta={beta_b}, band=[{band_b[0]},{band_b[1]}], "
        f"snr={snr_b}dB -> median|err|={bv[0]:.3f} deg, "
        f"p95={bv[1]:.3f} deg, conf={bv[2]:.3f}"
    )
    print()


def table_c() -> None:
    print("Table (c) — Harmonic source")
    print(f"  f0={_F0} Hz, {_N_HARM} harmonics amp 1/k, random phases, "
          f"1 s @ {FS} Hz, theta=+{_THETA_TRUE} deg, d={D} m")
    print(f"  {_N_SEEDS} seeds, no echo")

    def harmonic_source(rng):
        return _harmonic_source(_F0, _N_HARM, N, FS, rng)

    results = _sweep(harmonic_source, echo=None)
    _print_sweep_table(
        "Table (c) — Harmonic source: beta x band x SNR",
        f"  f0={_F0}Hz, {_N_HARM} harmonics, 1s@48kHz, theta=+{_THETA_TRUE}°, "
        f"d={D}m, {_N_SEEDS} seeds",
        results,
    )


def table_d() -> None:
    print("Table (d) — White noise with echo (0.005 s, 0.5)")
    print(f"  theta=+{_THETA_TRUE} deg, d={D} m, {_N_SEEDS} seeds")

    def white_source(rng):
        return rng.standard_normal(N)

    results = _sweep(white_source, echo=ECHO)
    _print_sweep_table(
        "Table (d) — White noise + echo (5ms, 0.5): beta x band x SNR",
        f"  theta=+{_THETA_TRUE}°, d={D}m, {_N_SEEDS} seeds",
        results,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _print_header()
    table_a()
    table_b()
    table_c()
    table_d()
