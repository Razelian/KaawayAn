"""
verify_ramp.py <trial_dir> [--node ID]...

Verify that node WAV files in a trial directory contain valid counter-ramp data
as defined in CONTEXT.md §4b.

Ramp contract:
  n = seq*512 + i  (global stereo sample index since boot)
  L[i] = int16 wrap of (n mod 65536)
  R[i] = ~L[i]  (bitwise NOT as int16 = -L-1)
  (0,0) never occurs in ramp data; recorder zero-fill is detectable.

Usage:
  python -m scripts.verify_ramp <trial_dir> [--node ID]...

Exit code: 0 if all nodes PASS, 1 if any node is UNDETECTED_LOSS.

Output is deterministic (no randomness, no timestamps).

UNTESTED ON HARDWARE: SerialSource path (CLI --serial option not implemented here;
  see rawdump.py for live capture). This tool reads already-captured WAVs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import subprocess
import sys
import wave

import numpy as np


_FS = 48_000
_SAMPLES_PER_FRAME = 512
_MS_PER_SAMPLE = 1000.0 / _FS


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "n/a"
    except Exception:
        return "n/a"


def _read_wav_int16(path: pathlib.Path) -> np.ndarray:
    """Read a stereo int16 WAV; return (n_samples, 2) int16 array."""
    with wave.open(str(path), "rb") as wf:
        n_ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"expected 16-bit WAV, got {sampwidth*8}-bit")
    arr = np.frombuffer(raw, dtype="<i2")
    if n_ch == 2:
        arr = arr.reshape(-1, 2)
    elif n_ch == 1:
        arr = np.stack([arr, arr], axis=1)
    else:
        raise ValueError(f"unexpected channel count {n_ch}")
    return arr


def _classify_samples(
    arr: np.ndarray,
) -> tuple[int, list[tuple[str, int, int, float]], int, int, int]:
    """
    Infer n0 from first non-zero sample, classify each sample as ok/zero/bad.

    Returns (n0, runs, ok_count, zero_count, bad_count).
    runs = list of ("zero"|"bad", start_idx, length, ms)
    """
    n = len(arr)
    L = arr[:, 0]
    R = arr[:, 1]

    # Find first non-zero sample to infer n0
    nonzero_mask = ~((L == 0) & (R == 0))
    nonzero_idx = np.where(nonzero_mask)[0]
    if len(nonzero_idx) == 0:
        # All zeros — cannot infer n0
        return 0, [("zero", 0, n, n * _MS_PER_SAMPLE)], 0, n, 0

    first_nz = nonzero_idx[0]
    l0_raw = int(L[first_nz]) & 0xFFFF  # treat as unsigned
    # n0 is the global index of the first non-zero sample
    # E[first_nz] should equal l0_raw as unsigned -> n0 + first_nz wraps to l0_raw
    # n0 = (l0_raw - first_nz) mod 65536
    n0 = int((l0_raw - first_nz) % 65536)

    indices = (np.arange(n, dtype=np.uint64) + n0) & 0xFFFF
    expected_left = indices.astype(np.uint16).view(np.int16)
    expected_right = np.bitwise_not(expected_left)
    zero_mask = (L == 0) & (R == 0)
    ok_mask = (L == expected_left) & (R == expected_right)
    label = np.where(zero_mask, 1, np.where(ok_mask, 0, 2)).astype(np.int8)
    zero_count = int(np.count_nonzero(label == 1))
    ok_count = int(np.count_nonzero(label == 0))
    bad_count = int(np.count_nonzero(label == 2))

    # Build runs of non-ok samples
    runs: list[tuple[str, int, int, float]] = []
    i = 0
    while i < n:
        if label[i] != 0:
            kind = "zero" if label[i] == 1 else "bad"
            start = i
            while i < n and label[i] == label[start]:
                i += 1
            length = i - start
            ms = length * _MS_PER_SAMPLE
            runs.append((kind, start, length, ms))
        else:
            i += 1

    return n0, runs, ok_count, zero_count, bad_count


def verify_node(
    wav_path: pathlib.Path,
    node_id: int | str,
    meta: dict | None = None,
) -> dict:
    """
    Verify one node's WAV against the ramp contract.

    Returns dict with keys:
      node_id, n_samples, n0, ok, zero, bad, runs,
      total_zero_ms, verdict ("PASS" or "UNDETECTED_LOSS"),
      first_bad_idx (int or None), warn_ramp_off (bool)
    """
    arr = _read_wav_int16(wav_path)
    n_samples = len(arr)
    n0, runs, ok, zero, bad = _classify_samples(arr)

    total_zero_ms = sum(r[3] for r in runs if r[0] == "zero")

    # Check meta for ramp mode flag (mic_flags bit 7)
    warn_ramp_off = False
    if meta is not None:
        node_meta = meta.get("nodes", {}).get(str(node_id), {})
        mic_flags = node_meta.get("mic_flags")
        if mic_flags is not None and not (mic_flags & 0x80):
            warn_ramp_off = True

    first_bad_idx = None
    for kind, start, length, ms in runs:
        if kind == "bad":
            first_bad_idx = start
            break

    verdict = "PASS" if bad == 0 else "UNDETECTED_LOSS"

    return {
        "node_id": node_id,
        "n_samples": n_samples,
        "n0": n0,
        "ok": ok,
        "zero": zero,
        "bad": bad,
        "runs": runs,
        "total_zero_ms": total_zero_ms,
        "verdict": verdict,
        "first_bad_idx": first_bad_idx,
        "warn_ramp_off": warn_ramp_off,
    }


def print_report(results: list[dict], meta: dict | None = None) -> None:
    """Print deterministic report to stdout."""
    git = _git_commit()
    print(f"verify_ramp  git={git}")
    print()

    for r in results:
        nid = r["node_id"]
        print(f"=== node {nid}  samples={r['n_samples']}  n0={r['n0']} ===")
        print(f"  ok={r['ok']}  zero={r['zero']}  bad={r['bad']}")
        print(f"  total_zero_ms={r['total_zero_ms']:.2f}")
        if r["first_bad_idx"] is not None:
            print(f"  FIRST BAD INDEX: {r['first_bad_idx']}")
        if r["warn_ramp_off"]:
            print("  WARN: STARTUP mic_flags bit 7 not set (ramp mode off?)")
        if meta is not None:
            node_meta = meta.get("nodes", {}).get(str(nid), {})
            meta_gap = node_meta.get("total_gap_ms")
            if meta_gap is not None:
                print(f"  meta.json total_gap_ms={meta_gap:.2f}  verify_ramp zero_ms={r['total_zero_ms']:.2f}")
        for run in r["runs"]:
            kind, start, length, ms = run
            print(f"  {kind}  start={start}  len={length}  ms={ms:.2f}")
        print(f"  => {r['verdict']}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.verify_ramp",
        description="Verify counter-ramp integrity in trial WAV files.",
    )
    parser.add_argument("trial_dir", type=pathlib.Path)
    parser.add_argument("--node", dest="nodes", type=int, action="append",
                        metavar="ID", help="Node ID(s) to check (default: all)")
    args = parser.parse_args(argv)

    trial_dir = args.trial_dir.resolve()
    if not trial_dir.is_dir():
        print(f"ERROR: {trial_dir} is not a directory", file=sys.stderr)
        return 2

    # Load meta.json if present
    meta: dict | None = None
    meta_path = trial_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    # Determine node IDs to check
    if args.nodes:
        node_ids = args.nodes
    else:
        # Auto-discover from node*.wav files
        wavs = sorted(trial_dir.glob("node*.wav"))
        node_ids = []
        for w in wavs:
            stem = w.stem  # "node1"
            try:
                node_ids.append(int(stem.replace("node", "")))
            except ValueError:
                pass

    if not node_ids:
        print("ERROR: no node WAV files found", file=sys.stderr)
        return 2

    results = []
    for nid in sorted(node_ids):
        wav_path = trial_dir / f"node{nid}.wav"
        if not wav_path.exists():
            print(f"ERROR: {wav_path} not found", file=sys.stderr)
            return 2
        r = verify_node(wav_path, nid, meta)
        results.append(r)

    print_report(results, meta)

    any_fail = any(r["verdict"] != "PASS" for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
