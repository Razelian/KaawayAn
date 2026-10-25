"""
kaawayan.capture — CLI capture tool and run_capture() helper.

run_capture(sources, recorder, assemblers, duration_s, clock, sleep,
            trial_id, kind, cfg_path, extra_meta) -> dict

    Loop: read each source, feed its assembler, route every complete frame to
    recorder.feed_frame(frame, t) where t = clock() at read time.
    Stop when duration_s has elapsed, or on KeyboardInterrupt.
    A source whose read() raises is logged, marked lost, others keep going;
    if ALL sources are lost, stops early.
    ALWAYS calls recorder.close(..., assemblers) in a finally block.

    Returns per-node summary dict:
        {node_id: {bytes, bytes_per_s, frames (by type name), total_gap_ms,
                   resets, crc_failures, resync_bytes_skipped, flagged, lost}}

CLI (python -m kaawayan.capture):
    --config  site.toml
    --out     trials/<trial_id>
    --kind    clap|tone|speaker|hover
    --seconds N
    [--true-theta 1=30,2=-30]
    [--true-xy 1.5,3.0]
    [--distance-m X]
    [--site S]
    [--notes T]
    --list-ports  prints port, serial_number, vid:pid then exits

UNTESTED ON HARDWARE: SerialSource, CLI real-port path.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as _time_mod
from typing import Any, Callable

log = logging.getLogger(__name__)

_FS = 48_000


# ---------------------------------------------------------------------------
# run_capture
# ---------------------------------------------------------------------------

def run_capture(
    sources: dict[int, Any],        # {node_id: source with read(size)->bytes and close()}
    recorder: Any,                   # TrialRecorder
    assemblers: dict[int, Any],      # {node_id: FrameAssembler}
    duration_s: float,
    clock: Callable[[], float] = _time_mod.monotonic,
    sleep: Callable[[float], None] = _time_mod.sleep,
    trial_id: str = "trial",
    kind: str = "unknown",
    cfg_path: Any = None,
    extra_meta: dict | None = None,
) -> dict[int, dict]:
    """Capture audio from all sources for duration_s seconds.

    Returns a per-node summary dict.
    """
    t_start = clock()
    t_end_target = t_start + duration_s

    # Per-node byte counters and frame counters
    bytes_read: dict[int, int] = {nid: 0 for nid in sources}
    frame_counts: dict[int, dict[str, int]] = {nid: {} for nid in sources}
    lost: dict[int, bool] = {nid: False for nid in sources}
    errors: dict[str, str] = {}
    _meta: dict | None = None  # filled by recorder.close()

    def _do_close() -> None:
        nonlocal _meta
        _extra = dict(extra_meta) if extra_meta else {}
        if errors:
            ce = dict(_extra.get("capture_errors", {}))
            ce.update(errors)
            _extra["capture_errors"] = ce
        _meta = recorder.close(trial_id, kind, cfg_path, _extra or None, assemblers)

    def _close_sources() -> None:
        for nid2, src2 in sources.items():
            if lost[nid2]:
                continue  # already closed when it raised
            try:
                src2.close()
            except Exception:
                pass

    try:
        while True:
            now = clock()
            if now >= t_end_target:
                break

            any_data = False
            for nid, src in list(sources.items()):
                if lost[nid]:
                    continue
                try:
                    data = src.read(4096)
                except Exception as exc:
                    msg = str(exc)
                    log.error("capture: node %d read() raised: %s", nid, msg)
                    errors[str(nid)] = msg
                    lost[nid] = True
                    try:
                        src.close()
                    except Exception:
                        pass
                    # Check if all sources lost
                    if all(lost[n] for n in sources):
                        elapsed = clock() - t_start
                        _do_close()
                        _close_sources()
                        return _build_summary(
                            sources, assemblers, bytes_read, frame_counts,
                            lost, errors, elapsed, _meta,
                        )
                    continue

                t_read = clock()
                if data:
                    any_data = True
                    bytes_read[nid] += len(data)
                    asm = assemblers[nid]
                    for frame, _ in asm.feed(data, t_read - t_start):
                        recorder.feed_frame(frame, t_read - t_start)
                        ftype = str(frame["type"].name)
                        frame_counts[nid][ftype] = frame_counts[nid].get(ftype, 0) + 1

            if not any_data:
                sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = clock() - t_start
        if _meta is None:
            _do_close()
        _close_sources()

    return _build_summary(
        sources, assemblers, bytes_read, frame_counts, lost, errors, elapsed, _meta,
    )


def _build_summary(
    sources: dict[int, Any],
    assemblers: dict[int, Any],
    bytes_read: dict[int, int],
    frame_counts: dict[int, dict[str, int]],
    lost: dict[int, bool],
    errors: dict[str, str],
    elapsed: float,
    meta: dict | None = None,
) -> dict[int, dict]:
    """Build per-node summary, pulling gap/reset/crc data from recorder meta."""
    out: dict[int, dict] = {}
    nodes_meta: dict[str, Any] = {}
    if meta is not None:
        nodes_meta = meta.get("nodes", {})
    for nid in sources:
        asm = assemblers[nid]
        nb = bytes_read[nid]
        bps = nb / elapsed if elapsed > 0 else 0.0
        nm = nodes_meta.get(str(nid), {})
        total_gap_ms = nm.get("total_gap_ms", 0.0)
        resets = nm.get("resets", 0)
        crc_failures = nm.get("crc_failures", asm.crc_failures)
        resync_bytes_skipped = nm.get("resync_bytes_skipped", asm.resync_bytes_skipped)
        flagged = (total_gap_ms > 100.0) or (resets > 0)
        out[nid] = {
            "bytes": nb,
            "bytes_per_s": bps,
            "frames": dict(frame_counts[nid]),
            "total_gap_ms": total_gap_ms,
            "resets": resets,
            "crc_failures": crc_failures,
            "resync_bytes_skipped": resync_bytes_skipped,
            "flagged": flagged,
            "lost": lost[nid],
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_true_theta(s: str) -> dict[str, float]:
    """Parse '1=30,2=-30' -> {'1': 30.0, '2': -30.0}."""
    result: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"--true-theta: expected 'id=angle' pairs, got {part!r}"
            )
        nid_s, angle_s = part.split("=", 1)
        try:
            result[nid_s.strip()] = float(angle_s.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--true-theta: could not parse angle {angle_s!r} as float"
            )
    return result


def _parse_true_xy(s: str) -> list[float]:
    """Parse '1.5,3.0' -> [1.5, 3.0]."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--true-xy: expected 'x,y', got {s!r}"
        )
    try:
        return [float(p.strip()) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--true-xy: could not parse {s!r} as two floats"
        )


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point. Returns exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m kaawayan.capture",
        description="Capture audio from nodes via USB serial.",
    )
    parser.add_argument("--config", required=False, default=None,
                        help="Path to site.toml")
    parser.add_argument("--out", required=False, default=None,
                        help="Output folder (refused if non-empty)")
    parser.add_argument("--kind",
                        choices=["clap", "tone", "speaker", "hover"],
                        default="clap")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--true-theta", default=None,
                        help="e.g. '1=30,2=-30'")
    parser.add_argument("--true-xy", default=None,
                        help="e.g. '1.5,3.0'")
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--site", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--list-ports", action="store_true",
                        help="List serial ports and exit")
    args = parser.parse_args(argv)

    if args.list_ports:
        import serial.tools.list_ports  # type: ignore
        for info in serial.tools.list_ports.comports():
            vid = f"{info.vid:04X}" if info.vid else "????"
            pid = f"{info.pid:04X}" if info.pid else "????"
            print(f"{info.device}  serial={info.serial_number or ''}  vid:pid={vid}:{pid}")
        return 0

    # Resolve config
    import pathlib
    cfg_path = pathlib.Path(args.config) if args.config else None
    if cfg_path is None:
        repo = pathlib.Path(__file__).resolve().parent.parent.parent
        cfg_path = repo / "config" / "site.toml"
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    from kaawayan.config import load_config
    cfg, _ = load_config(cfg_path)

    # Parse extras
    extra_meta: dict[str, Any] = {}

    if args.true_theta:
        try:
            extra_meta["true_theta_deg"] = _parse_true_theta(args.true_theta)
        except argparse.ArgumentTypeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if args.true_xy:
        try:
            extra_meta["true_xy_m"] = _parse_true_xy(args.true_xy)
        except argparse.ArgumentTypeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if args.distance_m is not None:
        extra_meta["distance_m"] = args.distance_m
    if args.site:
        extra_meta["site"] = args.site
    if args.notes:
        extra_meta["notes"] = args.notes

    # Out folder
    if args.out is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_id = f"trial_{ts}"
        out_dir = pathlib.Path("trials") / trial_id
    else:
        out_dir = pathlib.Path(args.out)
        trial_id = out_dir.name

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"ERROR: output folder is non-empty: {out_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    # Open SerialSource per node
    from kaawayan.record import FrameAssembler, SerialSource, TrialRecorder
    node_ids = [n["node_id"] for n in cfg.get("nodes", [])]
    recorder = TrialRecorder(out_dir, node_ids)
    assemblers: dict[int, FrameAssembler] = {nid: FrameAssembler() for nid in node_ids}

    sources: dict[int, Any] = {}
    for node in cfg.get("nodes", []):
        nid = node["node_id"]
        usb_serial = node.get("usb_serial", "")
        try:
            sources[nid] = SerialSource(usb_serial)
        except Exception as exc:
            print(f"ERROR: could not open node {nid} ({usb_serial}): {exc}", file=sys.stderr)
            for _src in sources.values():
                try:
                    _src.close()
                except Exception:
                    pass
            return 2

    print(f"Capturing {args.seconds} s into {out_dir} (kind={args.kind})")
    summary = run_capture(
        sources=sources,
        recorder=recorder,
        assemblers=assemblers,
        duration_s=args.seconds,
        trial_id=trial_id,
        kind=args.kind,
        cfg_path=cfg_path,
        extra_meta=extra_meta if extra_meta else None,
    )

    # Print summary table
    print(f"\n{'Node':>6}  {'bytes':>10}  {'bytes/s':>10}  {'total_gap_ms':>13}  {'resets':>6}  {'crc_fail':>8}  {'flagged':>8}")
    for nid in sorted(summary):
        s = summary[nid]
        flag = "YES" if s["flagged"] else "no"
        lost = " (LOST)" if s["lost"] else ""
        print(f"{nid:>6}  {s['bytes']:>10}  {s['bytes_per_s']:>10.1f}"
              f"  {s['total_gap_ms']:>13.1f}  {s['resets']:>6}  {s['crc_failures']:>8}  {flag:>8}{lost}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
