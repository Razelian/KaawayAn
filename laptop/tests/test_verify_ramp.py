"""
tests/test_verify_ramp.py — Tests for laptop/scripts/verify_ramp.py

WAVs are built via TrialRecorder from protocol.encode_frame frames carrying
ramp data (NOT hand-written WAVs).
"""

from __future__ import annotations

import json
import math
import pathlib
import wave

import numpy as np
import pytest

from kaawayan.protocol import encode_frame, FrameType
from kaawayan.record import FrameAssembler, TrialRecorder

# Import verify_ramp from scripts/ (not a package, use importlib)
import importlib.util
import sys

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location(
    "verify_ramp", _SCRIPTS / "verify_ramp.py"
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore
_spec.loader.exec_module(_mod)  # type: ignore
verify_node = _mod.verify_node
verify_ramp_main = _mod.main
print_report = _mod.print_report

_FS = 48_000
_SPF = 512  # samples per AUDIO frame


# ---------------------------------------------------------------------------
# Ramp builders
# ---------------------------------------------------------------------------

def _ramp_payload(seq: int, n_samples: int = _SPF) -> bytes:
    """Build ramp payload: L = int16 wrap of (n mod 65536), R = ~L."""
    n_start = seq * _SPF
    l_u16 = np.arange(n_start, n_start + n_samples, dtype=np.int64) % 65536
    l_u16 = l_u16.astype(np.uint16)
    l_i16 = l_u16.view(np.int16)
    r_i16 = (~l_u16).view(np.int16)
    arr = np.stack([l_i16, r_i16], axis=1).copy()
    return arr.tobytes()


def _ramp_audio_frame(node_id: int, seq: int, n_samples: int = _SPF) -> bytes:
    payload = _ramp_payload(seq, n_samples)
    # If n_samples != _SPF, still encode; FrameAssembler accepts any len
    return encode_frame(FrameType.AUDIO, node_id, seq, 0, payload)


def _startup_frame(node_id: int, mic_flags: int = 0x83) -> bytes:
    """Build STARTUP frame; mic_flags=0x83 means bit7 (ramp) + bits 0,1."""
    import struct
    payload = struct.pack("<BBBBIIHBB", 1, 0, 1, 0, 0, _FS, _SPF, 2, mic_flags)
    return encode_frame(FrameType.STARTUP, node_id, 0, 0, payload)


def _build_trial(
    tmp_path: pathlib.Path,
    node_id: int,
    seqs: list[int],
    mic_flags: int = 0x83,
) -> pathlib.Path:
    """
    Stream frames through TrialRecorder → writes node{id}.wav + meta.json.
    seqs: list of seq numbers to encode (missing seqs → gaps / zero-fill).
    Returns trial_dir.
    """
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(exist_ok=True)

    recorder = TrialRecorder(trial_dir, [node_id])
    asm = FrameAssembler()

    # Feed STARTUP first
    for frame, t in asm.feed(_startup_frame(node_id, mic_flags), 0.0):
        recorder.feed_frame(frame, t)

    t = 0.0
    for seq in seqs:
        raw = _ramp_audio_frame(node_id, seq)
        for frame, ft in asm.feed(raw, t):
            recorder.feed_frame(frame, ft)
        t += _SPF / _FS

    recorder.close("t_ramp", "tone", None, None, {node_id: asm})
    return trial_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanRamp:
    """3 s clean ramp → PASS, zero bad, zero zeros."""

    def _make(self, tmp_path: pathlib.Path) -> dict:
        nid = 1
        n_frames = int(math.ceil(3.0 * _FS / _SPF))  # 144
        trial_dir = _build_trial(tmp_path, nid, list(range(n_frames)))
        return verify_node(trial_dir / f"node{nid}.wav", nid)

    def test_verdict_pass(self, tmp_path: pathlib.Path) -> None:
        r = self._make(tmp_path)
        assert r["verdict"] == "PASS"

    def test_zero_bad(self, tmp_path: pathlib.Path) -> None:
        r = self._make(tmp_path)
        assert r["bad"] == 0

    def test_zero_zeros(self, tmp_path: pathlib.Path) -> None:
        r = self._make(tmp_path)
        assert r["zero"] == 0


class TestOneDroppedFrame:
    """One dropped AUDIO frame → PASS with one zero run of 512 samples."""

    def _make(self, tmp_path: pathlib.Path) -> tuple[dict, pathlib.Path]:
        nid = 1
        # seqs 0..4, skip 5, 6..10
        seqs = list(range(5)) + list(range(6, 11))
        trial_dir = _build_trial(tmp_path, nid, seqs)
        r = verify_node(trial_dir / f"node{nid}.wav", nid)
        return r, trial_dir

    def test_verdict_pass(self, tmp_path: pathlib.Path) -> None:
        r, _ = self._make(tmp_path)
        assert r["verdict"] == "PASS"

    def test_zero_run_512_samples(self, tmp_path: pathlib.Path) -> None:
        r, _ = self._make(tmp_path)
        zero_runs = [x for x in r["runs"] if x[0] == "zero"]
        assert len(zero_runs) == 1
        assert zero_runs[0][2] == 512  # length = 512 samples

    def test_zero_ms_approx(self, tmp_path: pathlib.Path) -> None:
        r, _ = self._make(tmp_path)
        expected_ms = 512 / _FS * 1000.0  # ≈10.666...
        assert abs(r["total_zero_ms"] - expected_ms) < 0.5

    def test_no_bad_samples(self, tmp_path: pathlib.Path) -> None:
        r, _ = self._make(tmp_path)
        assert r["bad"] == 0


class TestFirmwareSideLoss:
    """
    Firmware-side loss: 100 samples deleted INSIDE one frame's payload while seq
    is continuous → UNDETECTED_LOSS with first bad index at deletion point.
    """

    def _make(self, tmp_path: pathlib.Path) -> dict:
        nid = 1
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir(exist_ok=True)
        recorder = TrialRecorder(trial_dir, [nid])
        asm = FrameAssembler()

        for frame, t in asm.feed(_startup_frame(nid), 0.0):
            recorder.feed_frame(frame, t)

        t = 0.0
        # Frames 0..4 clean
        for seq in range(5):
            for frame, ft in asm.feed(_ramp_audio_frame(nid, seq), t):
                recorder.feed_frame(frame, ft)
            t += _SPF / _FS

        # Frame 5: delete 100 samples from the middle (keep 412 samples but
        # write them as a payload of 412*4 bytes with seq=5).
        # The frame LOOKS like seq=5 with 512-sample payload but actually
        # has only 412 samples encoded (rest dropped by firmware).
        # Simulate: take the normal seq=5 payload, delete samples [200:300].
        full_payload = _ramp_payload(5, _SPF)
        arr = np.frombuffer(full_payload, dtype="<i2").reshape(_SPF, 2).copy()
        # Delete samples 200..299 (100 samples)
        arr_cut = np.concatenate([arr[:200], arr[300:]], axis=0)  # 412 samples
        # But TrialRecorder expects 512-sample frames (2048 bytes) for WAV.
        # To simulate firmware dropping 100 samples: pad with ramp-WRONG data
        # (zeros from position 412 to 512) so the payload is still 2048 bytes
        # but samples 412..511 are wrong.
        pad = np.zeros((100, 2), dtype=np.int16)
        arr_bad = np.concatenate([arr_cut, pad], axis=0)  # 512 samples, wrong tail
        bad_frame = encode_frame(FrameType.AUDIO, nid, 5, 0, arr_bad.tobytes())
        for frame, ft in asm.feed(bad_frame, t):
            recorder.feed_frame(frame, ft)
        t += _SPF / _FS

        # Frames 6..10 continue with correct ramp (but shifted)
        for seq in range(6, 11):
            for frame, ft in asm.feed(_ramp_audio_frame(nid, seq), t):
                recorder.feed_frame(frame, ft)
            t += _SPF / _FS

        recorder.close("t_fw_loss", "tone", None, None, {nid: asm})
        return verify_node(trial_dir / f"node{nid}.wav", nid)

    def test_undetected_loss(self, tmp_path: pathlib.Path) -> None:
        r = self._make(tmp_path)
        assert r["verdict"] == "UNDETECTED_LOSS"

    def test_first_bad_index_at_deletion(self, tmp_path: pathlib.Path) -> None:
        r = self._make(tmp_path)
        # Deletion starts at sample 200 inside frame 5
        # => global index = 5*512 + 200 = 2760
        assert r["first_bad_idx"] is not None
        assert r["first_bad_idx"] == 5 * _SPF + 200


class TestWrapAcross65536:
    """Wrap across n=65536 (≥2.8 s of audio) → PASS."""

    def test_wrap_pass(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        # 65536 samples = 128 frames; add 10 more past the wrap
        n_frames = 128 + 10  # 138 frames ≈ 1.47 s
        trial_dir = _build_trial(tmp_path, nid, list(range(n_frames)))
        r = verify_node(trial_dir / f"node{nid}.wav", nid)
        assert r["verdict"] == "PASS"
        assert r["bad"] == 0


class TestLateConnect:
    """Late-connect (first seq=7) → n0 inferred, PASS."""

    def test_late_connect_pass(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        # Start at seq=7, no prior frames
        seqs = list(range(7, 20))
        trial_dir = _build_trial(tmp_path, nid, seqs)
        r = verify_node(trial_dir / f"node{nid}.wav", nid)
        # Note: TrialRecorder will zero-fill seqs 0..6 (7 frames = 3584 zero samples)
        # Those zeros are detected as zero-fills (ok for ramp verifier)
        # Then seqs 7..19 are ramp data
        # The verifier infers n0 from first non-zero sample (at offset 7*512=3584)
        assert r["verdict"] == "PASS"
        assert r["bad"] == 0


class TestNonRampNoise:
    """Non-ramp white noise → UNDETECTED_LOSS."""

    def test_noise_undetected_loss(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir(exist_ok=True)
        recorder = TrialRecorder(trial_dir, [nid])
        asm = FrameAssembler()

        for frame, t in asm.feed(_startup_frame(nid, mic_flags=0x03), 0.0):
            recorder.feed_frame(frame, t)

        rng = np.random.default_rng(42)
        t = 0.0
        for seq in range(20):
            samples = rng.integers(-32768, 32767, (_SPF, 2), dtype=np.int16)
            raw = encode_frame(FrameType.AUDIO, nid, seq, 0, samples.tobytes())
            for frame, ft in asm.feed(raw, t):
                recorder.feed_frame(frame, ft)
            t += _SPF / _FS

        recorder.close("t_noise", "tone", None, None, {nid: asm})
        r = verify_node(trial_dir / f"node{nid}.wav", nid)
        assert r["verdict"] == "UNDETECTED_LOSS"


class TestExitCodes:
    """Exit code 0 for PASS, 1 for UNDETECTED_LOSS."""

    def test_exit_0_clean(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        trial_dir = _build_trial(tmp_path, nid, list(range(20)))
        rc = verify_ramp_main([str(trial_dir), "--node", "1"])
        assert rc == 0

    def test_exit_1_noise(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir(exist_ok=True)
        recorder = TrialRecorder(trial_dir, [nid])
        asm = FrameAssembler()
        rng = np.random.default_rng(7)
        t = 0.0
        for seq in range(10):
            samples = rng.integers(-32768, 32767, (_SPF, 2), dtype=np.int16)
            raw = encode_frame(FrameType.AUDIO, nid, seq, 0, samples.tobytes())
            for frame, ft in asm.feed(raw, t):
                recorder.feed_frame(frame, ft)
            t += _SPF / _FS
        recorder.close("t_noise2", "tone", None, None, {nid: asm})
        rc = verify_ramp_main([str(trial_dir), "--node", "1"])
        assert rc == 1


class TestTwoNodesSeparate:
    """Two nodes reported separately."""

    def test_two_nodes(self, tmp_path: pathlib.Path) -> None:
        for nid in [1, 2]:
            trial_dir = tmp_path / "trial"
            trial_dir.mkdir(exist_ok=True)
            # Use _build_trial but we need both nodes in the same dir
        # Build fresh
        trial_dir = tmp_path / "trial2"
        trial_dir.mkdir(exist_ok=True)
        recorder = TrialRecorder(trial_dir, [1, 2])
        for nid in [1, 2]:
            asm = FrameAssembler()
            for frame, t in asm.feed(_startup_frame(nid), 0.0):
                recorder.feed_frame(frame, t)
            t = 0.0
            for seq in range(10):
                for frame, ft in asm.feed(_ramp_audio_frame(nid, seq), t):
                    recorder.feed_frame(frame, ft)
                t += _SPF / _FS
        recorder.close("t_two", "tone")

        r1 = verify_node(trial_dir / "node1.wav", 1)
        r2 = verify_node(trial_dir / "node2.wav", 2)
        assert r1["node_id"] == 1
        assert r2["node_id"] == 2
        assert r1["verdict"] == "PASS"
        assert r2["verdict"] == "PASS"


class TestDeterministicOutput:
    """Deterministic output (same input → identical stdout twice)."""

    def test_deterministic(self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture) -> None:
        nid = 1
        trial_dir = _build_trial(tmp_path, nid, list(range(20)))

        verify_ramp_main([str(trial_dir), "--node", "1"])
        out1 = capsys.readouterr().out

        verify_ramp_main([str(trial_dir), "--node", "1"])
        out2 = capsys.readouterr().out

        # Git commit line may differ if HEAD changes, but the rest must be equal.
        # Strip first line (git commit) for comparison.
        lines1 = out1.splitlines()[1:]
        lines2 = out2.splitlines()[1:]
        assert lines1 == lines2


class TestWarnRampOff:
    """WARN printed when mic_flags bit 7 not set."""

    def test_warn_when_bit7_not_set(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        # mic_flags = 0x03 (bits 0,1 set, bit 7 not set)
        trial_dir = _build_trial(tmp_path, nid, list(range(10)), mic_flags=0x03)
        # Load the meta.json that was written (mic_flags stored there)
        meta_path = trial_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        r = verify_node(trial_dir / f"node{nid}.wav", nid, meta)
        assert r["warn_ramp_off"] is True

    def test_no_warn_when_bit7_set(self, tmp_path: pathlib.Path) -> None:
        nid = 1
        # mic_flags = 0x83 (bit 7 + bits 0,1)
        trial_dir = _build_trial(tmp_path, nid, list(range(10)), mic_flags=0x83)
        meta_path = trial_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
        r = verify_node(trial_dir / f"node{nid}.wav", nid, meta)
        assert r["warn_ramp_off"] is False
