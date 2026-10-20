"""
Tests for kaawayan.windows — Slice 4t.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from kaawayan.windows import iter_windows, band_energy_db

FS = 48_000


def _make_stereo(signal: np.ndarray) -> np.ndarray:
    """Repeat mono signal as stereo (n, 2) int16."""
    s = (signal * 32767).astype(np.int16)
    return np.column_stack([s, s])


class TestIterWindows:
    def test_full_windows_only(self):
        """3 s of data, window=1 s, hop=0.5 s → 5 windows."""
        n = 3 * FS
        x = np.zeros((n, 2), dtype=np.int16)
        wins = list(iter_windows(x, FS, window_s=1.0, hop_s=0.5))
        assert len(wins) == 5

    def test_start_indices_correct(self):
        n = 3 * FS
        x = np.zeros((n, 2), dtype=np.int16)
        wins = list(iter_windows(x, FS, window_s=1.0, hop_s=0.5))
        expected_starts = [0, FS // 2, FS, 3 * FS // 2, 2 * FS]
        for (start, _win), expected in zip(wins, expected_starts):
            assert start == expected

    def test_window_shape(self):
        n = 2 * FS
        x = np.zeros((n, 2), dtype=np.int16)
        wins = list(iter_windows(x, FS, window_s=1.0, hop_s=1.0))
        for _start, win in wins:
            assert win.shape == (FS, 2)

    def test_no_partial_windows(self):
        """A partial window at the end is dropped."""
        # 2.7 s → floor((2.7 - 1.0) / 0.5) + 1 = floor(3.4) + 1 = 4 windows
        n = int(2.7 * FS)
        x = np.zeros((n, 2), dtype=np.int16)
        wins = list(iter_windows(x, FS, window_s=1.0, hop_s=0.5))
        assert len(wins) == 4

    def test_1s_hop_1s_3s_data(self):
        """3 s / 1 s window / 1 s hop → 3 windows."""
        n = 3 * FS
        x = np.zeros((n, 2), dtype=np.int16)
        wins = list(iter_windows(x, FS, window_s=1.0, hop_s=1.0))
        assert len(wins) == 3


class TestBandEnergyDb:
    def test_silence_below_minus_100(self):
        """Silence → very low power (floored at -120 dB)."""
        x = np.zeros((FS, 2), dtype=np.int16)
        db = band_energy_db(x, FS, (300, 1500))
        assert db <= -100.0

    def test_sine_0p1_fs_at_1khz(self):
        """Sine at 1 kHz, amplitude = 0.1 FS → band energy ≈ -23.0 ± 0.2 dB."""
        amp = 0.1  # fraction of full-scale
        t = np.arange(FS) / FS
        mono = (amp * np.sin(2 * math.pi * 1000 * t) * 32767).astype(np.int16)
        x = np.column_stack([mono, mono])
        db = band_energy_db(x, FS, (300, 4000))
        # Expected: 20*log10(0.1) ≈ -20 dB power is 10*log10(0.01) = -20 dB
        # Power of amp=0.1 sine: mean(sin^2) * amp^2 = 0.5 * 0.01 = 0.005
        # normalized: 0.005 (since /32768^2 * 32767^2 ≈ 0.005)
        # 10*log10(0.005) ≈ -23.0 dB
        assert abs(db - (-23.0)) < 0.2, f"Expected ~-23.0 dB, got {db:.3f} dB"

    def test_out_of_band_sine_very_low(self):
        """Sine at 5 kHz with band [300, 1500] should be near floor."""
        t = np.arange(FS) / FS
        mono = (0.5 * np.sin(2 * math.pi * 5000 * t) * 32767).astype(np.int16)
        x = np.column_stack([mono, mono])
        db = band_energy_db(x, FS, (300, 1500))
        assert db <= -40.0  # well below in-band signal
