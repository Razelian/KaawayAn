"""
Tests for kaawayan.evaluate — Slice 6b (Step 4).

Tests on synthetic CSVs in tmp_path.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib

import numpy as np
import pytest

from kaawayan.evaluate import position_errors, summarize_errors, evaluate_trial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_positions_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fieldnames = ["t_laptop_s", "x_m", "y_m", "crossing_deg", "n_nodes", "status"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_meta(path: pathlib.Path, **kwargs) -> None:
    path.write_text(json.dumps(kwargs))


# ---------------------------------------------------------------------------
# position_errors tests
# ---------------------------------------------------------------------------

class TestPositionErrors:
    def test_exact_match(self):
        errs = position_errors([(1.0, 2.0)], (1.0, 2.0))
        assert len(errs) == 1
        assert errs[0] == pytest.approx(0.0)

    def test_known_error(self):
        # (4, 0) truth (0, 0) → error = 4.0
        errs = position_errors([(4.0, 0.0)], (0.0, 0.0))
        assert errs[0] == pytest.approx(4.0)

    def test_multiple_points(self):
        errs = position_errors([(1.0, 0.0), (0.0, 1.0)], (0.0, 0.0))
        assert len(errs) == 2
        assert all(abs(e - 1.0) < 1e-9 for e in errs)

    def test_empty(self):
        errs = position_errors([], (0.0, 0.0))
        assert len(errs) == 0


# ---------------------------------------------------------------------------
# summarize_errors tests
# ---------------------------------------------------------------------------

class TestSummarizeErrors:
    def test_basic(self):
        errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = summarize_errors(errs)
        assert s["n"] == 5
        assert s["median_m"] == pytest.approx(3.0)
        assert s["max_m"] == pytest.approx(5.0)
        assert s["p95_m"] >= s["median_m"]

    def test_single_element(self):
        s = summarize_errors(np.array([2.5]))
        assert s["n"] == 1
        assert s["median_m"] == pytest.approx(2.5)
        assert s["p95_m"] == pytest.approx(2.5)
        assert s["max_m"] == pytest.approx(2.5)

    def test_empty(self):
        s = summarize_errors(np.array([]))
        assert s["n"] == 0
        assert math.isnan(s["median_m"])
        assert math.isnan(s["p95_m"])
        assert math.isnan(s["max_m"])


# ---------------------------------------------------------------------------
# evaluate_trial tests
# ---------------------------------------------------------------------------

class TestEvaluateTrial:
    def _setup_trial(self, tmp_path, positions_rows, true_xy=None,
                     config_hash="abc123", git_commit="def456"):
        _write_positions_csv(tmp_path / "positions.csv", positions_rows)
        meta = {"config_hash": config_hash, "git_commit": git_commit}
        if true_xy is not None:
            meta["true_xy_m"] = list(true_xy)
        _write_meta(tmp_path / "meta.json", **meta)

    def test_ok_rows_only(self, tmp_path):
        """Only 'ok' rows are used for error computation."""
        rows = [
            {"t_laptop_s": 1.0, "x_m": 1.0, "y_m": 0.0, "crossing_deg": 60.0,
             "n_nodes": 2, "status": "ok"},
            {"t_laptop_s": 2.0, "x_m": 99.0, "y_m": 99.0, "crossing_deg": 5.0,
             "n_nodes": 1, "status": "uncertain"},
        ]
        self._setup_trial(tmp_path, rows, true_xy=(0.0, 0.0))
        result = evaluate_trial(tmp_path)
        # Only 1 ok row
        assert result["n"] == 1
        assert result["median_m"] == pytest.approx(1.0)

    def test_ok_fraction(self, tmp_path):
        rows = [
            {"t_laptop_s": 0.0, "x_m": 1.0, "y_m": 0.0, "crossing_deg": 60.0,
             "n_nodes": 2, "status": "ok"},
            {"t_laptop_s": 1.0, "x_m": "", "y_m": "", "crossing_deg": 0.0,
             "n_nodes": 1, "status": "uncertain"},
            {"t_laptop_s": 2.0, "x_m": "", "y_m": "", "crossing_deg": 0.0,
             "n_nodes": 1, "status": "degraded"},
        ]
        self._setup_trial(tmp_path, rows, true_xy=(0.0, 0.0))
        result = evaluate_trial(tmp_path)
        assert result["ok_fraction"] == pytest.approx(1 / 3)

    def test_config_hash_git_commit_echoed(self, tmp_path):
        rows = []
        self._setup_trial(tmp_path, rows, config_hash="sha256test", git_commit="abc1234")
        result = evaluate_trial(tmp_path)
        assert result["config_hash"] == "sha256test"
        assert result["git_commit"] == "abc1234"

    def test_truth_xy_m_in_result(self, tmp_path):
        rows = []
        self._setup_trial(tmp_path, rows, true_xy=(1.5, 2.5))
        result = evaluate_trial(tmp_path)
        assert result["truth_xy_m"] == pytest.approx([1.5, 2.5])

    def test_no_meta_json(self, tmp_path):
        """No meta.json → no crash, n=0."""
        rows = [{"t_laptop_s": 0.0, "x_m": 1.0, "y_m": 0.0,
                 "crossing_deg": 60.0, "n_nodes": 2, "status": "ok"}]
        _write_positions_csv(tmp_path / "positions.csv", rows)
        result = evaluate_trial(tmp_path)
        # No truth → n=0 in error summary
        assert result["n"] == 0

    def test_no_positions_csv(self, tmp_path):
        """No positions.csv → n=0."""
        _write_meta(tmp_path / "meta.json", true_xy_m=[1.0, 1.0],
                    config_hash="n/a", git_commit="n/a")
        result = evaluate_trial(tmp_path)
        assert result["n"] == 0

    def test_multiple_ok_rows_summary(self, tmp_path):
        truth = (2.0, 3.0)
        # Points at distance 1, 2, 3 from truth
        points = [(3.0, 3.0), (4.0, 3.0), (5.0, 3.0)]
        rows = [
            {"t_laptop_s": float(i), "x_m": x, "y_m": y,
             "crossing_deg": 60.0, "n_nodes": 2, "status": "ok"}
            for i, (x, y) in enumerate(points)
        ]
        self._setup_trial(tmp_path, rows, true_xy=truth)
        result = evaluate_trial(tmp_path)
        assert result["n"] == 3
        assert result["median_m"] == pytest.approx(2.0)
        assert result["max_m"] == pytest.approx(3.0)
