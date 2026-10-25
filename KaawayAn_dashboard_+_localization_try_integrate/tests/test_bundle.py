"""Tests for tools/bundle.py."""

from __future__ import annotations

import zipfile
from pathlib import Path

from tools.bundle import (
    EXCLUDED_DIR_OR_FILE_PATTERNS,
    MODEL_WEIGHT_PATTERNS,
    create_bundle,
    should_exclude,
)


def test_bundle_no_excluded_patterns_slip_in(tmp_path):
    """Prove that no excluded pattern slips into the created bundle zip."""
    dash_root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "dist"

    zip_path = create_bundle(
        dash_root=dash_root,
        out_dir=out_dir,
        with_model=False,
    )

    assert zip_path.is_file(), "Bundle zip was not created"
    assert zip_path.stat().st_size <= 2 * 1024 * 1024, "Bundle exceeds 2 MB limit"

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "MANIFEST.txt" in namelist, "MANIFEST.txt missing from zip"
        assert len(namelist) == len(set(namelist)), "Bundle contains duplicate file entries"
        assert "kaawayan/tests/golden/frames.hex" in namelist

        # Check content of MANIFEST.txt
        manifest_content = zf.read("MANIFEST.txt").decode("utf-8")
        assert len(manifest_content.strip().splitlines()) >= 10

        # Check all files inside the zip
        for name in namelist:
            if name == "MANIFEST.txt":
                continue

            # Verify no excluded directory or extension
            for pat in EXCLUDED_DIR_OR_FILE_PATTERNS:
                raw_pat = pat.replace("*", "").lower()
                assert raw_pat not in name.lower(), f"Excluded pattern '{pat}' matched in zip member '{name}'"

            for pat in MODEL_WEIGHT_PATTERNS:
                raw_pat = pat.replace("*", "").lower()
                assert not name.lower().endswith(raw_pat), f"Model weight '{pat}' slipped in: '{name}'"

            # Check that only code.html is included from ui_assets
            if "ui_assets" in name:
                assert name.endswith("code.html"), f"Non-code.html asset slipped in from ui_assets: '{name}'"


def test_should_exclude_logic():
    """Unit test for should_exclude matcher."""
    assert should_exclude(".venv/lib/foo.py")
    assert should_exclude("foo/__pycache__/bar.cpython-311.pyc")
    assert should_exclude("some/file.pyc")
    assert should_exclude("database/events.db")
    assert should_exclude("database/events.db-wal")
    assert should_exclude("database/events.db-shm")
    assert should_exclude("staging/dumps/raw.bin")
    assert should_exclude("trials/trial_01/node1.wav")
    assert should_exclude("dashboard/ui_assets/screen.png")
    assert should_exclude("model/weights.pt", with_model=False)
    assert not should_exclude("model/weights.pt", with_model=True)
    assert not should_exclude("main.py")
    assert not should_exclude("dashboard/ui_assets/about_team/code.html")
