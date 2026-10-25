import json
from pathlib import Path

from kaawayan.acceptance import run_simulation


def test_simulated_acceptance_writes_labeled_reports(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "config" / "site.toml"
    report = run_simulation(config_path, tmp_path)

    assert report["mode"] == "SIMULATED"
    assert report["hardware_verified"] is False
    assert report["result"] == "SIMULATED PASS"
    assert len(report["checks"]) == 9
    assert all(
        item["status"].startswith(("SIMULATED ", "SKIPPED"))
        for item in report["checks"]
    )
    json_report = next(tmp_path.glob("*.json"))
    saved = json.loads(json_report.read_text(encoding="utf-8"))
    assert saved["config_sha256"] == report["config_sha256"]
    assert saved["integrity_sha256"]
    assert list(tmp_path.glob("*.md"))
