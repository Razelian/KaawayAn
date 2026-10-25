from pathlib import Path

import pytest

from kaawayan.config import ConfigError, load_config, validate_live_config


SITE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "site.toml"


def test_checked_in_site_config_is_explicitly_demo_only_until_surveyed():
    config, _ = load_config(SITE_CONFIG)
    with pytest.raises(ConfigError, match="placeholder"):
        validate_live_config(config)


def test_live_config_rejects_usb_serial_placeholder():
    config, _ = load_config(SITE_CONFIG)
    config["nodes"][0]["usb_serial"] = "REPLACE_WITH_NODE_SERIAL"
    with pytest.raises(ConfigError, match="usb_serial"):
        validate_live_config(config)


def test_live_config_accepts_a_supplied_serial_identifier():
    config, _ = load_config(SITE_CONFIG)
    for node in config["nodes"]:
        node["usb_serial"] = f"surveyed-{node['node_id']}"
    validate_live_config(config)


def test_config_rejects_duplicate_node_ids(tmp_path):
    source = SITE_CONFIG.read_text(encoding="utf-8")
    source = source.replace("node_id      = 2", "node_id      = 1", 1)
    config_path = tmp_path / "site.toml"
    config_path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate node_id"):
        load_config(config_path)


def test_config_rejects_heading_outside_circle(tmp_path):
    source = SITE_CONFIG.read_text(encoding="utf-8")
    source = source.replace("heading_deg  = 63.4", "heading_deg  = 360.0", 1)
    config_path = tmp_path / "site.toml"
    config_path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="heading_deg"):
        load_config(config_path)


@pytest.mark.parametrize("spacing", ["0.019", "0.501"])
def test_config_rejects_mic_spacing_outside_supported_range(tmp_path, spacing):
    source = SITE_CONFIG.read_text(encoding="utf-8")
    source = source.replace("mic_spacing_m          = 0.10", f"mic_spacing_m          = {spacing}", 1)
    config_path = tmp_path / "site.toml"
    config_path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError, match="mic_spacing_m"):
        load_config(config_path)
