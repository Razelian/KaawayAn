"""Tests for kaawayan.config – written BEFORE the implementation."""
import hashlib
import textwrap
import tomllib
from pathlib import Path
import pytest
from kaawayan.config import load_config, ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_TOML = textwrap.dedent("""\
    convention_version = 1
    mic_spacing_m = 0.10
    band_hz = [300, 4000]
    window_s = 1.0
    hop_s = 0.5
    pair_tolerance_s = 0.1
    gate_min_confidence = 0.3
    gate_min_band_energy_db = -40.0
    min_crossing_deg = 30

    [[nodes]]
    node_id = 1
    usb_serial = "ABCD1234"
    board = "xiao"
    x_m = 0.0
    y_m = 0.0
    heading_deg = 0.0
    tau0_s = 0.0
    channel_swap = false
    mic_model = "INMP441"

    [[nodes]]
    node_id = 2
    usb_serial = "EFGH5678"
    board = "devkit"
    x_m = 1.5
    y_m = 0.0
    heading_deg = 180.0
    tau0_s = 0.0
    channel_swap = false
    mic_model = "INMP441"
""")

# ---------------------------------------------------------------------------
# 1. Valid config loads and returns sha256 + dict
# ---------------------------------------------------------------------------

def test_load_returns_config_and_hash(tmp_path):
    f = tmp_path / "site.toml"
    f.write_text(VALID_TOML, encoding="utf-8")
    cfg, sha = load_config(f)
    assert cfg["convention_version"] == 1
    assert len(cfg["nodes"]) == 2
    expected = hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha == expected


def test_hash_changes_when_file_changes(tmp_path):
    f = tmp_path / "site.toml"
    f.write_text(VALID_TOML, encoding="utf-8")
    _, sha1 = load_config(f)
    f.write_text(VALID_TOML.replace("mic_spacing_m = 0.10", "mic_spacing_m = 0.12"), encoding="utf-8")
    _, sha2 = load_config(f)
    assert sha1 != sha2


# ---------------------------------------------------------------------------
# 2. Missing required global keys raise ConfigError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", [
    "convention_version",
    "mic_spacing_m",
    "band_hz",
    "window_s",
    "hop_s",
    "pair_tolerance_s",
    "gate_min_confidence",
    "gate_min_band_energy_db",
    "min_crossing_deg",
])
def test_missing_global_key_raises(tmp_path, missing_key):
    lines = [l for l in VALID_TOML.splitlines() if not l.startswith(missing_key)]
    bad_toml = "\n".join(lines)
    f = tmp_path / "site.toml"
    f.write_text(bad_toml, encoding="utf-8")
    with pytest.raises(ConfigError, match=missing_key):
        load_config(f)


# ---------------------------------------------------------------------------
# 3. Missing required per-node keys raise ConfigError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", [
    "node_id", "usb_serial", "board", "x_m", "y_m", "heading_deg",
])
def test_missing_node_key_raises(tmp_path, missing_key):
    # Remove only the first occurrence in the first node block
    lines = VALID_TOML.splitlines()
    removed = False
    filtered = []
    for l in lines:
        if not removed and l.strip().startswith(missing_key):
            removed = True
            continue
        filtered.append(l)
    bad_toml = "\n".join(filtered)
    f = tmp_path / "site.toml"
    f.write_text(bad_toml, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(f)


# ---------------------------------------------------------------------------
# 4. No nodes raises ConfigError
# ---------------------------------------------------------------------------

def test_no_nodes_raises(tmp_path):
    toml_no_nodes = "\n".join(
        l for l in VALID_TOML.splitlines()
        if not l.startswith("[[nodes]]") and not any(
            l.strip().startswith(k) for k in [
                "node_id", "usb_serial", "board", "x_m", "y_m",
                "heading_deg", "tau0_s", "channel_swap", "mic_model"
            ]
        )
    )
    f = tmp_path / "site.toml"
    f.write_text(toml_no_nodes, encoding="utf-8")
    with pytest.raises(ConfigError, match="node"):
        load_config(f)


# ---------------------------------------------------------------------------
# 5. Invalid board value raises ConfigError
# ---------------------------------------------------------------------------

def test_invalid_board_raises(tmp_path):
    bad = VALID_TOML.replace('board = "xiao"', 'board = "raspberry"', 1)
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="board"):
        load_config(f)


# ---------------------------------------------------------------------------
# 6. File not found raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.toml")


# ---------------------------------------------------------------------------
# 7. Real site.toml (config/site.toml) is loadable
# ---------------------------------------------------------------------------

def test_real_site_toml_loads():
    repo_root = Path(__file__).parent.parent.parent
    site_toml = repo_root / "config" / "site.toml"
    cfg, sha = load_config(site_toml)
    assert isinstance(sha, str) and len(sha) == 64
    assert len(cfg["nodes"]) >= 2


# ---------------------------------------------------------------------------
# 8. node_id uniqueness
# ---------------------------------------------------------------------------

def test_duplicate_node_id_raises(tmp_path):
    bad = VALID_TOML.replace("node_id = 2", "node_id = 1")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="node_id"):
        load_config(f)


# ---------------------------------------------------------------------------
# 9. mic_spacing_m > 0
# ---------------------------------------------------------------------------

def test_mic_spacing_zero_raises(tmp_path):
    bad = VALID_TOML.replace("mic_spacing_m = 0.10", "mic_spacing_m = 0.0")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="mic_spacing_m"):
        load_config(f)


def test_mic_spacing_negative_raises(tmp_path):
    bad = VALID_TOML.replace("mic_spacing_m = 0.10", "mic_spacing_m = -0.1")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="mic_spacing_m"):
        load_config(f)


# ---------------------------------------------------------------------------
# 10. band_hz = two numbers with 0 < lo < hi
# ---------------------------------------------------------------------------

def test_band_hz_not_two_elements_raises(tmp_path):
    bad = VALID_TOML.replace("band_hz = [300, 4000]", "band_hz = [300]")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="band_hz"):
        load_config(f)


def test_band_hz_lo_not_positive_raises(tmp_path):
    bad = VALID_TOML.replace("band_hz = [300, 4000]", "band_hz = [0, 4000]")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="band_hz"):
        load_config(f)


def test_band_hz_lo_not_less_than_hi_raises(tmp_path):
    bad = VALID_TOML.replace("band_hz = [300, 4000]", "band_hz = [4000, 300]")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="band_hz"):
        load_config(f)


# ---------------------------------------------------------------------------
# 11. window_s > 0  and  hop_s > 0
# ---------------------------------------------------------------------------

def test_window_s_zero_raises(tmp_path):
    bad = VALID_TOML.replace("window_s = 1.0", "window_s = 0.0")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="window_s"):
        load_config(f)


def test_hop_s_zero_raises(tmp_path):
    bad = VALID_TOML.replace("hop_s = 0.5", "hop_s = 0.0")
    f = tmp_path / "site.toml"
    f.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="hop_s"):
        load_config(f)


# ---------------------------------------------------------------------------
# 12. Per-node defaults: tau0_s, channel_swap, mic_model
# ---------------------------------------------------------------------------

_MINIMAL_NODE_TOML = textwrap.dedent("""\
    convention_version = 1
    mic_spacing_m = 0.10
    band_hz = [300, 4000]
    window_s = 1.0
    hop_s = 0.5
    pair_tolerance_s = 0.1
    gate_min_confidence = 0.3
    gate_min_band_energy_db = -40.0
    min_crossing_deg = 30

    [[nodes]]
    node_id = 1
    usb_serial = "ABCD1234"
    board = "xiao"
    x_m = 0.0
    y_m = 0.0
    heading_deg = 0.0
""")


def test_node_defaults_applied(tmp_path):
    """When tau0_s / channel_swap / mic_model are absent, defaults are filled in."""
    f = tmp_path / "site.toml"
    f.write_text(_MINIMAL_NODE_TOML, encoding="utf-8")
    cfg, _ = load_config(f)
    node = cfg["nodes"][0]
    assert node["tau0_s"] == 0.0
    assert node["channel_swap"] is False
    assert node["mic_model"] == "INMP441"


# ---------------------------------------------------------------------------
# 13. phat_beta: default, valid value, out-of-range raises ConfigError (slice 2b)
# ---------------------------------------------------------------------------

def test_phat_beta_default_is_1(tmp_path):
    """When phat_beta is absent, load_config sets it to 1.0."""
    f = tmp_path / "site.toml"
    f.write_text(VALID_TOML, encoding="utf-8")
    cfg, _ = load_config(f)
    assert cfg["phat_beta"] == 1.0


def test_phat_beta_valid_value_loads(tmp_path):
    """phat_beta = 0.5 is accepted."""
    # Must insert before [[nodes]] so TOML parses it as a global key.
    toml = VALID_TOML.replace(
        "[[nodes]]", "phat_beta = 0.5\n\n[[nodes]]", 1
    )
    f = tmp_path / "site.toml"
    f.write_text(toml, encoding="utf-8")
    cfg, _ = load_config(f)
    assert cfg["phat_beta"] == 0.5


@pytest.mark.parametrize("bad_beta", [-0.1, 1.1])
def test_phat_beta_out_of_range_raises(tmp_path, bad_beta):
    """phat_beta outside [0, 1] must raise ConfigError."""
    # Must insert before [[nodes]] so TOML parses it as a global key.
    toml = VALID_TOML.replace(
        "[[nodes]]", f"phat_beta = {bad_beta}\n\n[[nodes]]", 1
    )
    f = tmp_path / "site.toml"
    f.write_text(toml, encoding="utf-8")
    with pytest.raises(ConfigError, match="phat_beta"):
        load_config(f)

