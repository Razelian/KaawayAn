"""
kaawayan.config
===============
Load and validate config/site.toml.

Returns (config_dict, sha256_hex_string).
SHA-256 is computed over the raw file bytes so it matches any external hash.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when the TOML file is present but violates schema."""


# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------

_GLOBAL_REQUIRED = [
    "convention_version",
    "mic_spacing_m",
    "band_hz",
    "window_s",
    "hop_s",
    "pair_tolerance_s",
    "gate_min_confidence",
    "gate_min_band_energy_db",
    "min_crossing_deg",
]

_NODE_REQUIRED = [
    "node_id",
    "usb_serial",
    "board",
    "x_m",
    "y_m",
    "heading_deg",
]

_VALID_BOARDS = {"xiao", "devkit"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Path | str) -> tuple[dict, str]:
    """
    Load *path* as a TOML config file.

    Returns
    -------
    (cfg, sha256_hex)
        cfg         – the parsed dict (identical to what tomllib returns)
        sha256_hex  – lowercase hex SHA-256 of the raw file bytes

    Raises
    ------
    FileNotFoundError   – if the file does not exist
    ConfigError         – if required keys are missing or values are invalid
    """
    path = Path(path)
    raw = path.read_bytes()   # raises FileNotFoundError if missing
    sha256 = hashlib.sha256(raw).hexdigest()

    cfg = tomllib.loads(raw.decode("utf-8"))

    # -- global keys ---------------------------------------------------------
    for key in _GLOBAL_REQUIRED:
        if key not in cfg:
            raise ConfigError(f"missing required global key: '{key}'")

    # -- global value constraints -------------------------------------------
    if cfg.get("mic_spacing_m", 1) <= 0:
        raise ConfigError("mic_spacing_m must be > 0")

    band = cfg.get("band_hz", [])
    if len(band) != 2 or not (0 < band[0] < band[1]):
        raise ConfigError("band_hz must be [lo, hi] with 0 < lo < hi")

    if cfg.get("window_s", 1) <= 0:
        raise ConfigError("window_s must be > 0")

    if cfg.get("hop_s", 1) <= 0:
        raise ConfigError("hop_s must be > 0")

    # phat_beta: optional, default 1.0, must be in [0, 1]
    cfg.setdefault("phat_beta", 1.0)
    phat_beta = cfg["phat_beta"]
    if not (0.0 <= phat_beta <= 1.0):
        raise ConfigError(
            f"phat_beta must be in [0, 1], got {phat_beta}"
        )

    # -- nodes ---------------------------------------------------------------
    nodes = cfg.get("nodes", [])
    if not nodes:
        raise ConfigError("config must define at least one node under [[nodes]]")

    seen_ids: set[int] = set()
    for idx, node in enumerate(nodes):
        for key in _NODE_REQUIRED:
            if key not in node:
                raise ConfigError(
                    f"node[{idx}] missing required key: '{key}'"
                )
        board = node["board"]
        if board not in _VALID_BOARDS:
            raise ConfigError(
                f"node[{idx}] board '{board}' is invalid; must be one of {sorted(_VALID_BOARDS)}"
            )
        nid = node["node_id"]
        if nid in seen_ids:
            raise ConfigError(f"duplicate node_id {nid}")
        seen_ids.add(nid)

        # Apply per-node defaults
        node.setdefault("tau0_s", 0.0)
        node.setdefault("channel_swap", False)
        node.setdefault("mic_model", "INMP441")

    return cfg, sha256
