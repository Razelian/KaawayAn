"""KaawayAn Console — Streamlit dashboard synced to database/events.db or database/demo.db.

Run from repo root:  streamlit run dashboard/dashboard.py
Read-only: never writes to events.db. Live region refreshes via
st.fragment(run_every=2); static templates are cached per process.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

# Ensure imports resolve cleanly regardless of cwd
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from dashboard.icons import replace_material_icons_with_svg
from dashboard.db import (
    all_system_status,
    confidence_series,
    distinct_nodes,
    drone_count,
    environment_series,
    event_count,
    get_system_status,
    latest_bearing,
    latest_bearings_by_node,
    latest_drone_event,
    latest_environment,
    latest_events,
    latest_localization,
    max_event_id,
    node_last_seen,
    seconds_since_write,
)

ASSETS = Path(__file__).parent / "ui_assets"
REFRESH_S = 2
ONLINE_STALE_S = 10.0

# Load precompiled local Tailwind CSS
_TAILWIND_PATH = Path(__file__).parent / "static" / "tailwind.min.css"
TAILWIND_CSS = _TAILWIND_PATH.read_text(encoding="utf-8") if _TAILWIND_PATH.exists() else ""


def _get_site_config() -> tuple[dict[str, Any] | None, str | None]:
    """Load config/site.toml using only stdlib tomllib.

    The dashboard must not depend on importing kaawayan and must not fall back to hardcoded nodes.
    """
    cfg_file = Path(__file__).resolve().parent.parent / "config" / "site.toml"
    if not cfg_file.exists():
        return None, f"Site config file not found: {cfg_file}"

    try:
        with open(cfg_file, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        return None, f"Failed to parse site.toml: {exc}"

    nodes = data.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        return None, f"Site config contains no valid 'nodes' list: {cfg_file}"

    for idx, n in enumerate(nodes):
        if not isinstance(n, dict):
            return None, f"Node #{idx} is not a valid table/dict"
        for req in ("node_id", "x_m", "y_m", "heading_deg"):
            if req not in n:
                return None, f"Node #{idx} (id={n.get('node_id')}) is missing required key '{req}'"

    return data, None


@st.cache_resource
def _load_template(name: str) -> str:
    tpl = (ASSETS / name / "code.html").read_text(encoding="utf-8")
    tpl = re.sub(r"<aside.*?</aside>", "", tpl, flags=re.S)
    tpl = re.sub(r'<header class="fixed top-0.*?</header>', "", tpl, flags=re.S)
    tpl = tpl.replace('<div class="pl-72">', "<div>").replace('<div class="pl-56">', "<div>")

    # Inject local compiled Tailwind CSS and system font fallbacks into <head>
    style_block = f"""
    <style>
      {TAILWIND_CSS}
      html, body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important; }}
      code, pre, .font-mono, [class*="data-mono"] {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; }}
    </style>
    """
    if "<head>" in tpl:
        tpl = tpl.replace("<head>", f"<head>{style_block}")
    elif "<head " in tpl:
        tpl = re.sub(r'<head[^>]*>', lambda m: m.group(0) + style_block, tpl, count=1)

    # Convert external Material Symbols icons to local inline SVG
    tpl = replace_material_icons_with_svg(tpl)
    return tpl


def _wall_time(epoch: float | None = None) -> str:
    """Format timestamp using the host's actual local timezone (no hardcoded PHT)."""
    if epoch is not None:
        dt = datetime.fromtimestamp(epoch).astimezone()
    else:
        dt = datetime.now().astimezone()
    tz_str = dt.strftime("%Z")
    if not tz_str:
        tz_str = dt.strftime("%z")
    return dt.strftime(f"%d %b %Y · %H:%M {tz_str}")


def _assert_replace(
    tpl: str, target: str, replacement: str, page_name: str, missing: list[str]
) -> str:
    """Replace target with replacement; if target is missing, record error."""
    if target not in tpl:
        missing.append(f"[{page_name}] Expected placeholder {target!r} was missing in template")
        return tpl
    return tpl.replace(target, replacement)


def _assert_sub(
    pattern: str | re.Pattern,
    replacement: str | Any,
    tpl: str,
    page_name: str,
    desc: str,
    missing: list[str],
    flags: int = 0,
) -> str:
    """Regex substitution asserting that at least one match occurred."""
    new_tpl, count = re.subn(pattern, replacement, tpl, flags=flags)
    if count == 0:
        missing.append(f"[{page_name}] Expected pattern {desc!r} did not match")
    return new_tpl


def _event_rows_html(rows: list[dict], limit: int = 8) -> str:
    out = []
    for i, r in enumerate(rows[:limit]):
        eid = f"DET-{r['event_id']:03d}"
        conf = float(r.get("model_confidence") or 0.0) * 100
        drone = bool(r.get("predicted_drone", 0))
        theta = r.get("theta_deg")
        angle_str = f"{theta:+.1f}°" if theta is not None else "—"
        t_row = r.get("t_wall")
        time_label = (
            datetime.fromtimestamp(t_row).astimezone().strftime("%H:%M:%S")
            if t_row
            else "—"
        )

        if i == 0:
            cls = (
                "event-row bg-surface-container text-primary font-semibold "
                "cursor-pointer transition-colors hover:bg-surface-container-high"
            )
            dot = '<span class="w-2 h-2 rounded-full bg-telemetry-active"></span>'
            id_cls, conf_cls = "font-semibold", "text-primary font-bold"
        else:
            cls = (
                "event-row bg-surface-container-lowest text-on-surface cursor-pointer "
                "transition-colors hover:bg-surface-container-low"
            )
            dot = '<span class="w-1.5 h-1.5 rounded-full bg-outline-variant"></span>'
            id_cls, conf_cls = "text-on-surface-variant font-medium", "text-on-surface"

        label = "DRONE" if drone else "no-drone"
        out.append(
            f'<tr class="{cls}" data-conf="{conf:.1f}%" data-event-id="{eid}" data-node="{r.get("node_id", 0)}">'
            f'<td class="py-3.5 px-space-lg flex items-center gap-space-xs {id_cls}">{dot}<span>{eid}</span></td>'
            f'<td class="py-3.5 px-space-md font-mono text-on-surface">{time_label}</td>'
            f'<td class="py-3.5 px-space-md text-on-surface">Node {r.get("node_id", "—")}</td>'
            f'<td class="py-3.5 px-space-md text-right {conf_cls}">{conf:.1f}%</td>'
            f'<td class="py-3.5 px-space-md text-center font-bold text-primary">{angle_str}</td>'
            f'<td class="py-3.5 px-space-lg text-right text-secondary">{label}</td>'
            f"</tr>"
        )
    if not out:
        out.append(
            '<tr><td class="py-3.5 px-space-lg text-secondary" colspan="6">'
            "No events yet — start the pipeline (python main.py).</td></tr>"
        )
    return "\n".join(out)


def _curve_path(
    series: list[tuple[float, float]],
    w: float = 920.0,
    x0: float = 60.0,
    y_top: float = 20.0,
    y_bot: float = 240.0,
) -> str:
    if not series:
        return ""
    n = len(series)
    pts = [
        (x0 + (w * i / max(n - 1, 1)), y_bot - float(c) * (y_bot - y_top))
        for i, (_, c) in enumerate(series)
    ]
    return f"M {pts[0][0]:.0f},{pts[0][1]:.0f} " + " ".join(f"L {x:.0f},{y:.0f}" for x, y in pts[1:])


def _render_radar_beam_svg(theta_deg: float) -> str:
    """Generate the SVG snippet for the radar beam at theta_deg clamped to [-60, +60] on screen

    while showing the unclamped angle consistently.
    """
    theta_clamped = max(-60.0, min(60.0, theta_deg))
    rad = math.radians(theta_clamped)

    x_beam = 230.0 + 205.0 * math.sin(rad)
    y_beam = 240.0 - 205.0 * math.cos(rad)

    x_box = x_beam + (8 if theta_clamped >= 0 else -62)
    y_box = y_beam - 14
    x_txt = x_box + 29
    y_txt = y_box + 16

    sign_char = "+" if theta_deg >= 0 else ""
    label = f"{sign_char}{theta_deg:.1f}°"

    return f"""
    <!-- Active Beam at {label} -->
    <line stroke="#063423" stroke-linecap="round" stroke-width="3.5" x1="230" x2="{x_beam:.1f}" y1="240" y2="{y_beam:.1f}"></line>
    <!-- Beam Target Endpoint / Dot -->
    <circle cx="{x_beam:.1f}" cy="{y_beam:.1f}" fill="#1F7A4D" r="7" stroke="#FAF9F6" stroke-width="2.5"></circle>
    <!-- Angle Callout Marker Badge -->
    <rect fill="#effdf4" height="24" rx="12" stroke="#1F7A4D" stroke-width="1.5" width="62" x="{x_box:.1f}" y="{y_box:.1f}"></rect>
    <text fill="#063423" font-family="ui-monospace, monospace" font-size="12" font-weight="700" text-anchor="middle" x="{x_txt:.1f}" y="{y_txt:.1f}">{label}</text>
    """


def _render_overview(
    rows: list[dict],
    total: int,
    drones: int,
    cfg: Optional[dict[str, Any]],
    cfg_err: Optional[str],
) -> None:
    tpl = _load_template("monitoring_overview")
    missing: list[str] = []

    # System status
    status = all_system_status()
    mode = status.get("mode", {}).get("value", "live")
    live_label = "SIMULATED" if mode == "demo" else "LIVE"

    tpl = _assert_replace(tpl, "DEMO DATA", live_label, "Overview", missing)
    tpl = _assert_replace(tpl, "19 Oct 2026 · 14:32 PHT", _wall_time(), "Overview", missing)

    now = time.time()
    cfg_nodes = (cfg.get("nodes", []) if cfg else [])
    node_seen = node_last_seen([n["node_id"] for n in cfg_nodes] if cfg_nodes else None)

    online_count = sum(
        1 for n in cfg_nodes if (now - node_seen.get(n["node_id"], 0.0)) < ONLINE_STALE_S
    )
    tpl = _assert_replace(tpl, "2 Active Units", f"{online_count} Active Units", "Overview", missing)

    # Dynamic Node Status Cards (built from cfg_nodes only)
    node_cards_html = []
    for n in cfg_nodes:
        nid = n["node_id"]
        lt = node_seen.get(nid)
        is_online = (lt is not None and (now - lt) < ONLINE_STALE_S)
        env_node = latest_environment(node_id=nid)
        has_bme = (env_node is not None)
        status_txt = "Online" if is_online else "Offline"
        badge_cls = "text-telemetry-active" if is_online else "text-outline"
        dot_cls = "bg-telemetry-active" if is_online else "bg-outline-variant"
        board_desc = n.get("board_desc", f"ESP32-S3 Acoustic Unit")
        conn_desc = ("Audio + BME280 connected" if has_bme else "Audio connected") if is_online else "Disconnected"
        node_cards_html.append(f"""
        <div class="flex items-center justify-between p-space-sm bg-surface-container-low rounded-lg transition-colors hover:bg-surface-container">
          <div class="flex items-center gap-space-md">
            <div class="w-9 h-9 rounded-full bg-primary-fixed flex items-center justify-center font-headline-md text-on-primary-fixed font-bold">
              {nid}
            </div>
            <div class="flex flex-col">
              <div class="flex items-center gap-space-xs">
                <span class="font-data-mono-md text-data-mono-md text-primary font-semibold">Node {nid}</span>
                <span class="font-data-mono-sm text-data-mono-sm text-secondary">{board_desc}</span>
              </div>
              <span class="font-body-sm text-body-sm text-on-surface-variant">{conn_desc}</span>
            </div>
          </div>
          <div class="flex items-center gap-1.5 px-space-sm py-space-xs rounded-full bg-surface-container-lowest shadow-xs {badge_cls} font-data-mono-sm text-data-mono-sm">
            <span class="w-2 h-2 rounded-full {dot_cls}"></span>
            <span>{status_txt}</span>
          </div>
        </div>
        """)
    if not node_cards_html:
        node_cards_html.append('<div class="p-space-sm text-secondary text-sm">No nodes configured in site.toml</div>')

    tpl = _assert_sub(
        r'<div class="flex flex-col gap-space-sm">\s*<!-- Node A Item -->.*?</div>\s*</div>\s*<!-- Recent Detections Table Card -->',
        '<div class="flex flex-col gap-space-sm">\n' + "\n".join(node_cards_html) + '\n</div>\n</div>\n<!-- Recent Detections Table Card -->',
        tpl,
        "Overview",
        "Node Status Cards",
        missing,
        flags=re.S,
    )

    # Dynamic Angle Cards (built from cfg_nodes only)
    node_ids = [n["node_id"] for n in cfg_nodes]
    b_by_node = latest_bearings_by_node(node_ids) if node_ids else {}

    angle_cards_html = []
    for n in cfg_nodes:
        nid = n["node_id"]
        br = b_by_node.get(nid) or latest_bearing(node_id=nid)
        br_t = float(br["t_wall"]) if (br and br.get("t_wall")) else None
        br_fresh = (br and br.get("theta_deg") is not None and br_t is not None and (now - br_t) < Config.loc_max_age_s)
        if br_fresh:
            th = float(br["theta_deg"])
            theta_str = f"{th:+.1f}°"
            status_str = "Active"
        else:
            theta_str = "Standby"
            status_str = "Standby"
        angle_cards_html.append(f"""
        <div class="flex flex-col">
          <span class="font-label-sm text-label-sm text-secondary uppercase">Angle · Node {nid}</span>
          <span class="font-headline-xl text-headline-xl text-primary font-bold tracking-tight">{theta_str}</span>
          <span class="font-data-mono-sm text-data-mono-sm text-on-surface-variant">{status_str}</span>
        </div>
        """)
    if not angle_cards_html:
        angle_cards_html.append("""
        <div class="flex flex-col">
          <span class="font-label-sm text-label-sm text-secondary uppercase">Angle</span>
          <span class="font-headline-xl text-headline-xl text-primary font-bold tracking-tight">—</span>
          <span class="font-data-mono-sm text-data-mono-sm text-on-surface-variant">No nodes</span>
        </div>
        """)

    tpl = _assert_sub(
        r'(<!-- Key Metrics Cluster -->\s*<div[^>]*>\s*<div class="flex flex-col">\s*<span class="font-label-sm[^>]*>Confidence</span>.*?</div>\s*).*?(</div>\s*<!-- Time & Event Identifier -->)',
        r'\g<1>' + "\n".join(angle_cards_html) + r'\g<2>',
        tpl,
        "Overview",
        "Angle cards",
        missing,
        flags=re.S,
    )

    # Dynamic Radar Arc Beam (primary node: cfg_nodes[0] if configured)
    primary_id = cfg_nodes[0]["node_id"] if cfg_nodes else None
    b_prim = (b_by_node.get(primary_id) or latest_bearing(node_id=primary_id)) if primary_id else None
    b_prim_t = float(b_prim["t_wall"]) if (b_prim and b_prim.get("t_wall")) else None
    b_prim_fresh = (b_prim and b_prim.get("theta_deg") is not None and b_prim_t is not None and (now - b_prim_t) < Config.loc_max_age_s)

    if b_prim_fresh:
        th_prim = float(b_prim["theta_deg"])
        dynamic_beam = _render_radar_beam_svg(th_prim)
    else:
        dynamic_beam = ""  # Standby shows no beam, not a 0° beam!

    beam_regex = re.compile(
        r'<!-- Active Beam at \+28 degrees -->.*?<!-- Origin Base Station Node A -->', re.S
    )
    tpl = _assert_sub(
        beam_regex,
        f"{dynamic_beam}\n<!-- Origin Base Station Node A -->",
        tpl,
        "Overview",
        "Radar Beam",
        missing,
    )

    # Hero card: latest event with predicted_drone = 1
    drone_hero = latest_drone_event()
    if drone_hero is not None:
        conf_pct = float(drone_hero.get("model_confidence") or 0.0) * 100
        hero_conf_str = f"{conf_pct:.1f}"
        hero_eid = f"DET-{drone_hero.get('event_id', 0):03d}"
        hero_time = _wall_time(drone_hero.get("t_wall"))
        hero_title = "Drone-like audio detected"
        hero_badge = "Confidence High" if conf_pct >= Config.drone_high_confidence_threshold * 100 else "Confidence Match"
    else:
        hero_conf_str = "—"
        hero_eid = "—"
        hero_time = "—"
        hero_title = "No drone detected"
        hero_badge = "Standby"

    tpl = _assert_replace(tpl, ">94.2<", f">{hero_conf_str}<", "Overview", missing)
    tpl = _assert_replace(tpl, "+3.1%", "—", "Overview", missing)
    tpl = _assert_replace(tpl, ">DET-024<", f">{hero_eid}<", "Overview", missing)
    tpl = _assert_replace(tpl, "14:32:08 PHT", hero_time, "Overview", missing)
    tpl = _assert_replace(tpl, "Drone-like audio detected", hero_title, "Overview", missing)
    tpl = _assert_replace(tpl, "Confidence High", hero_badge, "Overview", missing)

    # Environmental sensor cards (strictly bme_status == 0)
    env = latest_environment()
    if env and env.get("temp_c") is not None:
        tpl = _assert_replace(tpl, ">29.4<", f">{env['temp_c']:.1f}<", "Overview", missing)
        tpl = _assert_replace(tpl, ">72.1<", f">{env['humidity_pct']:.1f}<", "Overview", missing)
    else:
        tpl = _assert_replace(tpl, ">29.4<", ">—<", "Overview", missing)
        tpl = _assert_replace(tpl, ">72.1<", ">—<", "Overview", missing)

    # Pressure is not transmitted in protocol v1
    tpl = _assert_replace(tpl, "1008.6", "n/a", "Overview", missing)
    tpl = _assert_replace(tpl, "48 kHz / 24-bit", "48 kHz / 16-bit", "Overview", missing)

    # Update timing text
    since_w = seconds_since_write(now)
    updated_str = f"updated {int(since_w)} s ago" if since_w is not None else "updated — ago"
    tpl = _assert_replace(tpl, "updated 2 s ago", updated_str, "Overview", missing)

    # Recent Detections table in Overview (each row shows its own t_wall time)
    overview_table_rows = []
    if rows:
        for r in rows[:4]:
            eid = f"DET-{r.get('event_id', 0):03d}"
            conf = float(r.get("model_confidence") or 0.0) * 100
            th = r.get("theta_deg")
            ang = f"{th:+.1f}°" if th is not None else "—"
            t_row = r.get("t_wall")
            time_label = (
                datetime.fromtimestamp(t_row).astimezone().strftime("%H:%M:%S")
                if t_row
                else "—"
            )
            overview_table_rows.append(
                f'<tr class="hover:bg-surface-container-low transition-colors group">'
                f'<td class="py-space-sm px-space-sm font-semibold text-primary">{time_label}</td>'
                f'<td class="py-space-sm px-space-sm"><span class="text-telemetry-active font-semibold">{conf:.1f}%</span></td>'
                f'<td class="py-space-sm px-space-sm text-center font-bold text-primary">{ang}</td>'
                f'<td class="py-space-sm px-space-sm text-right"><span class="bg-surface-container px-space-xs py-[1px] rounded text-primary">{eid}</span></td>'
                f'</tr>'
            )
    else:
        overview_table_rows.append(
            '<tr><td class="py-space-sm px-space-sm text-secondary" colspan="4">No detections recorded yet.</td></tr>'
        )
    tpl = _assert_sub(
        r'(<tbody[^>]*>).*?(</tbody>)',
        r'\g<1>' + "\n".join(overview_table_rows) + r'\g<2>',
        tpl,
        "Overview",
        "Recent detections table body",
        missing,
        flags=re.S,
    )

    if missing:
        st.error("TEMPLATE BINDING ERROR: " + " | ".join(missing))

    if cfg_err:
        err_banner = f"""
        <div style="background-color: #FEE2E2; border: 2px solid #DC2626; color: #991B1B; padding: 16px; border-radius: 12px; font-weight: bold; margin-bottom: 20px;">
          CONFIGURATION WARNING: {cfg_err}. Node angles and radar beam are uncalibrated.
        </div>
        """
        tpl = tpl.replace("<main ", f"{err_banner}<main ")

    st.session_state[f"render_{('Overview', st.session_state.get('anchor'))}"] = tpl
    components.html(tpl, height=1200, scrolling=True)


def _render_detection_page(rows: list[dict], cfg: Optional[dict[str, Any]] = None) -> None:
    tpl = _load_template("detection_history")
    missing: list[str] = []

    status = all_system_status()
    mode = status.get("mode", {}).get("value", "live")
    live_label = "SIMULATED" if mode == "demo" else "LIVE"

    tpl = _assert_replace(tpl, "DEMO DATA", live_label, "Detections", missing)
    tpl = _assert_replace(tpl, "19 Oct 2026 · 14:32 PHT", _wall_time(), "Detections", missing)
    tpl = _assert_replace(tpl, "ACOUSTIC ARRAY V2.4 LOCK", "ACOUSTIC SENSING ACTIVE", "Detections", missing)

    thresh_pct = int(Config.drone_detection_threshold * 100)
    tpl = _assert_replace(tpl, "80% Review Threshold", f"{thresh_pct}% Review Threshold", "Detections", missing)
    tpl = _assert_replace(tpl, "80% review threshold", f"{thresh_pct}% review threshold", "Detections", missing)
    tpl = _assert_replace(tpl, "≥ 80%", f"≥ {thresh_pct}%", "Detections", missing)

    # Update Table Header (thead) to 6 columns: Event, Time, Node, Confidence, Angle, Classification
    new_thead = """<thead>
<tr class="bg-surface-container-low/70 text-secondary font-label-sm text-label-sm uppercase tracking-wider">
<th class="py-space-sm px-space-lg font-semibold">Event</th>
<th class="py-space-sm px-space-md font-semibold">Time</th>
<th class="py-space-sm px-space-md font-semibold">Node</th>
<th class="py-space-sm px-space-md font-semibold text-right">Confidence</th>
<th class="py-space-sm px-space-md font-semibold text-center">Angle</th>
<th class="py-space-sm px-space-lg font-semibold text-right">Classification</th>
</tr>
</thead>"""
    tpl = _assert_sub(
        r'<thead>\s*<tr[^>]*>.*?</tr>\s*</thead>',
        new_thead,
        tpl,
        "Detections",
        "Table Header (thead)",
        missing,
        flags=re.S,
    )

    # Dynamic Detail Angle Cards (built from cfg_nodes only)
    cfg_nodes = cfg.get("nodes", []) if cfg else []
    node_ids = [n["node_id"] for n in cfg_nodes]
    b_by_node = latest_bearings_by_node(node_ids) if node_ids else {}
    now = time.time()

    detail_cards = []
    for n in cfg_nodes:
        nid = n["node_id"]
        br = b_by_node.get(nid) or latest_bearing(node_id=nid)
        br_t = float(br["t_wall"]) if (br and br.get("t_wall")) else None
        br_fresh = (br and br.get("theta_deg") is not None and br_t is not None and (now - br_t) < Config.loc_max_age_s)
        ang_str = f"{float(br['theta_deg']):+.1f}°" if br_fresh else "—"
        detail_cards.append(f"""
        <div class="bg-surface-container-low p-space-md rounded-xl flex flex-col items-center justify-center text-center">
          <span class="font-headline-lg text-headline-lg font-bold text-primary tracking-tight" id="detail-angle-{nid}">{ang_str}</span>
          <span class="font-label-sm text-label-sm uppercase tracking-wider text-secondary mt-0.5">Node {nid} angle</span>
        </div>
        """)

    if not detail_cards:
        detail_cards.append("""
        <div class="bg-surface-container-low p-space-md rounded-xl flex flex-col items-center justify-center text-center">
          <span class="font-headline-lg text-headline-lg font-bold text-primary tracking-tight">—</span>
          <span class="font-label-sm text-label-sm uppercase tracking-wider text-secondary mt-0.5">No nodes</span>
        </div>
        """)

    cols_cls = f"grid-cols-{min(max(len(detail_cards), 1), 4)}"
    detail_angles_html = f'<div class="grid {cols_cls} gap-space-sm">\n' + "\n".join(detail_cards) + "\n</div>"
    tpl = _assert_sub(
        r'<div class="grid grid-cols-2 gap-space-sm">\s*<div[^>]*>\s*<span[^>]*id="detail-angle-a".*?</div>\s*</div>\s*</div>',
        detail_angles_html,
        tpl,
        "Detections",
        "Detail angle cards",
        missing,
        flags=re.S,
    )

    # Filtered count
    tpl = _assert_replace(tpl, "5 Filtered", f"{len(rows)} Events", "Detections", missing)

    # Bind or remove Triangulation active
    loc = latest_localization()
    loc_t = float(loc["t_wall"]) if (loc and loc.get("t_wall")) else None
    is_loc_fresh = (loc and loc.get("status") == "ok" and loc_t is not None and (now - loc_t) < Config.loc_max_age_s)
    tri_status = "Triangulation active" if is_loc_fresh else "Triangulation standby"
    tpl = _assert_replace(tpl, "Triangulation active", tri_status, "Detections", missing)

    # Bind or remove VALIDATED and detail panel
    series = confidence_series(120)
    if rows:
        latest = rows[0]
        conf_pct = float(latest.get("model_confidence") or 0.0) * 100
        eid = f"DET-{latest.get('event_id', 0):03d}"
        tpl = _assert_replace(tpl, ">DET-024<", f">{eid}<", "Detections", missing)
        tpl = _assert_replace(tpl, ">94.2%<", f">{conf_pct:.1f}%<", "Detections", missing)
        tpl = tpl.replace('data-conf="94.2%"', f'data-conf="{conf_pct:.1f}%"')
        tpl = tpl.replace('data-event-id="DET-024"', f'data-event-id="{eid}"')

        is_drone = bool(latest.get("predicted_drone", 0))
        val_label = "DRONE DETECTED" if is_drone else "AMBIENT / NO-DRONE"
        tpl = _assert_replace(tpl, "VALIDATED", val_label, "Detections", missing)

        # Detail panel time
        t_w = latest.get("t_wall")
        detail_time = _wall_time(t_w) if t_w else "—"
        tpl = _assert_replace(tpl, "19 Oct 2026 · 14:32:08 PHT", detail_time, "Detections", missing)

        pmr = latest.get("peak_median_ratio")
        pmr_str = f"PMR: {pmr:.1f}" if pmr is not None else "—"
        tpl = _assert_replace(tpl, "SNR: 18.4 dB", pmr_str, "Detections", missing)
        tpl = _assert_replace(tpl, "T_SPAN: 6.00s", "T_SPAN: 1.00s", "Detections", missing)

        class_str = "Classification: Drone detected" if is_drone else "Classification: Ambient / no-drone"
        tpl = _assert_replace(tpl, "Payload: Quad-rotor signature", class_str, "Detections", missing)
        tpl = _assert_replace(tpl, "94.2% Peak", f"{conf_pct:.1f}% Peak", "Detections", missing)
    else:
        tpl = _assert_replace(tpl, "VALIDATED", "STANDBY", "Detections", missing)
        tpl = _assert_replace(tpl, ">DET-024<", ">—<", "Detections", missing)
        tpl = _assert_replace(tpl, ">94.2%<", ">—<", "Detections", missing)
        tpl = _assert_replace(tpl, "19 Oct 2026 · 14:32:08 PHT", "—", "Detections", missing)
        tpl = _assert_replace(tpl, "SNR: 18.4 dB", "—", "Detections", missing)
        tpl = _assert_replace(tpl, "T_SPAN: 6.00s", "T_SPAN: 1.00s", "Detections", missing)
        tpl = _assert_replace(tpl, "Payload: Quad-rotor signature", "Classification: —", "Detections", missing)
        tpl = _assert_replace(tpl, "94.2% Peak", "— Peak", "Detections", missing)

    curve = _curve_path([(i, c) for i, c in series])
    if curve:
        tpl = _assert_sub(
            r'<path d="M 60,200.*?id="event-curve"',
            f'<path d="{curve}" id="event-curve"',
            tpl,
            "Detections",
            "Event curve path",
            missing,
            flags=re.S,
        )

    # Bind real event rows (6 columns)
    live_rows = _event_rows_html(rows)
    tpl = _assert_sub(
        r'(<tbody[^>]*id="detection-table-body"[^>]*>).*?(</tbody>)',
        r'\g<1>' + live_rows + r'\g<2>',
        tpl,
        "Detections",
        "Detection table body",
        missing,
        flags=re.S,
    )

    if missing:
        st.error("TEMPLATE BINDING ERROR: " + " | ".join(missing))

    st.session_state[f"render_{('Detections', st.session_state.get('anchor'))}"] = tpl
    components.html(tpl, height=1100, scrolling=True)



def _render_localization_page(cfg: Optional[dict[str, Any]], cfg_err: Optional[str]) -> None:
    """Render the 2D arena floor-plan and multi-node triangulation map.

    Strictly complies with geometry rules: identical px/m scale on both axes, clipped rays,
    standby status, and offline rendering without external network requests.
    """
    if cfg_err or not cfg:
        st.error(f"CONFIGURATION ERROR: {cfg_err or 'Invalid site.toml'}")
        error_html = f"""
        <div style="background-color: #FEE2E2; border: 2px solid #DC2626; color: #991B1B; padding: 24px; border-radius: 12px; font-weight: bold; width: 100%; text-align: center;">
          CONFIGURATION ERROR: Site configuration (config/site.toml) is missing or invalid: {cfg_err}.<br/>
          Localization map is disabled because node positions cannot be trusted.
        </div>
        """
        components.html(error_html, height=200)
        return

    nodes = cfg.get("nodes", [])

    # Arena bounds from site.toml [arena] or computed from node positions + 20% margin
    if cfg.get("arena"):
        arena = cfg["arena"]
        x_min = float(arena.get("x_min", -1.0))
        x_max = float(arena.get("x_max", 5.0))
        y_min = float(arena.get("y_min", -0.5))
        y_max = float(arena.get("y_max", 5.5))
    else:
        xs = [float(n["x_m"]) for n in nodes] or [0.0, 3.0]
        ys = [float(n["y_m"]) for n in nodes] or [0.0, 0.0]
        span_x_raw = max(max(xs) - min(xs), 2.0)
        span_y_raw = max(max(ys) - min(ys), 3.0)
        margin_x = span_x_raw * 0.20
        margin_y = span_y_raw * 0.20
        x_min = min(xs) - margin_x
        x_max = max(xs) + margin_x
        y_min = min(ys) - margin_y
        y_max = max(ys) + span_y_raw + margin_y

    w_m = max(0.1, x_max - x_min)
    h_m = max(0.1, y_max - y_min)

    # Fit arena inside SVG viewbox with identical scale on both X and Y
    w_svg, h_svg = 640, 520
    margin_x = 40.0
    margin_y = 40.0
    avail_w = w_svg - 2 * margin_x
    avail_h = h_svg - 2 * margin_y

    scale = min(avail_w / w_m, avail_h / h_m)
    arena_w_px = w_m * scale
    arena_h_px = h_m * scale

    offset_x = (w_svg - arena_w_px) / 2.0
    offset_y = (h_svg - arena_h_px) / 2.0

    def to_svg(xm: float, ym: float) -> tuple[float, float]:
        xs = offset_x + (xm - x_min) * scale
        ys = offset_y + (y_max - ym) * scale  # Y axis inverted in SVG
        return xs, ys

    loc = latest_localization()
    now = time.time()
    loc_t_wall = float(loc["t_wall"]) if (loc and loc.get("t_wall") is not None) else None
    age_s = (now - loc_t_wall) if loc_t_wall is not None else None
    raw_status = str(loc.get("status") if loc else "none").lower()
    reason = str(loc.get("reason") if loc else "")
    crossing = loc.get("crossing_deg") if loc else None
    dead_nodes_raw = loc.get("dead_nodes") if loc else None
    dead_nodes = []
    if dead_nodes_raw:
        if isinstance(dead_nodes_raw, list):
            dead_nodes = dead_nodes_raw
        elif isinstance(dead_nodes_raw, str):
            try:
                dead_nodes = json.loads(dead_nodes_raw)
            except Exception:
                dead_nodes = []

    # Gating: status == "ok" AND age < Config.loc_max_age_s
    is_fresh_ok = (raw_status == "ok") and (age_s is not None and age_s < Config.loc_max_age_s)
    x_drone = float(loc["x_m"]) if (loc and is_fresh_ok and loc.get("x_m") is not None) else None
    y_drone = float(loc["y_m"]) if (loc and is_fresh_ok and loc.get("y_m") is not None) else None
    drone_recent = (loc.get("drone_recent") == 1) if loc else False

    # Status labels & colors
    if is_fresh_ok:
        if drone_recent:
            status_label = "DRONE POSITION LOCKED"
            status_badge_color = "#1F7A4D"
            dot_color = "#1F7A4D"
            dot_label = f"DRONE ({x_drone:.2f}m, {y_drone:.2f}m)"
        else:
            status_label = "SOUND SOURCE (not classified as drone)"
            status_badge_color = "#475569"
            dot_color = "#64748B"
            dot_label = f"SOUND SOURCE ({x_drone:.2f}m, {y_drone:.2f}m)"
    elif raw_status == "ok" and not is_fresh_ok:
        status_label = f"STALE FIX ({age_s:.1f}s ago)" if age_s is not None else "STALE FIX"
        status_badge_color = "#D97706"
    elif raw_status == "degraded":
        status_label = "DEGRADED"
        status_badge_color = "#D97706"
    elif raw_status == "uncertain":
        status_label = "UNCERTAIN"
        status_badge_color = "#D97706"
    else:
        status_label = "NO FIX"
        status_badge_color = "#64748B"

    # Drone dot SVG snippet
    if x_drone is not None and y_drone is not None:
        d_svg = to_svg(x_drone, y_drone)
        pulse = f"""
        <circle cx="{d_svg[0]:.1f}" cy="{d_svg[1]:.1f}" r="16" fill="{dot_color}" opacity="0.2">
            <animate attributeName="r" values="12;22;12" dur="2s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.4;0.05;0.4" dur="2s" repeatCount="indefinite"/>
        </circle>
        """ if drone_recent else ""
        drone_svg = f"""
        {pulse}
        <circle cx="{d_svg[0]:.1f}" cy="{d_svg[1]:.1f}" r="7" fill="{dot_color}" stroke="#FAF9F6" stroke-width="2.5"></circle>
        <text x="{d_svg[0] + 14:.1f}" y="{d_svg[1] + 4:.1f}" fill="{dot_color}" font-family="ui-monospace, monospace" font-size="11" font-weight="700">
            {dot_label}
        </text>
        """
    else:
        drone_svg = ""

    # Per-node bearing rays and node markers
    target_node_ids = [n["node_id"] for n in nodes]
    bearings_by_node = latest_bearings_by_node(target_node_ids)
    node_seen = node_last_seen(target_node_ids)

    rays_svg = []
    nodes_svg = []
    legend_items = []

    ray_colors = ["#1F7A4D", "#2563EB", "#7C3AED", "#D97706"]
    node_colors = ["#063423", "#1E3A8A", "#5B21B6", "#92400E"]

    for idx, node in enumerate(nodes):
        nid = node["node_id"]
        xn, yn = float(node["x_m"]), float(node["y_m"])
        heading = float(node["heading_deg"])
        board = node.get("board", "node")
        p_svg = to_svg(xn, yn)

        r_color = ray_colors[idx % len(ray_colors)]
        n_color = node_colors[idx % len(node_colors)]

        br = bearings_by_node.get(nid)
        br_t_wall = float(br["t_wall"]) if (br and br.get("t_wall")) else None
        br_age = (now - br_t_wall) if br_t_wall is not None else 999.0
        theta = float(br["theta_deg"]) if (br and br.get("theta_deg") is not None) else None
        gate_ok = bool(br.get("gate_ok", True)) if br else False

        is_dead = (nid in dead_nodes)
        is_fresh = (br_age < Config.loc_max_age_s)

        # Draw ray only if bearing is fresh and node is not dead; non-gated bearings dashed and only briefly (< 1s)
        if theta is not None and not is_dead and is_fresh and (gate_ok or br_age < 1.0):
            az = math.radians(heading + theta)
            dx = math.cos(az)
            dy = math.sin(az)

            # Clip ray strictly to arena box [x_min, x_max] x [y_min, y_max]
            t_candidates = []
            if dx > 1e-6:
                t = (x_max - xn) / dx
                if t > 0:
                    t_candidates.append(t)
            elif dx < -1e-6:
                t = (x_min - xn) / dx
                if t > 0:
                    t_candidates.append(t)

            if dy > 1e-6:
                t = (y_max - yn) / dy
                if t > 0:
                    t_candidates.append(t)
            elif dy < -1e-6:
                t = (y_min - yn) / dy
                if t > 0:
                    t_candidates.append(t)

            t_clip = min(t_candidates) if t_candidates else 5.0
            x_end = xn + t_clip * dx
            y_end = yn + t_clip * dy
            r_end = to_svg(x_end, y_end)

            dash = "" if gate_ok else 'stroke-dasharray="6 4" opacity="0.6"'
            rays_svg.append(
                f'<line x1="{p_svg[0]:.1f}" y1="{p_svg[1]:.1f}" x2="{r_end[0]:.1f}" y2="{r_end[1]:.1f}" '
                f'stroke="{r_color}" stroke-width="2.5" {dash} stroke-linecap="round"></line>'
            )
            theta_str = f"{theta:+.1f}°"
        else:
            theta_str = "standby"

        lt = node_seen.get(nid)
        dt = (now - lt) if lt is not None else None
        if is_dead:
            badge_node = "DEAD"
            marker_color = "#DC2626"
        elif dt is not None and dt < ONLINE_STALE_S:
            badge_node = "Online"
            marker_color = n_color
        else:
            badge_node = f"Stale ({dt:.0f}s)" if dt is not None else "No data"
            marker_color = "#94A3B8"

        nodes_svg.append(f"""
        <circle cx="{p_svg[0]:.1f}" cy="{p_svg[1]:.1f}" r="12" fill="{marker_color}"></circle>
        <circle cx="{p_svg[0]:.1f}" cy="{p_svg[1]:.1f}" r="5" fill="#FAF9F6"></circle>
        <text x="{p_svg[0]:.1f}" y="{p_svg[1] + 24:.1f}" fill="{marker_color}" font-family="ui-monospace, monospace" font-size="11" font-weight="700" text-anchor="middle">
          NODE {nid} ({theta_str})
        </text>
        """)

        legend_items.append(f"""
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full" style="background-color: {marker_color};"></span>
          <span>Node {nid} ({board} @ {xn:.1f},{yn:.1f}m · {badge_node})</span>
        </div>
        """)

    legend_items.append("""
    <div class="flex items-center gap-2">
      <span class="w-3 h-3 rounded-full bg-[#1F7A4D]"></span>
      <span>Triangulated Fix</span>
    </div>
    """)

    sub_status_txt = reason if reason else (status_label if not is_fresh_ok else "Tracking Active")
    xy_display = f"({x_drone:.2f}m, {y_drone:.2f}m)" if is_fresh_ok else sub_status_txt

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <style>
        {TAILWIND_CSS}
        body {{ background-color: #FAF9F6; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        code, pre, .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
      </style>
    </head>
    <body class="p-6">
      <div class="max-w-[1400px] mx-auto flex flex-col gap-6">
        <!-- Top Metrics Header -->
        <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div class="flex items-center gap-2 mb-1">
              <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold tracking-wider uppercase text-white" style="background-color: {status_badge_color};">
                {status_label}
              </span>
              <span class="text-xs font-mono text-gray-500">REALTIME 2D MULTI-NODE ACOUSTIC TRACKER</span>
            </div>
            <h1 class="text-2xl font-bold text-[#063423]">Field Arena Floor Plan & Triangulation</h1>
          </div>
          <div class="flex items-center gap-4 bg-white px-4 py-2 rounded-xl shadow-sm border border-[#E2E6DF]">
            <div>
              <div class="text-[11px] uppercase tracking-wider text-gray-500">Crossing Angle</div>
              <div class="text-lg font-bold font-mono text-[#063423]">{f"{crossing:.1f}°" if crossing else "—"}</div>
            </div>
            <div class="h-8 w-px bg-gray-200"></div>
            <div>
              <div class="text-[11px] uppercase tracking-wider text-gray-500">Estimated (X, Y)</div>
              <div class="text-lg font-bold font-mono text-[#063423]">{xy_display}</div>
            </div>
          </div>
        </div>

        <!-- Floor Plan Graphic Card -->
        <div class="bg-white rounded-2xl shadow-sm border border-[#E2E6DF] p-6 flex flex-col items-center">
          <svg class="w-full max-w-[800px] h-auto overflow-visible select-none" viewBox="0 0 {w_svg} {h_svg}">
            <!-- Background Arena Rect (clamped and centered with identical scale) -->
            <rect x="{offset_x:.1f}" y="{offset_y:.1f}" width="{arena_w_px:.1f}" height="{arena_h_px:.1f}" fill="#EFFDF4" rx="12" stroke="#E2E6DF" stroke-width="1.5"></rect>

            <!-- Coordinate Grid Origin -->
            <line x1="{offset_x:.1f}" y1="{offset_y + arena_h_px:.1f}" x2="{offset_x + arena_w_px:.1f}" y2="{offset_y + arena_h_px:.1f}" stroke="#CBD5E1" stroke-width="1.5"></line>
            <line x1="{offset_x:.1f}" y1="{offset_y:.1f}" x2="{offset_x:.1f}" y2="{offset_y + arena_h_px:.1f}" stroke="#CBD5E1" stroke-width="1.5"></line>

            <!-- Bearing Rays -->
            {"".join(rays_svg)}

            <!-- Nodes -->
            {"".join(nodes_svg)}

            <!-- Intersected Drone Point -->
            {drone_svg}
          </svg>

          <!-- Legend -->
          <div class="flex flex-wrap items-center justify-center gap-6 mt-4 pt-4 border-t border-gray-100 text-xs font-mono text-gray-600">
            {"".join(legend_items)}
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    components.html(html_content, height=750, scrolling=True)


def _render_environment(cfg: Optional[dict[str, Any]]) -> None:
    tpl = _load_template("environmental_conditions")
    missing: list[str] = []

    cfg_nodes = cfg.get("nodes", []) if cfg else []
    now = time.time()
    node_seen = node_last_seen([n["node_id"] for n in cfg_nodes] if cfg_nodes else None)

    # Dynamic Per-node Telemetry Cards (built from cfg_nodes only)
    env_node_cards = []
    last_env_t = None
    for n in cfg_nodes:
        nid = n["node_id"]
        lt = node_seen.get(nid)
        is_online = (lt is not None and (now - lt) < ONLINE_STALE_S)
        status_txt = "Online" if is_online else "Offline"
        status_color = "text-telemetry-active" if is_online else "text-outline"
        dot_color = "bg-telemetry-active" if is_online else "bg-outline-variant"
        board_desc = n.get("board_desc", n.get("board", "ESP32 Acoustic Unit"))

        env_n = latest_environment(node_id=nid)
        if env_n and env_n.get("t_wall"):
            last_env_t = max(last_env_t or 0.0, float(env_n["t_wall"]))

        temp_str = f"{env_n['temp_c']:.1f}" if env_n and env_n.get("temp_c") is not None else "—"
        rh_str = f"{env_n['humidity_pct']:.1f}" if env_n and env_n.get("humidity_pct") is not None else "—"

        env_node_cards.append(f"""
        <!-- Node {nid} Telemetry Card -->
        <div class="bg-surface-container-lowest rounded-xl p-space-lg shadow-[0_1px_3px_rgba(23,35,30,0.04)] flex flex-col md:flex-row md:items-center justify-between gap-space-lg">
          <div class="flex items-center gap-space-md min-w-[240px]">
            <div class="w-12 h-12 rounded-xl bg-surface-container flex items-center justify-center font-headline-md text-headline-md text-primary font-bold">{nid}</div>
            <div class="flex flex-col">
              <div class="flex items-center gap-space-sm">
                <span class="font-headline-md text-headline-md text-on-surface font-semibold">Node {nid}</span>
                <div class="flex items-center gap-space-xs bg-surface-container-low {status_color} px-space-sm py-[2px] rounded-full font-data-mono-sm text-data-mono-sm">
                  <span class="w-2 h-2 rounded-full {dot_color}"></span>
                  <span>{status_txt}</span>
                </div>
              </div>
              <span class="font-data-mono-sm text-data-mono-sm text-secondary tracking-wider">{board_desc}</span>
            </div>
          </div>
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-space-lg flex-1 border-t md:border-t-0 md:border-l border-border-subtle pt-space-md md:pt-0 md:pl-space-lg">
            <div class="flex items-center gap-space-md">
              <div class="w-10 h-10 rounded-lg bg-surface-container flex items-center justify-center">
                <svg class="text-[24px] text-telemetry-active inline-block align-middle" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M15 13V5c0-1.66-1.34-3-3-3S9 3.34 9 5v8c-1.21.91-2 2.37-2 4 0 2.76 2.24 5 5 5s5-2.24 5-5c0-1.63-.79-3.09-2-4zm-4-8c0-.55.45-1 1-1s1 .45 1 1h-1v1h1v2h-1v1h1v2h-2V5z"/></svg>
              </div>
              <div class="flex flex-col">
                <span class="font-label-sm text-label-sm uppercase tracking-wider text-secondary">Temperature</span>
                <div class="flex items-baseline gap-1">
                  <span class="font-data-mono-lg text-headline-md text-on-surface font-bold">{temp_str}</span>
                  <span class="font-body-md text-body-md text-secondary font-medium">°C</span>
                </div>
              </div>
            </div>
            <div class="flex items-center gap-space-md">
              <div class="w-10 h-10 rounded-lg bg-surface-container flex items-center justify-center">
                <svg class="text-[24px] text-telemetry-active inline-block align-middle" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
              </div>
              <div class="flex flex-col">
                <span class="font-label-sm text-label-sm uppercase tracking-wider text-secondary">Humidity</span>
                <div class="flex items-baseline gap-1">
                  <span class="font-data-mono-lg text-headline-md text-on-surface font-bold">{rh_str}</span>
                  <span class="font-body-md text-body-md text-secondary font-medium">% RH</span>
                </div>
              </div>
            </div>
            <div class="flex items-center gap-space-md">
              <div class="w-10 h-10 rounded-lg bg-surface-container flex items-center justify-center">
                <svg class="text-[24px] text-telemetry-active inline-block align-middle" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M20.38 8.57l-1.23 1.85a8 8 0 0 1-.22 7.58H5.07A8 8 0 0 1 15.58 6.85l1.85-1.23A10 10 0 0 0 2 12c0 5.52 4.48 10 10 10s10-4.48 10-10c0-1.22-.22-2.39-.62-3.43zM10.59 15.41a2 2 0 0 0 2.83 0l5.66-8.49-8.49 5.66a2 2 0 0 0 0 2.83z"/></svg>
              </div>
              <div class="flex flex-col">
                <span class="font-label-sm text-label-sm uppercase tracking-wider text-secondary">Barometric Pressure</span>
                <div class="flex items-baseline gap-1">
                  <span class="font-data-mono-lg text-headline-md text-on-surface font-bold">n/a</span>
                  <span class="font-body-md text-body-md text-secondary font-medium">hPa</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

    if not env_node_cards:
        env_node_cards.append("""
        <div class="bg-surface-container-lowest rounded-xl p-space-lg shadow-sm text-secondary text-sm">
          No nodes configured in site.toml
        </div>
        """)

    tpl = _assert_sub(
        r'(<!-- Dual Node Telemetry Summary Cards.*?<div class="flex flex-col gap-space-md">).*?(</div>\s*<!-- 3 Parallel Environmental Comparison Charts)',
        r'\g<1>\n' + "\n".join(env_node_cards) + r'\n\g<2>',
        tpl,
        "Environment",
        "Per-node Telemetry Cards Container",
        missing,
        flags=re.S,
    )

    # Bind or remove fake 14:00-14:30 chart time labels
    tpl = _assert_sub(
        r'<div class="flex justify-between[^>]*>\s*<span[^>]*>14:00</span>.*?</div>',
        '<div class="flex justify-between pl-12 pr-4 pt-space-xs font-data-mono-sm text-data-mono-sm text-secondary"><span>-30m</span><span>-20m</span><span>-10m</span><span>now</span></div>',
        tpl,
        "Environment",
        "Chart time labels (14:00-14:30)",
        missing,
    )

    # Updated timing text
    if last_env_t:
        up_sec = int(max(0.0, now - last_env_t))
        up_text = f"Updated {up_sec} s ago"
    else:
        up_text = "Updated —"
    tpl = _assert_replace(tpl, "Updated 2 s ago", up_text, "Environment", missing)

    # Replace static mock charts:
    node_a_id = cfg_nodes[0]["node_id"] if len(cfg_nodes) > 0 else 1
    node_b_id = cfg_nodes[1]["node_id"] if len(cfg_nodes) > 1 else 2
    s1 = environment_series(node_a_id, 30)
    s2 = environment_series(node_b_id, 30)

    svg_matches = list(re.finditer(r'<div class="w-full h-64 relative flex items-center justify-center">\s*<svg.*?</svg>\s*</div>', tpl, flags=re.S))
    if len(svg_matches) >= 3:
        # Chart 1: Temperature
        if s1 or s2:
            pts1 = [(i, (r["temp_c"] - 20.0) / 20.0) for i, r in enumerate(s1) if r.get("temp_c") is not None]
            pts2 = [(i, (r["temp_c"] - 20.0) / 20.0) for i, r in enumerate(s2) if r.get("temp_c") is not None]
            c1 = _curve_path(pts1, w=900, x0=60, y_top=20, y_bot=155) if pts1 else ""
            c2 = _curve_path(pts2, w=900, x0=60, y_top=20, y_bot=155) if pts2 else ""
            p1_tag = f'<path d="{c1}" fill="none" stroke="#063423" stroke-width="3"/>' if c1 else ""
            p2_tag = f'<path d="{c2}" fill="none" stroke="#755843" stroke-width="2.5" stroke-dasharray="6 4"/>' if c2 else ""
            chart1_svg = f"""<div class="w-full h-64 relative flex items-center justify-center">
              <svg class="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 1000 180">
                <line stroke="#E5E9E2" stroke-width="1" x1="45" x2="1000" y1="20" y2="20"></line>
                <text class="font-data-mono-sm text-[12px]" fill="#755843" x="5" y="24">40°C</text>
                <line stroke="#E5E9E2" stroke-width="1" x1="45" x2="1000" y1="87" y2="87"></line>
                <text class="font-data-mono-sm text-[12px]" fill="#755843" x="5" y="91">30°C</text>
                <line stroke="#E5E9E2" stroke-width="1" x1="45" x2="1000" y1="155" y2="155"></line>
                <text class="font-data-mono-sm text-[12px]" fill="#755843" x="5" y="159">20°C</text>
                {p1_tag}
                {p2_tag}
              </svg>
            </div>"""
        else:
            chart1_svg = """<div class="w-full h-48 flex items-center justify-center text-secondary font-mono text-xs bg-surface-container-low rounded-xl">Awaiting temperature telemetry from nodes...</div>"""

        # Chart 2: Relative Humidity
        if s1 or s2:
            pts1_rh = [(i, (r["humidity_pct"] - 30.0) / 70.0) for i, r in enumerate(s1) if r.get("humidity_pct") is not None]
            pts2_rh = [(i, (r["humidity_pct"] - 30.0) / 70.0) for i, r in enumerate(s2) if r.get("humidity_pct") is not None]
            c1_rh = _curve_path(pts1_rh, w=900, x0=60, y_top=20, y_bot=155) if pts1_rh else ""
            c2_rh = _curve_path(pts2_rh, w=900, x0=60, y_top=20, y_bot=155) if pts2_rh else ""
            p1_rh_tag = f'<path d="{c1_rh}" fill="none" stroke="#063423" stroke-width="3"/>' if c1_rh else ""
            p2_rh_tag = f'<path d="{c2_rh}" fill="none" stroke="#755843" stroke-width="2.5" stroke-dasharray="6 4"/>' if c2_rh else ""
            chart2_svg = f"""<div class="w-full h-64 relative flex items-center justify-center">
              <svg class="w-full h-full overflow-visible" preserveAspectRatio="none" viewBox="0 0 1000 180">
                <line stroke="#E5E9E2" stroke-width="1" x1="45" x2="1000" y1="20" y2="20"></line>
                <text class="font-data-mono-sm text-[12px]" fill="#755843" x="5" y="24">100%</text>
                <line stroke="#E5E9E2" stroke-width="1" x1="45" x2="1000" y1="87" y2="87"></line>
                <text class="font-data-mono-sm text-[12px]" fill="#755843" x="5" y="91">65%</text>
                <line stroke="#E5E9E2" stroke-width="1" x1="45" x2="1000" y1="155" y2="155"></line>
                <text class="font-data-mono-sm text-[12px]" fill="#755843" x="5" y="159">30%</text>
                {p1_rh_tag}
                {p2_rh_tag}
              </svg>
            </div>"""
        else:
            chart2_svg = """<div class="w-full h-48 flex items-center justify-center text-secondary font-mono text-xs bg-surface-container-low rounded-xl">Awaiting humidity telemetry from nodes...</div>"""

        # Replace static pressure chart with clear notice
        pressure_notice = """
        <div class="w-full h-48 flex items-center justify-center text-secondary font-mono text-xs bg-surface-container-low rounded-xl">
          Atmospheric pressure telemetry is not transmitted in protocol v1 (n/a)
        </div>
        """

        m3 = svg_matches[2]
        m2 = svg_matches[1]
        m1 = svg_matches[0]
        tpl = tpl[:m3.start()] + pressure_notice + tpl[m3.end():]
        tpl = tpl[:m2.start()] + chart2_svg + tpl[m2.end():]
        tpl = tpl[:m1.start()] + chart1_svg + tpl[m1.end():]

    if missing:
        st.error("TEMPLATE BINDING ERROR: " + " | ".join(missing))

    st.session_state[f"render_{('Environment', st.session_state.get('anchor'))}"] = tpl
    components.html(tpl, height=1100, scrolling=True)


def _render_about() -> None:
    tpl = _load_template("about_team")
    missing: list[str] = []
    tpl = _assert_replace(tpl, "19 Oct 2026 · 14:32 PHT", _wall_time(), "About", missing)
    tpl = _assert_sub(
        r'(<h3[^>]*>Detection</h3>\s*<span[^>]*>)98\.2%(</span>)',
        r'\g<1>CRNN\g<2>',
        tpl,
        "About",
        "Detection 98.2%",
        missing,
    )
    if missing:
        st.error("TEMPLATE BINDING ERROR: " + " | ".join(missing))
    st.session_state[f"render_{('About', st.session_state.get('anchor'))}"] = tpl
    components.html(tpl, height=1100, scrolling=True)


st.set_page_config(page_title="KaawayAn Console", layout="wide")


@st.fragment(run_every=REFRESH_S)
def _live_fragment() -> None:
    cfg, cfg_err = _get_site_config()

    # System Status Banners
    status = all_system_status()
    now_t = time.time()

    # 1. SIMULATED DATA banner (mode=demo)
    mode = status.get("mode", {}).get("value")
    if mode == "demo":
        st.warning("⚠️ SIMULATED DATA — Dashboard is displaying synthetic simulation frames from demo ingest")

    # 2. CLASSIFIER DISABLED/FAILED banner
    cls_status = status.get("classifier", {}).get("value")
    if cls_status == "disabled":
        st.warning("⚠️ CLASSIFIER DISABLED — Machine learning classifier is disabled; running with stub predict")
    elif cls_status == "failed":
        st.error("🚨 CLASSIFIER FAILED — Neural network classifier failed to load or encountered repeated errors")

    # 3. INGEST STALE banner (heartbeat older than 5 s)
    hb = status.get("ingest_heartbeat_t")
    if hb:
        hb_age = now_t - float(hb["t_wall"])
        if hb_age > 5.0:
            st.error(f"🚨 INGEST STALE — No heartbeat received from ingest process for {hb_age:.1f}s")
    else:
        since_w = seconds_since_write(now_t)
        if since_w is not None and since_w > 5.0:
            st.error(f"🚨 INGEST STALE — No database writes for {since_w:.1f}s")

    # 4. NODE LOST banner
    lost_nodes = []
    for k, v in status.items():
        if k.startswith("node_") and k.endswith("_state"):
            if v.get("value") == "lost":
                nid_part = k.replace("node_", "").replace("_state", "")
                lost_nodes.append(f"Node {nid_part}")
    if lost_nodes:
        st.error(f"🚨 NODE LOST — Hardware connection lost to: {', '.join(lost_nodes)}")

    page = st.session_state.get("page", "Overview")
    rows = latest_events(50)

    if page == "Overview":
        _render_overview(rows, event_count(), drone_count(), cfg, cfg_err)
    elif page == "Localization":
        _render_localization_page(cfg, cfg_err)
    elif page == "Detections":
        _render_detection_page(rows, cfg)
    elif page == "Environment":
        _render_environment(cfg)
    else:
        _render_about()


def main() -> None:
    st.sidebar.title("KaawayAn Console")
    st.sidebar.caption("Read-only · SQLite WAL")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Localization", "Detections", "Environment", "About"],
        key="page",
    )
    total, drones = event_count(), drone_count()
    st.sidebar.metric("Total events", total)
    st.sidebar.metric("Drone predictions", drones)

    loc = latest_localization()
    now = time.time()
    age_s = (now - float(loc["t_wall"])) if (loc and loc.get("t_wall") is not None) else None
    if loc and loc.get("status") == "ok" and (age_s is not None and age_s < Config.loc_max_age_s):
        st.sidebar.success(f"2D Fix: ({float(loc['x_m']):.2f}m, {float(loc['y_m']):.2f}m)")
    elif loc and loc.get("status") == "degraded":
        st.sidebar.warning(f"Degraded: {loc.get('reason', 'lost node')}")
    elif loc and loc.get("status") == "uncertain":
        st.sidebar.info(f"Uncertain: {loc.get('reason', 'awaiting fix')}")
    else:
        st.sidebar.caption("No 2D fix")

    # Node liveness from t_wall
    cfg, cfg_err = _get_site_config()
    if cfg_err:
        st.sidebar.error("Config Error: site.toml")
    else:
        target_nodes = [n["node_id"] for n in cfg.get("nodes", [])]
        node_seen = node_last_seen(target_nodes)
        st.sidebar.subheader("Nodes")
        for n in cfg.get("nodes", []):
            nid = n["node_id"]
            lt = node_seen.get(nid)
            if lt is not None:
                dt = now - lt
                if dt < ONLINE_STALE_S:
                    st.sidebar.success(f"Node {nid}: Online ({dt:.1f}s ago)")
                else:
                    st.sidebar.warning(f"Node {nid}: Stale ({dt:.0f}s ago)")
            else:
                st.sidebar.error(f"Node {nid}: Offline")

    _live_fragment()


if __name__ == "__main__":
    main()
