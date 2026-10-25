"""Minimal events.db reader. Run from repo root: streamlit run dashboard/test_dashboard.py.
Collapsible sidebar tabs + light-green theme. Delete when the real dashboard takes over."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

try:
    from dashboard.db import (
        confidence_series,
        distinct_nodes,
        drone_count,
        event_count,
        latest_events,
        max_event_id,
        seconds_since_write,
    )
except ImportError:  # run as script: dashboard/ itself is on sys.path
    from db import (
        confidence_series,
        distinct_nodes,
        drone_count,
        event_count,
        latest_events,
        max_event_id,
        seconds_since_write,
    )

st.set_page_config(page_title="events.db check", layout="wide",
                   initial_sidebar_state="expanded")

# Theme from ui_assets DESIGN.md: light-green surfaces, dark-brown text.
st.markdown(
    """<style>
    .stApp { background-color: #EFFDF4; color: #3E2A1E; }
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #3E2A1E;
    }
    div[data-testid="stMetricValue"] > div { color: #3E2A1E; }
    div[data-testid="stMetricLabel"] > div { color: #795C46; }
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #3E2A1E;
    }
    section[data-testid="stSidebar"] { background-color: #E3F1E9; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label {
        color: #3E2A1E;
    }
    </style>""",
    unsafe_allow_html=True,
)

st.title("events.db — live check")

total = event_count()
drones = drone_count()
anchor = max_event_id()
stale = seconds_since_write()
nodes = distinct_nodes()

with st.sidebar:
    st.header("Pages")
    page = st.radio("Go to", ["Overview", "Detections", "Confidence", "About"],
                    label_visibility="collapsed")
    st.divider()
    st.metric("Total events", total)
    st.metric("Drone predictions", drones)
    if stale is None:
        st.warning("events.db not found yet")
    elif stale < 30:
        st.success(f"Pipeline live ({stale:.0f}s ago)")
    else:
        st.warning(f"Pipeline stale ({stale:.0f}s ago)")

if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Total events", total)
    c2.metric("Drone predictions", drones)
    c3.metric("Max event_id", anchor if anchor is not None else "—")
    st.subheader(f"Nodes seen: {nodes if nodes else '—'}")

elif page == "Detections":
    st.subheader("Latest 20 events")
    rows = latest_events(20)
    if rows:
        st.dataframe(
            [{"event_id": r["event_id"], "node": r["node_id"],
              "drone": bool(r["predicted_drone"]),
              "conf": round(float(r["model_confidence"]), 4)}
             for r in rows],
            use_container_width=True,
        )
    else:
        st.warning("No rows. Start the pipeline: python main.py")

elif page == "Confidence":
    st.subheader("Confidence (last 120)")
    series = confidence_series(120)
    if series:
        st.line_chart({e: c for e, c in series})
    else:
        st.warning("No data points yet.")

else:  # About
    st.subheader("About")
    st.write("Minimal DB reader. Full UI lives in ui_assets/ (see dashboard_plan.md).")
    st.write(f"Max event_id: {anchor if anchor is not None else '—'}")

if st.button("Refresh now"):
    st.rerun()
st.caption(f"Auto-refresh in 2s · read at {time.strftime('%H:%M:%S')}")
time.sleep(2)
st.rerun()
