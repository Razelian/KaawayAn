#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${KAAWAYAN_CONFIG:-config/site.toml}"
INGEST_ARGS=(main.py --config "$CONFIG")
if [[ "${KAAWAYAN_NO_CLASSIFIER:-0}" == "1" ]]; then
  INGEST_ARGS+=(--no-classifier)
fi

python -m streamlit run dashboard/dashboard.py &
dashboard_pid=$!
ingest_pid=""

shutdown() {
  [[ -z "$ingest_pid" ]] || kill -TERM "$ingest_pid" 2>/dev/null || true
  kill -TERM "$dashboard_pid" 2>/dev/null || true
  [[ -z "$ingest_pid" ]] || wait "$ingest_pid" 2>/dev/null || true
  wait "$dashboard_pid" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

while kill -0 "$dashboard_pid" 2>/dev/null; do
  python "${INGEST_ARGS[@]}" &
  ingest_pid=$!
  wait "$ingest_pid" || status=$?
  ingest_pid=""
  [[ -n "${status:-}" ]] && echo "Ingest exited with status $status; restarting in 2 seconds." >&2
  unset status
  sleep 2
done
echo "Dashboard process exited; stopping station launcher." >&2
exit 1
