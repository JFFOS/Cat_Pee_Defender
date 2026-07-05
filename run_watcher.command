#!/bin/bash
# Cat Pee-Zone Watcher — one-click headless deploy.
#
#   • Double-click in Finder to start the watcher in the background, or
#   • add to System Settings ▸ General ▸ Login Items to auto-start at login.
#
# It runs headless (no preview window), keeps running after this window closes,
# and logs to logs/watcher.log. Running it again while it's already up is a no-op.
set -uo pipefail

# Resolve the folder this script lives in, so it works no matter where it's run.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PY="/opt/anaconda3/envs/Cat_pee/bin/python"   # the Cat_pee conda env interpreter
MAIN="$DIR/main.py"
LOG="$DIR/logs/watcher.log"
mkdir -p "$DIR/logs"

# Don't start a second copy if one is already running.
if pgrep -f "$MAIN" >/dev/null 2>&1; then
  echo "✅ Watcher already running (pid $(pgrep -f "$MAIN" | tr '\n' ' '))."
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "❌ Python not found at: $PY"
  echo "   Edit PY= in this script to point at your environment's python."
  exit 1
fi

echo "🚀 Starting Cat Pee-Zone Watcher (headless)…"
nohup "$PY" -u "$MAIN" >>"$LOG" 2>&1 &   # -u = unbuffered, so the log fills live
disown 2>/dev/null || true

sleep 2
if pgrep -f "$MAIN" >/dev/null 2>&1; then
  echo "✅ Started. PID $(pgrep -f "$MAIN"). Logging to: $LOG"
  echo "   You can close this window; the watcher keeps running."
else
  echo "❌ Failed to start — see the log:"
  tail -n 20 "$LOG" 2>/dev/null
fi
