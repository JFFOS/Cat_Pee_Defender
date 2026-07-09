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
# Remember where the log currently ends so we only read *this* run's output below.
START_OFFSET=$([ -f "$LOG" ] && wc -c < "$LOG" || echo 0)
nohup "$PY" -u "$MAIN" >>"$LOG" 2>&1 &   # -u = unbuffered, so the log fills live
disown 2>/dev/null || true

sleep 2
if ! pgrep -f "$MAIN" >/dev/null 2>&1; then
  echo "❌ Failed to start — see the log:"
  tail -n 20 "$LOG" 2>/dev/null
  exit 1
fi

echo "✅ Started. PID $(pgrep -f "$MAIN"). Logging to: $LOG"

# Surface the startup config banner in this window. The model load takes a few
# seconds, so poll the fresh part of the log until the banner appears (or give up).
echo "   Loading model / config…"
for _ in $(seq 1 30); do
  NEW="$(tail -c +$((START_OFFSET + 1)) "$LOG" 2>/dev/null)"
  if printf '%s' "$NEW" | grep -q "watcher config"; then
    printf '%s\n' "$NEW" | sed -n '/----- watcher config -----/,/\[watch\] running/p' \
      | sed 's/^/   /'
    break
  fi
  # Bail early if the process died during startup.
  pgrep -f "$MAIN" >/dev/null 2>&1 || { echo "   ⚠️  exited during startup:"; tail -n 20 "$LOG"; break; }
  sleep 1
done

echo "   You can close this window; the watcher keeps running."
