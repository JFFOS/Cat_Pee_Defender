#!/bin/bash
# Cat Pee-Zone Watcher — stop the headless watcher. Double-click to run.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN="$DIR/main.py"

if pgrep -f "$MAIN" >/dev/null 2>&1; then
  echo "🛑 Stopping watcher (pid $(pgrep -f "$MAIN" | tr '\n' ' '))…"
  pkill -f "$MAIN"
  sleep 1
  if pgrep -f "$MAIN" >/dev/null 2>&1; then
    echo "⚠️  Still running; forcing…"; pkill -9 -f "$MAIN"
  fi
  echo "✅ Stopped."
else
  echo "ℹ️  Watcher is not running."
fi
