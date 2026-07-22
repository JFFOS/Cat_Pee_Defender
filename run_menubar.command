#!/bin/bash
# Cat Watcher — menu-bar app launcher.
#
#   • Double-click in Finder to put the 🐾 icon in your Mac menu bar, or
#   • add to System Settings ▸ General ▸ Login Items to show it at every login.
#
# The menu-bar app lets you see at a glance whether the watcher is running and
# gives you quick Start/Stop/Restart + Test actions. It runs detached, so you
# can close this Terminal window and the icon stays. Quitting the menu-bar app
# does NOT stop the watcher.
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PY="/opt/anaconda3/envs/Cat_pee/bin/python"   # the Cat_pee conda env interpreter
APP="$DIR/menubar_app.py"
LOG="$DIR/logs/menubar.log"
mkdir -p "$DIR/logs"

# Don't start a second menu-bar icon if one is already up.
if pgrep -f "$APP" >/dev/null 2>&1; then
  echo "✅ Menu-bar app already running (pid $(pgrep -f "$APP" | tr '\n' ' '))."
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "❌ Python not found at: $PY"
  echo "   Edit PY= in this script to point at your environment's python."
  exit 1
fi

echo "🐾 Launching Cat Watcher menu-bar app…"
nohup "$PY" -u "$APP" >>"$LOG" 2>&1 &
disown 2>/dev/null || true

sleep 2
if pgrep -f "$APP" >/dev/null 2>&1; then
  echo "✅ Look for the 🐾 icon in your menu bar. You can close this window."
else
  echo "❌ Failed to start — see the log:"
  tail -n 20 "$LOG" 2>/dev/null
  exit 1
fi
