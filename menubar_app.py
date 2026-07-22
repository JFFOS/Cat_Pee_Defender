"""Menu-bar companion for the cat pee-zone watcher (macOS).

A small always-there icon in the Mac menu bar so you can tell at a glance whether
the headless watcher (`main.py`) is running, and fire quick actions without
opening Terminal.

Icon at a glance:
    🐾  running & inside active hours (actively watching)
    😴  running but outside active hours (idle, camera released)
    🚫  not running

Menu:
    • live status (pid, hours state, last heartbeat, last event) — read-only
    • Start / Stop / Restart the watcher
    • Mute alarm — manual override that silences the loud alarm (Discord still
      alerts). Toggles a flag file the watcher reads; a checkmark + notification
      show the current state.
    • Show preview — open/close the running watcher's live detection window (a
      flag file the watcher reads; no restart needed).
    • Test Sound / Test Discord (delegates to main.py)
    • Open the log, clips folder, and events.csv in Finder/Console

Run it with `run_menubar.command` (double-click) or:
    /opt/anaconda3/envs/Cat_pee/bin/python menubar_app.py

Quitting this app does NOT stop the watcher — the watcher is a separate process.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import rumps

from config import Settings

PROJECT_DIR = Path(__file__).resolve().parent
MAIN = PROJECT_DIR / "main.py"
LOG = PROJECT_DIR / "logs" / "watcher.log"
PY = os.environ.get("CAT_PEE_PYTHON", "/opt/anaconda3/envs/Cat_pee/bin/python")

POLL_SECONDS = 3.0


# --- tiny active-hours helpers (mirror main.py, but without importing cv2) ----
def _hm_to_min(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _within_window(start_hm: str, end_hm: str) -> bool:
    start, end = _hm_to_min(start_hm), _hm_to_min(end_hm)
    if start == end:
        return True
    lt = time.localtime()
    cur = lt.tm_hour * 60 + lt.tm_min
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end  # overnight window


# --- process discovery --------------------------------------------------------
def watcher_pids() -> list[int]:
    """PIDs of the running watcher (matched exactly like the .command scripts)."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", str(MAIN)],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return []
    return [int(p) for p in out.split() if p.strip().isdigit()]


def start_watcher() -> str:
    """Launch the watcher headless & detached, appending to the log. No-op if up."""
    if watcher_pids():
        return "already running"
    if not Path(PY).exists():
        return f"python not found: {PY}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as log:
        subprocess.Popen(
            [PY, "-u", str(MAIN)],
            cwd=str(PROJECT_DIR),
            stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,   # survive this app closing
        )
    return "starting"


def stop_watcher() -> str:
    pids = watcher_pids()
    if not pids:
        return "not running"
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    return f"stopped pid {' '.join(map(str, pids))}"


# --- log / event tails --------------------------------------------------------
def _tail(path: Path, nbytes: int = 8192) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def last_heartbeat() -> str | None:
    for line in reversed(_tail(LOG).splitlines()):
        if "alive" in line:
            # e.g. "[watch] 14:03:01 alive — watching, no cat"
            return line.split("]", 1)[-1].strip()
    return None


def last_event(settings: Settings) -> str | None:
    text = _tail(Path(settings.events_csv))
    rows = [ln for ln in text.splitlines() if ln and not ln.startswith("timestamp")]
    if not rows:
        return None
    parts = rows[-1].split(",")
    if len(parts) < 5:
        return rows[-1]
    ts, event, zone, conf, dwell = parts[:5]
    return f"{event} in {zone} · conf {conf} · {ts}"


def _reveal(path: Path) -> None:
    """Open a file/folder in Finder (or its default app)."""
    if path.exists():
        subprocess.run(["open", "-R" if path.is_file() else "", str(path)],
                       check=False)
    else:
        rumps.notification("Cat Watcher", "Not found yet", str(path))


class CatWatcherApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Cat Watcher", title="🐾", quit_button=None)
        self.settings = Settings()

        # Read-only status lines (updated on each poll).
        self.status_item = rumps.MenuItem("Status: …")
        self.hours_item = rumps.MenuItem("Hours: …")
        self.heartbeat_item = rumps.MenuItem("Heartbeat: …")
        self.event_item = rumps.MenuItem("Last event: …")

        # Manual mute override (checkmark reflects the shared flag file's state).
        self.mute_item = rumps.MenuItem("🔇  Mute alarm", callback=self.on_toggle_mute)
        # Live preview toggle (checkmark reflects the shared flag file's state).
        self.preview_item = rumps.MenuItem("🖥  Show preview", callback=self.on_toggle_preview)

        self.menu = [
            self.status_item,
            self.hours_item,
            self.heartbeat_item,
            self.event_item,
            None,
            rumps.MenuItem("▶️  Start watcher", callback=self.on_start),
            rumps.MenuItem("⏹  Stop watcher", callback=self.on_stop),
            rumps.MenuItem("🔄  Restart watcher", callback=self.on_restart),
            None,
            self.mute_item,
            self.preview_item,
            rumps.MenuItem("🔊  Test sound", callback=self.on_test_sound),
            rumps.MenuItem("📨  Test Discord", callback=self.on_test_discord),
            None,
            rumps.MenuItem("📄  Open log", callback=self.on_open_log),
            rumps.MenuItem("🎞  Open clips folder", callback=self.on_open_clips),
            rumps.MenuItem("🗒  Open events.csv", callback=self.on_open_events),
            None,
            rumps.MenuItem("Quit menu bar (watcher keeps running)",
                           callback=rumps.quit_application),
        ]

        self.refresh(None)
        self.timer = rumps.Timer(self.refresh, POLL_SECONDS)
        self.timer.start()

    # --- mute override --------------------------------------------------------
    def is_muted(self) -> bool:
        return Path(self.settings.mute_flag).exists()

    def set_muted(self, muted: bool) -> None:
        p = Path(self.settings.mute_flag)
        if muted:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        else:
            p.unlink(missing_ok=True)

    # --- live preview ---------------------------------------------------------
    def preview_on(self) -> bool:
        return Path(self.settings.preview_flag).exists()

    def set_preview(self, on: bool) -> None:
        p = Path(self.settings.preview_flag)
        if on:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        else:
            p.unlink(missing_ok=True)

    # --- polling --------------------------------------------------------------
    def refresh(self, _sender) -> None:
        pids = watcher_pids()
        active = _within_window(self.settings.active_start, self.settings.active_end)
        muted = self.is_muted()
        if pids:
            base = "🐾" if active else "😴"
            # Show the mute state right in the menu-bar title so it's visible
            # at a glance without opening the menu.
            self.title = f"🔇{base}" if muted else base
            self.status_item.title = f"● Running (pid {pids[0]})"
        else:
            self.title = "🔇🚫" if muted else "🚫"
            self.status_item.title = "○ Stopped"

        self.mute_item.state = 1 if muted else 0
        self.preview_item.state = 1 if self.preview_on() else 0

        alarm_ok = _within_window(self.settings.alarm_start, self.settings.alarm_end)
        if not pids:
            self.hours_item.title = "Hours: watcher stopped"
        elif active:
            if muted:
                state = "alarm MUTED (manual override)"
            elif alarm_ok:
                state = "alarm armed"
            else:
                state = "alarm quiet (Discord only)"
            self.hours_item.title = f"Hours: active · {state}"
        else:
            note = " · alarm MUTED" if muted else ""
            self.hours_item.title = (
                f"Hours: idle (active {self.settings.active_start}"
                f"–{self.settings.active_end}){note}"
            )

        hb = last_heartbeat()
        self.heartbeat_item.title = f"Heartbeat: {hb}" if hb else "Heartbeat: (none yet)"
        ev = last_event(self.settings)
        self.event_item.title = f"Last event: {ev}" if ev else "Last event: (none yet)"

    # --- actions --------------------------------------------------------------
    def on_start(self, _s) -> None:
        rumps.notification("Cat Watcher", "Start", start_watcher())
        self.refresh(None)

    def on_stop(self, _s) -> None:
        rumps.notification("Cat Watcher", "Stop", stop_watcher())
        self.refresh(None)

    def on_restart(self, _s) -> None:
        stop_watcher()
        # Give the old process a moment to release the camera before reopening.
        time.sleep(1.5)
        rumps.notification("Cat Watcher", "Restart", start_watcher())
        self.refresh(None)

    def on_toggle_mute(self, _s) -> None:
        # Flip the shared flag, then pop up a notice confirming the new state.
        now_muted = not self.is_muted()
        self.set_muted(now_muted)
        if now_muted:
            rumps.notification(
                "Cat Watcher", "🔇 Alarm muted",
                "Loud alarm suppressed until you unmute. Discord alerts still fire.",
            )
        else:
            rumps.notification(
                "Cat Watcher", "🔊 Alarm unmuted",
                "Loud alarm re-armed (subject to alarm hours).",
            )
        self.refresh(None)

    def on_toggle_preview(self, _s) -> None:
        # The preview window is drawn by the watcher process, so it only appears
        # when the watcher is running. Flip the shared flag either way so the
        # choice sticks and takes effect as soon as the watcher is up.
        now_on = not self.preview_on()
        self.set_preview(now_on)
        if now_on and not watcher_pids():
            rumps.notification(
                "Cat Watcher", "🖥 Preview will open when watching",
                "The watcher is stopped — the preview appears once it's running.",
            )
        elif now_on:
            rumps.notification(
                "Cat Watcher", "🖥 Preview opening",
                "Live detection window is opening (press q in it to close).",
            )
        else:
            rumps.notification("Cat Watcher", "🖥 Preview closed", "Live window hidden.")
        self.refresh(None)

    def on_test_sound(self, _s) -> None:
        subprocess.Popen([PY, str(MAIN), "--test-sound"], cwd=str(PROJECT_DIR))

    def on_test_discord(self, _s) -> None:
        subprocess.Popen([PY, str(MAIN), "--test-discord"], cwd=str(PROJECT_DIR))

    def on_open_log(self, _s) -> None:
        if LOG.exists():
            subprocess.run(["open", "-a", "Console", str(LOG)], check=False)
        else:
            rumps.notification("Cat Watcher", "No log yet", str(LOG))

    def on_open_clips(self, _s) -> None:
        _reveal(Path(self.settings.clip_dir))

    def on_open_events(self, _s) -> None:
        _reveal(Path(self.settings.events_csv))


if __name__ == "__main__":
    CatWatcherApp().run()
