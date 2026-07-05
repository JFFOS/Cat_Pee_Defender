"""Deterrent + notification side effects: local sound and Discord webhook."""
from __future__ import annotations

import datetime as _dt
import os
import signal
import subprocess

import cv2
import numpy as np
import requests


class Alarm:
    """A looping alarm that plays until explicitly stopped.

    Spawns macOS `afplay` in its own process group and replays the sound on a
    loop, so the deterrent can run for as long as the cat stays in the danger
    zone and be silenced the instant it leaves. Non-blocking; failures are
    swallowed so a missing sound file or audio hiccup never takes down the watcher.
    """

    def __init__(self, sound_path: str):
        self.sound_path = sound_path
        self._proc: subprocess.Popen | None = None

    @property
    def playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.playing:
            return
        try:
            # Loop afplay forever; path passed as an argument (not interpolated)
            # so an odd filename can't break or inject into the command.
            script = 'f=$1; while true; do afplay "$f"; done'
            self._proc = subprocess.Popen(
                ["/bin/sh", "-c", script, "sh", self.sound_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # own process group so we can kill afplay too
            )
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[alerts] could not start alarm: {exc}")
            self._proc = None

    def stop(self) -> None:
        """Silence the alarm immediately (kills the loop and the current afplay)."""
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except Exception:  # pragma: no cover - already gone
            pass
        self._proc = None


def send_discord(webhook_url: str | None, message: str, frame: "np.ndarray | None" = None) -> bool:
    """Post a message (and optional snapshot) to a Discord webhook.

    Returns True on apparent success. Network errors are caught and logged so
    the watch loop keeps running.
    """
    if not webhook_url:
        print("[alerts] no DISCORD_WEBHOOK_URL set; skipping Discord alert")
        return False

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"{message}\n🕒 {stamp}"

    try:
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                files = {"file": ("snapshot.jpg", buf.tobytes(), "image/jpeg")}
                resp = requests.post(
                    webhook_url,
                    data={"content": content},
                    files=files,
                    timeout=10,
                )
            else:
                resp = requests.post(webhook_url, json={"content": content}, timeout=10)
        else:
            resp = requests.post(webhook_url, json={"content": content}, timeout=10)

        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"[alerts] Discord alert failed: {exc}")
        return False


def send_discord_video(webhook_url: str | None, message: str, video_path: str,
                       max_bytes: int = 8_000_000) -> bool:
    """Upload a short video clip to a Discord webhook.

    Discord caps webhook uploads (~10 MB on non-boosted servers), so if the clip
    exceeds `max_bytes` we fall back to a text note pointing at the local file.
    """
    if not webhook_url:
        print("[alerts] no DISCORD_WEBHOOK_URL set; skipping Discord video")
        return False

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        size = os.path.getsize(video_path)
        if size > max_bytes:
            note = (f"{message}\n🎥 clip is {size/1e6:.1f} MB — too large for Discord; "
                    f"saved locally at {video_path}\n🕒 {stamp}")
            return send_discord(webhook_url, note)

        with open(video_path, "rb") as f:
            files = {"file": (os.path.basename(video_path), f.read(), "video/mp4")}
            resp = requests.post(
                webhook_url,
                data={"content": f"{message}\n🕒 {stamp}"},
                files=files,
                timeout=60,
            )
        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"[alerts] Discord video failed: {exc}")
        return False
