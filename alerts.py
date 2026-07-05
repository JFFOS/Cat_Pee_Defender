"""Deterrent + notification side effects: local sound and Discord webhook."""
from __future__ import annotations

import datetime as _dt
import os
import subprocess

import cv2
import numpy as np
import requests


def play_sound(sound_path: str) -> None:
    """Play the alarm through the Mac speakers without blocking the watch loop.

    Uses macOS's built-in `afplay`. Failures are swallowed so a missing sound
    file or audio hiccup never takes down the watcher.
    """
    try:
        subprocess.Popen(
            ["afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[alerts] could not play sound: {exc}")


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
