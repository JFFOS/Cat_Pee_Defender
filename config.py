"""Configuration for the cat pee-zone watcher.

Runtime tunables live here as defaults. Secrets (the Discord webhook URL) are read
from a local .env file so they never end up hard-coded in source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env sitting next to this file (if present) into os.environ.
load_dotenv(Path(__file__).with_name(".env"))

PROJECT_DIR = Path(__file__).parent
ZONES_PATH = PROJECT_DIR / "zones.json"
SOUND_PATH = PROJECT_DIR / "alarm.wav"
# yolov8m chosen over n/s: on the real (blurry, wide-angle, small-cat) webcam frames
# the nano/small models missed the cat, while yolov8m detected it reliably at ~0.9
# confidence. Heavier, but we only infer every Nth frame so it stays light on Apple Silicon.
MODEL_PATH = PROJECT_DIR / "yolov8m.pt"  # auto-downloaded on first use if missing

# Where event records, snapshots, and recorded clips are written.
LOG_DIR = PROJECT_DIR / "logs"
EVENTS_CSV = LOG_DIR / "events.csv"
SNAPSHOT_DIR = LOG_DIR / "snapshots"
CLIP_DIR = LOG_DIR / "clips"

# COCO class index for "cat".
CAT_CLASS_ID = 15


def _auto_device() -> str:
    """Prefer Apple's Metal (MPS) backend when available, else CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Settings:
    # Capture
    camera_index: int = 0

    # Detection
    conf_threshold: float = 0.40      # min YOLO confidence to count a cat
    infer_imgsz: int = 640            # inference resolution (640 detects the small blurry cat best)
    process_every_n: int = 3          # run YOLO on every Nth grabbed frame
    device: str = field(default_factory=_auto_device)

    # Dwell / alert logic
    dwell_seconds: float = 1.0        # continuous in-zone time before firing
    presence_gap_grace: float = 1.5   # tolerate detection dropouts up to this long
    alert_cooldown_s: float = 60.0    # min gap between two Discord alerts (sound loops regardless)

    # Active hours — the watcher only detects & alarms within this daily window
    # (local time, 24-hour "HH:MM"). Outside it, it idles quietly. Set both the
    # same value (e.g. "00:00"/"00:00") to disable the limit and watch 24/7.
    active_start: str = "09:30"       # <-- change to move the start of watching
    active_end: str = "22:00"         # <-- change to move the end (22:00 = 10:00pm)
    heartbeat_s: float = 60.0         # print an "alive" status line this often (headless log)

    # Recording — a clip holds only the frames where the cat is in an unsafe zone,
    # packed together (empty room is never recorded). A long visit is split into
    # short segments so each part stays small enough to upload to Discord.
    clip_width: int = 960             # downscale width (H.264 keeps these small)
    clip_fps: float = 12.0            # playback fps for saved clips
    max_clip_seconds: float = 60.0    # cap per segment; long visits upload as parts 1..N

    # Discord video
    discord_video: bool = True        # upload the event clip to Discord
    discord_max_bytes: int = 8_000_000  # skip upload above this (Discord ~10MB cap)

    # Local retention — Discord keeps the full archive, so on disk we only keep the
    # most recent files for debugging. Older snapshots/clips are pruned automatically.
    keep_recent: int = 10             # newest snapshots and clips to keep locally

    # Paths (resolved from module-level constants above)
    model_path: str = str(MODEL_PATH)
    sound_path: str = str(SOUND_PATH)
    zones_path: str = str(ZONES_PATH)
    events_csv: str = str(EVENTS_CSV)
    snapshot_dir: str = str(SNAPSHOT_DIR)
    clip_dir: str = str(CLIP_DIR)

    @property
    def discord_webhook_url(self) -> str | None:
        return os.environ.get("DISCORD_WEBHOOK_URL")

    @property
    def discord_urgent_webhook_url(self) -> str | None:
        """Webhook for urgent unsafe-zone alerts.

        Routed to a separate channel so the normal "cat spotted" channel can be
        muted while these stay noisy. Falls back to the normal webhook if unset.
        """
        return os.environ.get("DISCORD_URGENT_WEBHOOK_URL") or self.discord_webhook_url
