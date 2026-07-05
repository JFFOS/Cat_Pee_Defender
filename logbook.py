"""Event logging: CSV records, saved snapshots, and 10-minute clip recording.

Every time the cat is caught in an unsafe (sofa) zone we:
  - save a JPEG snapshot,
  - start recording a fixed-length video clip,
  - append a row to logs/events.csv.
"""
from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

import cv2

CSV_HEADER = [
    "timestamp", "event", "zone", "confidence", "dwell_s", "snapshot", "clip"
]


def _ensure_dirs(settings) -> None:
    Path(settings.snapshot_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.clip_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.events_csv).parent.mkdir(parents=True, exist_ok=True)


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_snapshot(settings, frame) -> str:
    """Write a snapshot JPEG and return its path."""
    _ensure_dirs(settings)
    path = Path(settings.snapshot_dir) / f"snap_{_stamp()}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def append_event(settings, event: str, zone: str, confidence: float,
                 dwell_s: float, snapshot: str = "", clip: str = "") -> None:
    """Append one row to events.csv, writing the header if the file is new."""
    _ensure_dirs(settings)
    csv_path = Path(settings.events_csv)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow([
            _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event, zone, f"{confidence:.2f}", f"{dwell_s:.1f}", snapshot, clip,
        ])


class ClipRecorder:
    """Records a single fixed-length video clip via cv2.VideoWriter.

    Call `start()` to begin, feed frames to `write()`, and `maybe_finish()` each
    loop to auto-close once the duration elapses. Optionally downscales frames to
    `max_width` (used for the small clip uploaded to Discord).
    """

    def __init__(self, settings, seconds: float | None = None, fps: float | None = None,
                 max_width: int | None = None, prefix: str = "clip", label: str = "rec"):
        self.settings = settings
        self.seconds = seconds if seconds is not None else settings.record_seconds
        self.fps = fps if fps is not None else settings.record_fps
        self.max_width = max_width
        self.prefix = prefix
        self.label = label
        self.writer: cv2.VideoWriter | None = None
        self.path: str | None = None
        self.until: float = 0.0
        self._out_size: tuple[int, int] | None = None

    @property
    def active(self) -> bool:
        return self.writer is not None

    def _resize(self, frame):
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * scale)))
        return frame

    def start(self, frame, now: float) -> str:
        _ensure_dirs(self.settings)
        out = self._resize(frame)
        h, w = out.shape[:2]
        self._out_size = (w, h)
        self.path = str(Path(self.settings.clip_dir) / f"{self.prefix}_{_stamp()}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self.until = now + self.seconds
        print(f"[{self.label}] started {self.seconds:.0f}s clip -> {self.path}")
        return self.path

    def write(self, frame) -> None:
        if self.writer is not None:
            self.writer.write(self._resize(frame))

    def maybe_finish(self, now: float) -> str | None:
        """Auto-close when the duration elapses; return the path if it just finished."""
        if self.writer is not None and now >= self.until:
            return self.stop()
        return None

    def stop(self) -> str | None:
        if self.writer is not None:
            self.writer.release()
            print(f"[{self.label}] finished clip -> {self.path}")
            self.writer = None
            return self.path
        return None
