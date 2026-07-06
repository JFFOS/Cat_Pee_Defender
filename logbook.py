"""Event logging: CSV records, saved snapshots, and 10-minute clip recording.

Every time the cat is caught in an unsafe (sofa) zone we:
  - save a JPEG snapshot,
  - start recording a fixed-length video clip,
  - append a row to logs/events.csv.
"""
from __future__ import annotations

import csv
import datetime as _dt
from collections import deque
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


def prune_dir(directory: str, pattern: str, keep: int) -> None:
    """Keep only the `keep` newest files matching `pattern`; delete the rest.

    Discord holds the full archive, so locally we retain just the most recent
    files for debugging. Safe if fewer than `keep` files exist.
    """
    if keep is None or keep <= 0:
        return
    files = sorted(
        Path(directory).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def save_snapshot(settings, frame) -> str:
    """Write a snapshot JPEG and return its path."""
    _ensure_dirs(settings)
    path = Path(settings.snapshot_dir) / f"snap_{_stamp()}.jpg"
    cv2.imwrite(str(path), frame)
    prune_dir(settings.snapshot_dir, "snap_*.jpg", settings.keep_recent)
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
                 max_width: int | None = None, prefix: str = "clip", label: str = "rec",
                 preroll: float | None = None):
        self.settings = settings
        self.seconds = seconds if seconds is not None else settings.record_seconds
        self.fps = fps if fps is not None else settings.record_fps
        self.max_width = max_width
        self.prefix = prefix
        self.label = label
        self.preroll = preroll if preroll is not None else 0.0
        self.writer: cv2.VideoWriter | None = None
        self.codec: str = "mp4v"
        self.path: str | None = None
        self.until: float = 0.0
        self._out_size: tuple[int, int] | None = None
        # Rolling buffer of recent (timestamp, resized frame) so a clip can begin
        # `preroll` seconds before the cat is spotted. Fed via buffer() every loop.
        self._preroll_buf: deque[tuple[float, "cv2.Mat"]] = deque()

    @property
    def active(self) -> bool:
        return self.writer is not None

    def _resize(self, frame):
        if self.max_width and frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * scale)))
        return frame

    def _open_writer(self, w: int, h: int) -> cv2.VideoWriter:
        """Prefer H.264 (avc1) for far smaller files; fall back to mp4v if the
        OpenCV/FFmpeg build can't open an H.264 writer."""
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*codec), self.fps, (w, h)
            )
            if writer.isOpened():
                self.codec = codec
                return writer
            writer.release()
        # Last resort: return the (unopened) mp4v writer so write() just no-ops.
        self.codec = "mp4v"
        return cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))

    def buffer(self, frame, now: float) -> None:
        """Keep the last `preroll` seconds of frames on hand so start() can
        prepend them. Call every loop while watching (cheap: one resize + prune)."""
        if self.preroll <= 0:
            return
        self._preroll_buf.append((now, self._resize(frame)))
        cutoff = now - self.preroll
        while self._preroll_buf and self._preroll_buf[0][0] < cutoff:
            self._preroll_buf.popleft()

    def start(self, frame, now: float, include_preroll: bool = True) -> str:
        _ensure_dirs(self.settings)
        out = self._resize(frame)
        h, w = out.shape[:2]
        self._out_size = (w, h)
        self.path = str(Path(self.settings.clip_dir) / f"{self.prefix}_{_stamp()}.mp4")
        self.writer = self._open_writer(w, h)
        self.until = now + self.seconds
        pre_n = 0
        if include_preroll and self.preroll > 0:
            cutoff = now - self.preroll
            # Only frames that match the writer's dimensions (guards against a
            # mid-run resolution change) get flushed as pre-roll.
            for ts, buf_frame in self._preroll_buf:
                if ts >= cutoff and (buf_frame.shape[1], buf_frame.shape[0]) == (w, h):
                    self.writer.write(buf_frame)
                    pre_n += 1
        pre_note = f" (+{pre_n} pre-roll frames)" if pre_n else ""
        print(f"[{self.label}] started {self.seconds:.0f}s {self.codec} clip{pre_note} "
              f"-> {self.path}")
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
            # Keep only the newest clips locally (the one just finished is newest).
            prune_dir(self.settings.clip_dir, f"{self.prefix}_*.mp4", self.settings.keep_recent)
            return self.path
        return None
