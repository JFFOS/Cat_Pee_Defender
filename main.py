"""Cat pee-zone watcher.

Watches a webcam feed, detects the cat with YOLOv8n, and fires an alert (loud
local alarm + Discord message with a snapshot) when the cat lingers inside a
"warning" zone (the sofa) for a dwell period.

Usage:
    python setup_zone.py                 # first: draw zones + make alarm.wav
    python main.py                       # live watch
    python main.py --show                # live watch with preview window
    python main.py --source clip.mp4     # test against a recorded video
    python main.py --test-sound          # play the alarm and exit
    python main.py --test-discord        # send a test Discord alert and exit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from alerts import play_sound, send_discord, send_discord_video
from config import CAT_CLASS_ID, Settings
from logbook import ClipRecorder, append_event, save_snapshot


def load_zones(settings: Settings) -> tuple[list[dict], list[dict], int | None, int | None]:
    """Return (unsafe_boxes, safe_boxes, frame_w, frame_h) from zones.json.

    Accepts the legacy "warning" key as an alias for "unsafe".
    """
    path = Path(settings.zones_path)
    if not path.exists():
        sys.exit(
            f"No zones file at {path}. Run `python setup_zone.py` first to draw the zones."
        )
    data = json.loads(path.read_text())
    unsafe = data.get("unsafe", data.get("warning", []))
    safe = data.get("safe", [])
    return unsafe, safe, data.get("frame_w"), data.get("frame_h")


def scale_boxes(boxes: list[dict], ref_w, ref_h, cur_w: int, cur_h: int) -> list[dict]:
    """Rescale saved boxes if the live frame size differs from setup time."""
    if not ref_w or not ref_h or (ref_w == cur_w and ref_h == cur_h):
        return boxes
    sx, sy = cur_w / ref_w, cur_h / ref_h
    return [
        {
            "x": int(b["x"] * sx),
            "y": int(b["y"] * sy),
            "w": int(b["w"] * sx),
            "h": int(b["h"] * sy),
        }
        for b in boxes
    ]


def point_in_boxes(px: float, py: float, boxes: list[dict]) -> bool:
    for b in boxes:
        if b["x"] <= px <= b["x"] + b["w"] and b["y"] <= py <= b["y"] + b["h"]:
            return True
    return False


def draw_overlay(frame, unsafe: list[dict], safe: list[dict],
                 detections: list[tuple], in_zone: bool, recording: bool = False):
    for b in safe:
        cv2.rectangle(frame, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 200, 0), 2)
    for b in unsafe:
        cv2.rectangle(frame, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 0, 255), 2)
    for (x1, y1, x2, y2, conf, hit) in detections:
        color = (0, 0, 255) if hit else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"cat {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if in_zone:
        cv2.putText(frame, "CAT IN UNSAFE ZONE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 90, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def run_watch(settings: Settings, source, show: bool, no_sound: bool) -> None:
    from ultralytics import YOLO

    unsafe_boxes, safe_boxes, ref_w, ref_h = load_zones(settings)
    print(f"[watch] loaded {len(unsafe_boxes)} unsafe + {len(safe_boxes)} safe zone(s)")

    print(f"[watch] loading model on device={settings.device} ...")
    model = YOLO(settings.model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(
            "Could not open the video source. If using the camera, check the index and that "
            "Terminal/PyCharm has Camera permission (System Settings -> Privacy & Security -> Camera)."
        )

    webhook = settings.discord_webhook_url
    if not webhook:
        print("[watch] WARNING: DISCORD_WEBHOOK_URL not set; alerts will be sound-only.")

    # One clip per visit: records only the frames where the cat is in an unsafe zone.
    event_rec = ClipRecorder(
        settings, seconds=settings.max_clip_seconds, fps=settings.clip_fps,
        max_width=settings.clip_width, prefix="event", label="rec",
    )
    frame_i = 0
    # Visit = cat present anywhere in frame (this is the "cat passed by" tier).
    present_since: float | None = None
    last_seen_present: float = 0.0
    # Unsafe = cat inside an unsafe zone (this tier triggers the loud deterrent).
    unsafe_since: float | None = None
    last_seen_unsafe: float = 0.0
    last_alert: float = -1e9
    event_max_conf: float = 0.0          # best confidence seen during this visit
    unsafe = unsafe_boxes
    safe = safe_boxes

    def finalize_clip(path: str | None, reason: str) -> None:
        """Log the finished visit clip and upload it to Discord."""
        if not path:
            return
        dwell = (last_seen_present - present_since) if present_since else 0.0
        append_event(settings, "clip", "any", event_max_conf, max(dwell, 0.0), clip=path)
        if settings.discord_video:
            msg = f"🎥 Cat visit clip ({reason}). conf={event_max_conf:.2f}"
            print(f"[watch] uploading visit clip to Discord: {path}")
            send_discord_video(webhook, msg, path, max_bytes=settings.discord_max_bytes)

    print("[watch] running. Press Ctrl-C (or 'q' in the preview window) to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str):
                    print("[watch] end of video source.")
                    break
                time.sleep(0.05)
                continue

            frame_i += 1
            now = time.time()
            detections: list[tuple] = []
            cat_in_unsafe = False

            if frame_i % settings.process_every_n == 0:
                h, w = frame.shape[:2]
                unsafe = scale_boxes(unsafe_boxes, ref_w, ref_h, w, h)
                safe = scale_boxes(safe_boxes, ref_w, ref_h, w, h)

                results = model.predict(
                    frame,
                    imgsz=settings.infer_imgsz,
                    conf=settings.conf_threshold,
                    classes=[CAT_CLASS_ID],
                    device=settings.device,
                    verbose=False,
                )
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        # Safe zones override: a cat in a safe area never counts.
                        hit = point_in_boxes(cx, cy, unsafe) and not point_in_boxes(cx, cy, safe)
                        detections.append((x1, y1, x2, y2, conf, hit))
                        if hit:
                            cat_in_unsafe = True

                cat_present = len(detections) > 0
                best_conf_any = max((d[4] for d in detections), default=0.0)
                best_conf_unsafe = max((d[4] for d in detections if d[5]), default=0.0)

                # --- TIER 1: any cat anywhere ("passed by") -> log, notify, record ---
                if cat_present:
                    last_seen_present = now
                    event_max_conf = max(event_max_conf, best_conf_any)
                    if present_since is None:  # start of a new visit
                        present_since = now
                        event_max_conf = best_conf_any
                        print("[watch] cat spotted; recording visit.")
                        snapshot = draw_overlay(frame.copy(), unsafe, safe, detections,
                                                cat_in_unsafe)
                        snap = save_snapshot(settings, snapshot)
                        event_rec.start(frame, now)
                        append_event(settings, "cat_seen", "frame", best_conf_any, 0.0,
                                     snapshot=snap, clip=event_rec.path or "")
                        send_discord(webhook, f"🐾 Cat spotted (conf {best_conf_any:.2f})",
                                     snapshot)
                elif present_since is not None:
                    # Visit ends once no cat has been seen for longer than the grace window.
                    if now - last_seen_present > settings.presence_gap_grace:
                        finalize_clip(event_rec.stop(), "left the frame")
                        present_since = None

                # --- TIER 2: cat in an UNSAFE zone -> loud deterrent + urgent alert ---
                if cat_in_unsafe:
                    last_seen_unsafe = now
                    if unsafe_since is None:
                        unsafe_since = now
                elif unsafe_since is not None:
                    if now - last_seen_unsafe > settings.presence_gap_grace:
                        unsafe_since = None

                if unsafe_since is not None:
                    dwell = now - unsafe_since
                    if dwell >= settings.dwell_seconds and (now - last_alert) >= settings.alert_cooldown_s:
                        msg = (
                            f"🚨 Cat on the sofa (UNSAFE zone) for {int(dwell)}s! "
                            f"conf={best_conf_unsafe:.2f}"
                        )
                        print(f"[watch] ALERT: {msg}")
                        if not no_sound:
                            play_sound(settings.sound_path)
                        snapshot = draw_overlay(frame.copy(), unsafe, safe, detections, True)
                        snap_path = save_snapshot(settings, snapshot)
                        send_discord(webhook, msg, snapshot)
                        append_event(settings, "alert", "sofa", best_conf_unsafe, dwell,
                                     snapshot=snap_path, clip=event_rec.path or "")
                        last_alert = now

            # --- record every frame while a visit is in progress ---
            if present_since is not None and event_rec.active:
                event_rec.write(draw_overlay(frame.copy(), unsafe, safe, detections,
                                             cat_in_unsafe, recording=True))
                capped = event_rec.maybe_finish(now)  # split very long visits
                if capped:
                    finalize_clip(capped, "still present, split clip")
                    event_rec.start(frame, now)

            if show:
                view = draw_overlay(frame.copy(), unsafe, safe, detections,
                                    cat_in_unsafe, recording=event_rec.active)
                cv2.imshow("Cat Watch (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[watch] stopped by user.")
    finally:
        finalize_clip(event_rec.stop(), "watcher stopped")
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cat pee-zone watcher")
    parser.add_argument("--source", help="Video file to watch instead of the camera")
    parser.add_argument("--show", action="store_true", help="Show a preview window")
    parser.add_argument("--no-sound", action="store_true", help="Do not play the local alarm")
    parser.add_argument("--test-sound", action="store_true", help="Play the alarm and exit")
    parser.add_argument("--test-discord", action="store_true", help="Send a test Discord alert and exit")
    args = parser.parse_args()

    settings = Settings()

    if args.test_sound:
        print("[test] playing alarm...")
        play_sound(settings.sound_path)
        time.sleep(2.5)
        return

    if args.test_discord:
        print("[test] sending test Discord alert...")
        ok = send_discord(settings.discord_webhook_url, "✅ Cat watcher test alert (no image).")
        print("[test] sent." if ok else "[test] failed — check DISCORD_WEBHOOK_URL in .env.")
        return

    source = args.source if args.source else settings.camera_index
    run_watch(settings, source, show=args.show, no_sound=args.no_sound)


if __name__ == "__main__":
    main()
