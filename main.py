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
import numpy as np

from alerts import Alarm, send_discord, send_discord_video
from config import CAT_CLASS_ID, Settings
from logbook import ClipRecorder, append_event, save_snapshot


def load_zones(settings: Settings) -> tuple[list[dict], int | None, int | None]:
    """Return (unsafe_boxes, frame_w, frame_h) from zones.json.

    Accepts the legacy "warning" key as an alias for "unsafe".
    """
    path = Path(settings.zones_path)
    if not path.exists():
        sys.exit(
            f"No zones file at {path}. Run `python setup_zone.py` first to draw the zones."
        )
    data = json.loads(path.read_text())
    unsafe = data.get("unsafe", data.get("warning", []))
    return unsafe, data.get("frame_w"), data.get("frame_h")


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


def _hm_to_min(s: str) -> int:
    """Parse an 'HH:MM' string into minutes since midnight."""
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _within_window(start_hm: str, end_hm: str) -> bool:
    """True if the current local time is inside a daily [start, end) window.

    Handles windows that cross midnight (start > end). If start == end the
    window is treated as disabled (always True / runs 24/7).
    """
    start, end = _hm_to_min(start_hm), _hm_to_min(end_hm)
    if start == end:
        return True
    lt = time.localtime()
    cur = lt.tm_hour * 60 + lt.tm_min
    if start < end:
        return start <= cur < end
    return cur >= start or cur < end  # overnight window


def within_active_window(settings: Settings) -> bool:
    """True if now is inside the detection + Discord window (active hours)."""
    return _within_window(settings.active_start, settings.active_end)


def within_alarm_window(settings: Settings) -> bool:
    """True if now is inside the nested window where the loud alarm may play."""
    return _within_window(settings.alarm_start, settings.alarm_end)


def point_in_boxes(px: float, py: float, boxes: list[dict]) -> bool:
    for b in boxes:
        if b["x"] <= px <= b["x"] + b["w"] and b["y"] <= py <= b["y"] + b["h"]:
            return True
    return False


def draw_overlay(frame, unsafe: list[dict],
                 detections: list[tuple], in_zone: bool, recording: bool = False):
    # Draw the unsafe zones as a single outline around their combined shape:
    # overlapping rects merge into one contour (no inner lines), while keeping
    # the true footprint instead of one big bounding box.
    if unsafe:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for b in unsafe:
            cv2.rectangle(mask, (b["x"], b["y"]),
                          (b["x"] + b["w"], b["y"] + b["h"]), 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
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

    unsafe_boxes, ref_w, ref_h = load_zones(settings)
    print(f"[watch] loaded {len(unsafe_boxes)} unsafe zone(s)")

    print(f"[watch] loading model on device={settings.device} ...")
    model = YOLO(settings.model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(
            "Could not open the video source. If using the camera, check the index and that "
            "Terminal/PyCharm has Camera permission (System Settings -> Privacy & Security -> Camera)."
        )

    webhook = settings.discord_webhook_url            # normal activity (spotted + clips)
    urgent_webhook = settings.discord_urgent_webhook_url  # unsafe-zone alerts
    if not webhook:
        print("[watch] WARNING: DISCORD_WEBHOOK_URL not set; alerts will be sound-only.")
    elif urgent_webhook != webhook:
        print("[watch] urgent unsafe-zone alerts routed to a separate Discord channel.")

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
    event_seg: int = 0                   # segment index within the current visit (part N)
    event_unsafe: bool = False           # did the cat enter an unsafe zone during this visit?
    unsafe = unsafe_boxes
    alarm = Alarm(settings.sound_path)   # loops while the cat is in the unsafe zone

    def finalize_clip(path: str | None, reason: str) -> None:
        """Log the finished visit clip segment and upload it to Discord."""
        if not path:
            return
        dwell = (last_seen_present - present_since) if present_since else 0.0
        append_event(settings, "clip", "any", event_max_conf, max(dwell, 0.0), clip=path)
        if settings.discord_video:
            # Visits that touched an unsafe zone go to the urgent channel; plain
            # "passed by" visits stay on the normal channel.
            if event_unsafe:
                dest = urgent_webhook
                msg = f"🚨🎥 Cat in UNSAFE zone — part {event_seg} (conf {event_max_conf:.2f})"
                channel = "urgent"
            else:
                dest = webhook
                msg = f"🎥 Cat visit — part {event_seg} (conf {event_max_conf:.2f})"
                channel = "normal"
            print(f"[watch] uploading clip part {event_seg} to Discord {channel} channel "
                  f"({reason}): {path}")
            send_discord_video(dest, msg, path, max_bytes=settings.discord_max_bytes)

    print(f"[watch] running (active {settings.active_start}–{settings.active_end}; "
          f"alarm {settings.alarm_start}–{settings.alarm_end}). "
          "Press Ctrl-C (or 'q' in the preview window) to stop.")
    was_active = True
    last_heartbeat = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str):
                    print("[watch] end of video source.")
                    break
                time.sleep(0.05)
                continue

            now = time.time()

            # --- Heartbeat: periodic "still alive" line for the headless log ---
            if now - last_heartbeat >= settings.heartbeat_s:
                active = within_active_window(settings)
                if active:
                    state = "cat present" if present_since is not None else "watching, no cat"
                else:
                    state = f"idle (off-hours, active {settings.active_start}-{settings.active_end})"
                print(f"[watch] {time.strftime('%H:%M:%S', time.localtime(now))} alive — {state}",
                      flush=True)
                last_heartbeat = now

            # --- Active-hours gate: outside the window, idle quietly ---
            if not within_active_window(settings):
                if was_active:  # just crossed into off-hours: silence + wrap up
                    alarm.stop()
                    if present_since is not None:
                        finalize_clip(event_rec.stop(), "off-hours")
                        present_since = None
                    unsafe_since = None
                    print(f"[watch] outside active hours ({settings.active_start}–"
                          f"{settings.active_end}); idling.")
                    was_active = False
                if show:
                    idle = frame.copy()
                    cv2.putText(idle, f"IDLE (active {settings.active_start}-{settings.active_end})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
                    cv2.imshow("Cat Watch (q to quit)", idle)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(0.5)
                continue
            if not was_active:
                print("[watch] active hours resumed; watching.")
                was_active = True

            # Alarm is only allowed to sound inside the nested alarm window. If we
            # cross out of it while the cat lingers, silence the loop but keep
            # detecting and alerting Discord.
            alarm_allowed = not no_sound and within_alarm_window(settings)
            if alarm.playing and not alarm_allowed:
                alarm.stop()
                print("[watch] outside alarm hours "
                      f"({settings.alarm_start}–{settings.alarm_end}); "
                      "alarm silenced (still alerting Discord).")

            frame_i += 1
            detections: list[tuple] = []
            cat_in_unsafe = False

            if frame_i % settings.process_every_n == 0:
                h, w = frame.shape[:2]
                unsafe = scale_boxes(unsafe_boxes, ref_w, ref_h, w, h)

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
                        hit = point_in_boxes(cx, cy, unsafe)
                        detections.append((x1, y1, x2, y2, conf, hit))
                        if hit:
                            cat_in_unsafe = True

                cat_present = len(detections) > 0
                best_conf_any = max((d[4] for d in detections), default=0.0)
                best_conf_unsafe = max((d[4] for d in detections if d[5]), default=0.0)

                if cat_present:
                    zone_note = " [UNSAFE ZONE]" if cat_in_unsafe else ""
                    print(
                        f"[detect] {len(detections)} cat(s) conf={best_conf_any:.2f}"
                        f"{zone_note}",
                        flush=True,
                    )

                # --- TIER 1: any cat anywhere ("passed by") -> log, notify, record ---
                if cat_present:
                    last_seen_present = now
                    event_max_conf = max(event_max_conf, best_conf_any)
                    if present_since is None:  # start of a new visit
                        present_since = now
                        event_max_conf = best_conf_any
                        event_seg = 1
                        event_unsafe = False
                        print("[watch] cat spotted; recording visit.")
                        snapshot = draw_overlay(frame.copy(), unsafe, detections,
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
                    event_unsafe = True  # this visit's clip routes to the urgent channel
                    if unsafe_since is None:
                        unsafe_since = now
                elif unsafe_since is not None:
                    # Cat has been out of the zone longer than the grace window.
                    if now - last_seen_unsafe > settings.presence_gap_grace:
                        unsafe_since = None
                        if alarm.playing:
                            alarm.stop()
                            print("[watch] cat left the unsafe zone; alarm stopped.")

                if unsafe_since is not None:
                    dwell = now - unsafe_since
                    if dwell >= settings.dwell_seconds:
                        # Keep the loud deterrent looping for as long as the cat
                        # stays — but only during the nested alarm-sound window.
                        if alarm_allowed and not alarm.playing:
                            print("[watch] ALARM: looping until the cat leaves the zone.")
                            alarm.start()
                        # Discord alert + logged snapshot, throttled so we don't spam.
                        if (now - last_alert) >= settings.alert_cooldown_s:
                            msg = (
                                f"🚨 Cat on the sofa (UNSAFE zone) for {int(dwell)}s! "
                                f"conf={best_conf_unsafe:.2f}"
                            )
                            print(f"[watch] ALERT: {msg}")
                            snapshot = draw_overlay(frame.copy(), unsafe, detections, True)
                            snap_path = save_snapshot(settings, snapshot)
                            send_discord(urgent_webhook, msg, snapshot)
                            append_event(settings, "alert", "sofa", best_conf_unsafe, dwell,
                                         snapshot=snap_path, clip=event_rec.path or "")
                            last_alert = now

            # --- record every frame while a visit is in progress ---
            if present_since is not None and event_rec.active:
                event_rec.write(draw_overlay(frame.copy(), unsafe, detections,
                                             cat_in_unsafe, recording=True))
                capped = event_rec.maybe_finish(now)  # split long visits into parts
                if capped:
                    finalize_clip(capped, "segment full")
                    event_seg += 1
                    event_rec.start(frame, now)

            if show:
                view = draw_overlay(frame.copy(), unsafe, detections,
                                    cat_in_unsafe, recording=event_rec.active)
                cv2.imshow("Cat Watch (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[watch] stopped by user.")
    finally:
        alarm.stop()
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
        print("[test] looping alarm for ~8s (as it would while a cat stays)...")
        alarm = Alarm(settings.sound_path)
        alarm.start()
        time.sleep(8)
        alarm.stop()
        print("[test] alarm stopped.")
        return

    if args.test_discord:
        print("[test] sending test Discord alert to the normal channel...")
        ok = send_discord(settings.discord_webhook_url, "✅ Cat watcher test alert (normal channel).")
        print("[test] sent." if ok else "[test] failed — check DISCORD_WEBHOOK_URL in .env.")
        urgent = settings.discord_urgent_webhook_url
        if urgent and urgent != settings.discord_webhook_url:
            print("[test] sending test Discord alert to the urgent channel...")
            ok_u = send_discord(urgent, "🚨 Cat watcher test alert (urgent channel).")
            print("[test] sent." if ok_u else "[test] failed — check DISCORD_URGENT_WEBHOOK_URL in .env.")
        return

    source = args.source if args.source else settings.camera_index
    run_watch(settings, source, show=args.show, no_sound=args.no_sound)


if __name__ == "__main__":
    main()
