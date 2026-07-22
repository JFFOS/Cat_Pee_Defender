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
from config import CAT_CLASS_ID, PERSON_CLASS_ID, Settings
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


def _quit_or_hide_preview(settings: Settings, cli_show: bool) -> bool:
    """Handle 'q' in the preview window.

    With CLI --show the preview is the whole point, so 'q' quits the watcher
    (returns True). For a menu-bar-toggled preview 'q' just closes the window —
    clear the flag and tear the window down, but keep watching (returns False).
    """
    if cli_show:
        return True
    Path(settings.preview_flag).unlink(missing_ok=True)
    cv2.destroyAllWindows()
    return False


def open_capture(source, settings: Settings) -> cv2.VideoCapture:
    """Open the video source; for a live camera, request the capped resolution.

    Capping capture at ~720p shrinks every frame the loop handles (copies,
    overlays, pre-roll buffer) with no downstream loss: YOLO infers at 640 and
    clips are written at 960 wide regardless.
    """
    cap = cv2.VideoCapture(source)
    if not isinstance(source, str) and cap.isOpened():
        if settings.capture_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.capture_width)
        if settings.capture_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.capture_height)
    return cap


def release_gpu_cache(settings: Settings) -> None:
    """Hand PyTorch's cached MPS memory back to the OS (no-op on CPU).

    The MPS allocator keeps the full inference working set cached between
    predicts; releasing it periodically keeps the watcher's resident footprint
    near its actual need at the cost of a cheap allocator re-warm.
    """
    if settings.device != "mps":
        return
    try:
        import torch

        torch.mps.empty_cache()
    except Exception:
        pass


def point_in_boxes(px: float, py: float, boxes: list[dict]) -> bool:
    for b in boxes:
        if b["x"] <= px <= b["x"] + b["w"] and b["y"] <= py <= b["y"] + b["h"]:
            return True
    return False


def _rect_overlaps_box(x1: int, y1: int, x2: int, y2: int, b: dict) -> bool:
    """True if the axis-aligned rect [x1,y1,x2,y2] overlaps unsafe box `b`."""
    return not (
        x2 < b["x"] or x1 > b["x"] + b["w"] or
        y2 < b["y"] or y1 > b["y"] + b["h"]
    )


def zone_ids_of_boxes(boxes: list[dict]) -> list[int]:
    """Group unsafe boxes into distinct danger zones and return a zone id per box.

    Two boxes that overlap belong to the same zone (they merge into one outline
    in the overlay), so their zone ids are equal. Disjoint boxes get different
    ids. Used to tell whether a cat and a human are in the *same* zone.
    """
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            bj = boxes[j]
            if _rect_overlaps_box(bi["x"], bi["y"], bi["x"] + bi["w"],
                                  bi["y"] + bi["h"], bj):
                parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def rect_zone_ids(x1: int, y1: int, x2: int, y2: int,
                  boxes: list[dict], zone_id: list[int]) -> set[int]:
    """Set of danger-zone ids whose boxes the detection rect overlaps."""
    return {
        zone_id[i]
        for i, b in enumerate(boxes)
        if _rect_overlaps_box(x1, y1, x2, y2, b)
    }


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
    for (x1, y1, x2, y2, conf, hit, is_cat) in detections:
        if is_cat:
            color = (0, 0, 255) if hit else (0, 200, 0)  # red in-zone, green otherwise
            label = f"cat {conf:.2f}"
        else:
            color = (255, 160, 0)  # blue for people
            label = f"person {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if in_zone:
        cv2.putText(frame, "CAT IN UNSAFE ZONE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 90, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def print_startup_config(settings: Settings, source, is_file: bool,
                         webhook: str | None, urgent_webhook: str | None,
                         n_zones: int) -> None:
    """Dump the effective runtime parameters to the log at startup."""
    src = f"file {source!r}" if is_file else f"camera index {source}"
    if not webhook:
        discord = "sound-only (no webhook)"
    elif urgent_webhook != webhook:
        discord = "on (separate urgent channel)"
    else:
        discord = "on (single channel)"
    lines = [
        "----- watcher config -----",
        f"  source            : {src}",
        f"  zones             : {n_zones} unsafe zone(s) from {settings.zones_path}",
        f"  model / device    : {settings.model_path} on {settings.device}"
        f" ({'fp16' if settings.infer_half and settings.device != 'cpu' else 'fp32'})",
        f"  capture cap       : "
        + (f"{settings.capture_width}x{settings.capture_height}"
           if settings.capture_width and settings.capture_height else "camera default"),
        f"  active hours      : {settings.active_start}–{settings.active_end} (local)",
        f"  alarm hours       : {settings.alarm_start}–{settings.alarm_end} (local)",
        f"  cat conf          : acquire ≥ {settings.conf_threshold:.2f}, "
        f"keep ≥ {settings.conf_keep:.2f}",
        f"  person conf       : ≥ {settings.person_conf_threshold:.2f}"
        f"  (suppress alarm w/ human: {settings.suppress_alarm_with_human})",
        f"  min brightness    : ≥ {settings.min_brightness:.0f} (detection paused when darker)",
        f"  infer imgsz / N   : {settings.infer_imgsz}px every {settings.process_every_n} frame(s)",
        f"  dwell / gap grace : dwell {settings.dwell_seconds:.1f}s, "
        f"presence {settings.presence_gap_grace:.1f}s, companion {settings.companion_grace:.1f}s",
        f"  alert cooldown    : {settings.alert_cooldown_s:.0f}s between Discord alerts",
        f"  clips             : {settings.clip_width}px @ {settings.clip_fps:.0f}fps, "
        f"≤ {settings.max_clip_seconds:.0f}s/part, {settings.clip_preroll_seconds:.0f}s pre-roll",
        f"  discord video     : {discord} (≤ {settings.discord_max_bytes/1e6:.1f}MB)",
        f"  local retention   : keep newest {settings.keep_recent} snapshots/clips",
        f"  heartbeat         : every {settings.heartbeat_s:.0f}s",
        "--------------------------",
    ]
    print("\n".join(lines), flush=True)


def run_watch(settings: Settings, source, show: bool, no_sound: bool) -> None:
    from ultralytics import YOLO

    unsafe_boxes, ref_w, ref_h = load_zones(settings)
    print(f"[watch] loaded {len(unsafe_boxes)} unsafe zone(s)")

    print(f"[watch] loading model on device={settings.device} ...")
    model = YOLO(settings.model_path)

    # A file source plays straight through; a live camera (int index) is released
    # during off-hours so its LED goes dark instead of streaming to nobody.
    is_file = isinstance(source, str)
    cap = open_capture(source, settings)
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
        preroll=settings.clip_preroll_seconds,
    )
    frame_i = 0
    # Visit = cat present anywhere in frame (this is the "cat passed by" tier).
    present_since: float | None = None
    last_seen_present: float = 0.0
    # Unsafe = cat inside an unsafe zone (this tier triggers the loud deterrent).
    unsafe_since: float | None = None
    last_seen_unsafe: float = 0.0
    last_alert: float = -1e9
    last_companion: float = -1e9         # last time a human shared the cat's zone (companion suppression)
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

    print_startup_config(settings, source, is_file, webhook, urgent_webhook, len(unsafe_boxes))
    print("[watch] running. Press Ctrl-C (or 'q' in the preview window) to stop.")
    was_active = True
    last_heartbeat = 0.0
    last_dark_log = 0.0
    last_cache_release = time.time()
    show_prev = show
    try:
        while True:
            now = time.time()

            # Preview can be toggled live from the menu bar via a flag file (or
            # forced on for the whole run with --show). When it turns off, tear
            # the window down so it actually disappears.
            show_now = show or Path(settings.preview_flag).exists()
            if show_prev and not show_now:
                cv2.destroyAllWindows()
            show_prev = show_now

            # --- Active-hours gate: outside the window, idle quietly with the
            # camera released so a live webcam's LED goes dark. We check this
            # *before* reading a frame so nothing keeps the camera streaming. ---
            if not within_active_window(settings):
                if was_active:  # just crossed into off-hours: silence + wrap up
                    alarm.stop()
                    if present_since is not None:
                        finalize_clip(event_rec.stop(), "off-hours")
                        present_since = None
                    unsafe_since = None
                    if not is_file and cap is not None:
                        cap.release()   # turn the webcam off (LED dark) while idle
                        cap = None
                    # Idle for hours: hand the GPU inference cache back to the OS.
                    release_gpu_cache(settings)
                    print(f"[watch] outside active hours ({settings.active_start}–"
                          f"{settings.active_end}); idling"
                          f"{'' if is_file else ' (camera released)'}.")
                    was_active = False
                if now - last_heartbeat >= settings.heartbeat_s:
                    print(f"[watch] {time.strftime('%H:%M:%S', time.localtime(now))} alive — "
                          f"idle (off-hours, active {settings.active_start}-{settings.active_end})"
                          f"{'' if is_file else '; camera off'}",
                          flush=True)
                    last_heartbeat = now
                if show_now:
                    idle = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(idle, f"IDLE (active {settings.active_start}-{settings.active_end})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
                    cv2.imshow("Cat Watch (q to quit)", idle)
                    if cv2.waitKey(1) & 0xFF == ord("q") and _quit_or_hide_preview(settings, show):
                        break
                time.sleep(0.5)
                continue
            if not was_active:
                if cap is None:  # was released during off-hours: reopen the camera
                    cap = open_capture(source, settings)
                    if not cap.isOpened():
                        sys.exit(
                            "Could not reopen the camera after idle hours. Check the index "
                            "and Camera permission (System Settings -> Privacy & Security -> Camera)."
                        )
                print("[watch] active hours resumed; watching"
                      f"{'' if is_file else ' (camera on)'}.")
                was_active = True

            ok, frame = cap.read()
            if not ok:
                if is_file:
                    print("[watch] end of video source.")
                    break
                time.sleep(0.05)
                continue

            # --- Heartbeat: periodic "still alive" line for the headless log ---
            if now - last_heartbeat >= settings.heartbeat_s:
                state = "cat present" if present_since is not None else "watching, no cat"
                print(f"[watch] {time.strftime('%H:%M:%S', time.localtime(now))} alive — {state}",
                      flush=True)
                last_heartbeat = now

            # Periodically return PyTorch's cached MPS memory to the OS so the
            # resident footprint tracks actual need instead of the peak.
            if now - last_cache_release >= settings.gpu_cache_release_s:
                release_gpu_cache(settings)
                last_cache_release = now

            # Keep the last few seconds on hand so a visit clip can begin a bit
            # before the cat is first spotted (pre-roll).
            event_rec.buffer(frame, now)

            # Alarm is only allowed to sound inside the nested alarm window, and
            # can be manually muted from the menu bar (a cross-process flag file).
            # In either case we silence the loop but keep detecting and alerting
            # Discord.
            muted = Path(settings.mute_flag).exists()
            # Near-dark frames make YOLO hallucinate; suppress the loud alarm when
            # the scene is too dim to trust (Discord alerts still fire).
            mean_brightness = float(frame.mean())
            too_dark = mean_brightness < settings.min_brightness
            alarm_allowed = (
                not no_sound and not muted and not too_dark
                and within_alarm_window(settings)
            )
            if alarm.playing and not alarm_allowed:
                alarm.stop()
                if muted:
                    print("[watch] alarm manually muted from menu bar; "
                          "silenced (still alerting Discord).")
                elif too_dark:
                    print(f"[watch] scene too dark (brightness {mean_brightness:.0f} < "
                          f"{settings.min_brightness:.0f}); alarm silenced "
                          "and detection paused.")
                else:
                    print("[watch] outside alarm hours "
                          f"({settings.alarm_start}–{settings.alarm_end}); "
                          "alarm silenced (still alerting Discord).")

            frame_i += 1
            detections: list[tuple] = []
            cat_in_unsafe = False

            # In a near-dark scene YOLO only produces noise, so skip inference
            # entirely: no false detections, no phantom Discord alerts, less
            # compute. An in-progress visit lapses via the usual grace window.
            if too_dark:
                if now - last_dark_log >= settings.heartbeat_s:
                    print(f"[watch] {time.strftime('%H:%M:%S', time.localtime(now))} "
                          f"too dark (brightness {mean_brightness:.0f} < "
                          f"{settings.min_brightness:.0f}); detection paused.",
                          flush=True)
                    last_dark_log = now

            if not too_dark and frame_i % settings.process_every_n == 0:
                h, w = frame.shape[:2]
                unsafe = scale_boxes(unsafe_boxes, ref_w, ref_h, w, h)

                results = model.predict(
                    frame,
                    imgsz=settings.infer_imgsz,
                    conf=settings.conf_keep,  # low floor; hysteresis is applied in code below
                    classes=[PERSON_CLASS_ID, CAT_CLASS_ID],
                    device=settings.device,
                    # fp16 halves activation memory; verified to give identical
                    # detections to fp32 on this model/device.
                    quantize=16 if settings.infer_half and settings.device != "cpu" else None,
                    verbose=False,
                )
                zone_id = zone_ids_of_boxes(unsafe)  # group id per unsafe box
                for r in results:
                    for box in r.boxes:
                        is_cat = int(box.cls[0]) == CAT_CLASS_ID
                        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        hit = point_in_boxes(cx, cy, unsafe)
                        detections.append((x1, y1, x2, y2, conf, hit, is_cat))

                # Confidence hysteresis. YOLO ran at the lower `conf_keep` floor, so
                # a cat hovering near threshold isn't dropped and re-acquired every
                # frame (the flicker). A "confirmed" detection (>= conf_threshold)
                # may *start* a visit / dwell timer; a "tentative" one (only
                # >= conf_keep) merely *sustains* one already open.
                cats_all = [d for d in detections if d[6]]
                cats_conf = [d for d in cats_all if d[4] >= settings.conf_threshold]
                humans = [d for d in detections if not d[6]]
                humans_conf = [d for d in humans if d[4] >= settings.person_conf_threshold]

                cat_present = bool(cats_conf)                    # acquire: open a visit
                cat_present_keep = bool(cats_all)                # sustain: keep it open
                cat_in_unsafe = any(d[5] for d in cats_conf)     # acquire: start dwell
                cat_in_unsafe_keep = any(d[5] for d in cats_all)  # sustain: keep dwell alive
                best_conf_any = max((d[4] for d in cats_all), default=0.0)
                best_conf_unsafe = max((d[4] for d in cats_all if d[5]), default=0.0)

                # Danger zones holding a cat (any conf) and a *confident* human. A
                # phantom low-conf "person" must not be able to mute a real alarm,
                # so the human side requires conf_threshold.
                cat_zone_ids: set[int] = set()
                for d in cats_all:
                    if d[5]:
                        cat_zone_ids |= rect_zone_ids(d[0], d[1], d[2], d[3], unsafe, zone_id)
                human_zone_ids: set[int] = set()
                for d in humans_conf:
                    human_zone_ids |= rect_zone_ids(d[0], d[1], d[2], d[3], unsafe, zone_id)

                # Human sharing the cat's zone => likely playing: suppress the alarm
                # (Discord still fires). Remember *when* we last saw that so a
                # one-frame miss of the person doesn't briefly un-suppress and let
                # the alarm blare — suppression is held for `companion_grace` seconds.
                if settings.suppress_alarm_with_human and (cat_zone_ids & human_zone_ids):
                    last_companion = now
                companion = (
                    settings.suppress_alarm_with_human
                    and (now - last_companion) <= settings.companion_grace
                )

                if cat_present_keep:
                    zone_note = " [UNSAFE ZONE]" if cat_in_unsafe_keep else ""
                    human_note = (
                        " +human(same zone)" if companion
                        else " +human" if humans_conf else ""
                    )
                    print(
                        f"[detect] {len(cats_all)} cat(s){human_note} "
                        f"conf={best_conf_any:.2f}{zone_note}",
                        flush=True,
                    )

                # --- TIER 1: any cat anywhere ("passed by") -> log, notify, record ---
                # A tentative (keep-level) cat sustains an open visit; only a
                # confirmed one opens a new one.
                if cat_present_keep:
                    last_seen_present = now
                    event_max_conf = max(event_max_conf, best_conf_any)
                if present_since is None:
                    if cat_present:  # start of a new visit (confident sighting)
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
                elif not cat_present_keep and now - last_seen_present > settings.presence_gap_grace:
                    # Visit ends once not even a tentative cat has been seen for
                    # longer than the grace window.
                    finalize_clip(event_rec.stop(), "left the frame")
                    present_since = None

                # --- TIER 2: cat in an UNSAFE zone -> loud deterrent + urgent alert ---
                if cat_in_unsafe_keep:
                    last_seen_unsafe = now
                    event_unsafe = True  # this visit's clip routes to the urgent channel
                    if unsafe_since is None and cat_in_unsafe:
                        unsafe_since = now  # only a confident in-zone cat starts the dwell timer
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
                        if companion:
                            # A human shares the cat's zone — almost certainly
                            # playing with it. Stay quiet (silence if already loud).
                            if alarm.playing:
                                alarm.stop()
                                print("[watch] human in the cat's zone; alarm suppressed "
                                      "(likely playing).")
                        elif alarm_allowed and not alarm.playing:
                            # Keep the loud deterrent looping for as long as the cat
                            # stays — but only during the nested alarm-sound window.
                            print("[watch] ALARM: looping until the cat leaves the zone.")
                            alarm.start()
                        # Discord alert + logged snapshot, throttled so we don't spam.
                        if (now - last_alert) >= settings.alert_cooldown_s:
                            # On a frame where the in-zone cat has momentarily
                            # flickered out, fall back to the best conf seen this
                            # visit instead of reporting a misleading 0.00.
                            alert_conf = best_conf_unsafe if best_conf_unsafe > 0 else event_max_conf
                            if companion:
                                msg = (
                                    f"🐾🧑 Cat + human together in the UNSAFE zone for "
                                    f"{int(dwell)}s (likely playing) — alarm off. "
                                    f"conf={alert_conf:.2f}"
                                )
                                event_name = "companion"
                            else:
                                msg = (
                                    f"🚨 Cat on the sofa (UNSAFE zone) for {int(dwell)}s! "
                                    f"conf={alert_conf:.2f}"
                                )
                                event_name = "alert"
                            print(f"[watch] ALERT: {msg}")
                            snapshot = draw_overlay(frame.copy(), unsafe, detections, True)
                            snap_path = save_snapshot(settings, snapshot)
                            send_discord(urgent_webhook, msg, snapshot)
                            append_event(settings, event_name, "sofa", alert_conf, dwell,
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
                    # Continuation segment picks up exactly where part N ended —
                    # no pre-roll (that would duplicate the tail of the prior part).
                    event_rec.start(frame, now, include_preroll=False)

            if show_now:
                view = draw_overlay(frame.copy(), unsafe, detections,
                                    cat_in_unsafe, recording=event_rec.active)
                cv2.imshow("Cat Watch (q to quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q") and _quit_or_hide_preview(settings, show):
                    break
    except KeyboardInterrupt:
        print("\n[watch] stopped by user.")
    finally:
        alarm.stop()
        finalize_clip(event_rec.stop(), "watcher stopped")
        if cap is not None:
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
