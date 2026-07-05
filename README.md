# Cat Pee-Zone Watcher 🐱🚨

Lightweight webcam watcher for a MacBook (Apple Silicon). It detects the cat with
YOLOv8m and works in two tiers:

- **Any cat sighting** → logs it, records a compact visit clip, and sends a Discord
  "🐾 Cat spotted" notice + the clip.
- **Cat in an unsafe zone** (the sofa) for a dwell period → plays a **loud alarm**
  and sends an **urgent Discord alert**.

It uses object detection (not fur color), so a grey cat is detected reliably
regardless of lighting.

## Setup

1. **Install dependencies** (in the `Cat_pee` conda env):
   ```bash
   pip install -r requirements.txt
   ```

2. **Grant camera permission**: the app that launches Python (Terminal or PyCharm)
   must be allowed under **System Settings → Privacy & Security → Camera**, or the
   webcam returns empty frames.

3. **Draw the zones + create the alarm sound**:
   ```bash
   python setup_zone.py
   ```
   The camera is assumed **fixed**, so zones are saved as absolute pixel
   rectangles in `zones.json` and stay valid until you move the camera.
   - **UNSAFE (red):** drag box(es) over the sofa / pee danger area. Enter to
     accept each, Esc to finish. Overlapping boxes are auto-merged into one clean
     rectangle at watch time.

   This also (re)generates `alarm.wav`.

4. **Configure Discord**: copy `.env.example` to `.env` and paste your webhook URL:
   ```bash
   cp .env.example .env
   ```

## Run

```bash
python main.py --show          # live watch with a preview window
python main.py                 # live watch, headless
```

The first run auto-downloads the YOLOv8m weights (`yolov8m.pt`, ~50 MB). The
medium model is used because the nano/small models missed the small, blurry cat
on the real wide-angle webcam frames, while `yolov8m` detects it reliably at
~0.9 confidence. Inference runs only every `process_every_n` frames, so it stays
light on Apple Silicon.

### Testing / tuning

```bash
python main.py --test-sound            # play the alarm and exit
python main.py --test-discord          # send a test Discord message and exit
python main.py --source clip.mp4 --show  # run against a recorded video
```

Tune behavior in `config.py`: `dwell_seconds`, `conf_threshold`,
`alert_cooldown_s`, `process_every_n`, `infer_imgsz`.

## How it works — two tiers

**Tier 1 — any cat, anywhere ("passed by"):** whenever a cat appears in frame, a
`cat_seen` row is logged, a snapshot + Discord "🐾 Cat spotted" notice is sent,
and a **visit clip starts recording**. The clip contains only the frames while
the cat is around (empty room is never recorded). When the cat leaves the frame
(after `presence_gap_grace`), the clip is finalized and **uploaded to Discord**.

**Tier 2 — cat in an UNSAFE zone (the deterrent):** if the cat's box center is
inside an unsafe zone continuously for `dwell_seconds` (default 5s), the **loud
alarm starts looping and keeps playing until the cat leaves** the zone (it stops
`presence_gap_grace` after the last in-zone sighting). An urgent `🚨` Discord
alert also fires, throttled by `alert_cooldown_s` so repeats don't spam.

Very long visits are split into chunks of `max_clip_seconds`.

## Logs

Everything is written under `logs/`:
- `logs/events.csv` — one row per event: `timestamp, event, zone, confidence,
  dwell_s, snapshot, clip`. Event types: `cat_seen`, `alert`, `clip`.
- `logs/snapshots/` — a JPEG per event.
- `logs/clips/` — one compact `.mp4` per visit (downscaled to `clip_width`).

Discord keeps the full archive, so on disk only the newest `keep_recent` (default
10) snapshots and clips are retained for debugging; older files are auto-deleted.

## Discord video

Because Discord caps webhook uploads (~10 MB on non-boosted servers), clips are
downscaled (`clip_width`, default 960) and kept to the frames with the cat. If a
clip still exceeds `discord_max_bytes` it is kept locally and Discord gets a note
pointing to the file instead. Set `discord_video = False` in `config.py` to skip
video uploads entirely.

> Note: this detects the cat *being on the sofa*, a practical proxy for "about to
> pee." It does not classify the peeing posture itself.
