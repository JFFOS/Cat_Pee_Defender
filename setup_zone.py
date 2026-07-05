"""One-time setup: draw the sofa warning zone(s) and generate the alarm sound.

Run once (or whenever the camera moves):

    python setup_zone.py

Grab a frame from the webcam, drag one or more boxes over the sofa (the pee
danger area). Press Enter/Space to accept each box, then Esc or `c` to finish.
The boxes are saved to zones.json. A loud alarm.wav is also (re)generated.
"""
from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import cv2
import numpy as np

from config import Settings, ZONES_PATH, SOUND_PATH


def generate_alarm_wav(path: Path, seconds: float = 2.0, sample_rate: int = 44100) -> None:
    """Synthesize a harsh, loud two-tone siren and write it as a mono 16-bit WAV."""
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    # Sweep between two frequencies a few times a second for a jarring siren.
    lfo = np.sign(np.sin(2 * np.pi * 4 * t))  # square LFO, 4 Hz
    freq = np.where(lfo > 0, 900.0, 1400.0)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    # Square wave carrier is harsher/louder-sounding than a sine.
    wave_data = np.sign(np.sin(phase))
    audio = (wave_data * 0.9 * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(struct.pack("<h", s) for s in audio))
    print(f"[setup] wrote alarm sound -> {path}")


def _grab_frame(settings: Settings):
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open the camera. Check the camera index and that Terminal/PyCharm "
            "has Camera permission (System Settings -> Privacy & Security -> Camera)."
        )
    frame = None
    for _ in range(10):  # warm up for a stable exposure
        ok, frame = cap.read()
        if not ok:
            frame = None
    cap.release()
    if frame is None:
        raise SystemExit("Failed to read a frame from the camera.")
    return frame


def _draw_boxes(frame, label: str, color, existing: list[dict]) -> list[dict]:
    """Interactively collect one or more boxes of a single category."""
    boxes: list[dict] = []
    win = f"Draw {label} zone(s) (ENTER=accept, ESC=finish)"
    while True:
        preview = frame.copy()
        for group, gcolor, gname in existing + [(boxes, color, label)]:
            for b in group:
                cv2.rectangle(preview, (b["x"], b["y"]),
                              (b["x"] + b["w"], b["y"] + b["h"]), gcolor, 2)
                cv2.putText(preview, gname, (b["x"] + 4, b["y"] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, gcolor, 2)
        roi = cv2.selectROI(win, preview, showCrosshair=True, fromCenter=False)
        x, y, bw, bh = (int(v) for v in roi)
        if bw == 0 or bh == 0:
            break
        boxes.append({"x": x, "y": y, "w": bw, "h": bh})
        print(f"[setup] added {label} box #{len(boxes)}: x={x} y={y} w={bw} h={bh}")
    cv2.destroyAllWindows()
    return boxes


def draw_zones(settings: Settings) -> None:
    frame = _grab_frame(settings)
    h, w = frame.shape[:2]

    print(
        "\nStep 1 — draw the UNSAFE zone(s) (the sofa / pee danger area, shown RED):\n"
        "  - Drag a box, press ENTER/SPACE to accept, repeat for more.\n"
        "  - Press ESC (with no active drag) when done.\n"
    )
    unsafe = _draw_boxes(frame, "UNSAFE", (0, 0, 255), existing=[])

    if not unsafe:
        raise SystemExit("No unsafe zones drawn; nothing saved.")

    data = {"frame_w": w, "frame_h": h, "unsafe": unsafe}
    ZONES_PATH.write_text(json.dumps(data, indent=2))
    print(f"[setup] wrote {len(unsafe)} unsafe zone(s) -> {ZONES_PATH}")


def main() -> None:
    settings = Settings()
    generate_alarm_wav(SOUND_PATH)
    draw_zones(settings)
    print("\nSetup complete. Next: put your webhook in .env, then run `python main.py --show`.")


if __name__ == "__main__":
    main()
