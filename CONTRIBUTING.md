# Contributing to Cat Pee Defender

Thanks for your interest! This is a small computer-vision side project, but
contributions — bug reports, ideas, and pull requests — are welcome.

## License

By contributing, you agree that your contributions are licensed under the
project's **GNU Affero General Public License v3.0** (see [`LICENSE`](LICENSE)).
Note that the project depends on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics),
which is itself AGPL-3.0.

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your Discord webhook URL(s)
python setup_zone.py        # draw the unsafe (sofa) zones
python main.py              # run the watcher
python main.py --source clip.mp4   # or test against a recorded video
```

The YOLOv8 weights (`yolov8*.pt`) download automatically on first run and are
git-ignored — never commit them.

## Reporting bugs

Open an issue with:
- what you did and what you expected vs. what happened,
- your OS, Python version, and `device` (CPU / MPS / CUDA),
- relevant `[watch]` / `[detect]` log lines (redact any webhook URLs).

## Pull requests

- Keep changes focused; one logical change per PR.
- Match the surrounding style — the code favors small, well-commented functions
  and descriptive names over cleverness.
- Explain the "why" in comments where behavior isn't obvious (see the existing
  zone-grouping and pre-roll code for the tone).
- Test your change end to end against a real or recorded feed before opening the
  PR, and describe how you verified it.

## Security & privacy

- **Never commit secrets.** Discord webhook URLs live in `.env` (git-ignored);
  only placeholder values belong in `.env.example`.
- Don't commit captured footage, snapshots, or `logs/` output.
