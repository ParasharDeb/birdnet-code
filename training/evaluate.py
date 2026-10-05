"""Compare the custom classifier with stock BirdNET on your own recordings.

    .\\.venv\\Scripts\\python.exe -m training.evaluate                 # files in public/
    .\\.venv\\Scripts\\python.exe -m training.evaluate path\\to\\a.wav ...

Every 3 s window is scored (0.5 s apart by default). For each target, prints the
best score from each model and how many windows reach 0.70. These files are
excluded from training, so this is an honest check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from app.custom_model import TargetClassifier
from app.species import BY_KEY, resolve_targets

from .common import MODEL_PATH, ROOT, BirdNetRunner, load_audio, windows

THRESHOLD = 0.70


def fmt(value, width: int, spec: str = "") -> str:
    return f"{'-' if value is None else format(value, spec):>{width}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--step", type=float, default=0.5)
    args = ap.parse_args()

    files = args.files or sorted((ROOT / "public").glob("*"))
    if not files:
        sys.exit("No files given and public/ is empty.")
    if not MODEL_PATH.exists():
        sys.exit(f"No model at {MODEL_PATH} — run `python -m training.train` first.")

    clf = TargetClassifier(MODEL_PATH)
    runner = BirdNetRunner(threads=4)
    index = {label: i for i, label in enumerate(runner.labels)}
    stock_cols = {k: [index[l] for l in r.all_labels] for k, r in resolve_targets(runner.labels).items()}

    for path in files:
        starts, chunks = windows(load_audio(path), args.step)
        emb, scores = runner.run(chunks)
        proba = clf.predict(emb)
        print(f"\n{path.name}  ({len(chunks)} windows)")
        print(f"  {'target':<24}{'custom best':>12}{'  at':>8}{'>=0.70':>8}{'stock best':>12}{'>=0.70':>8}")
        rows = []
        for key, spec in BY_KEY.items():
            c_best = c_at = c_n = None
            if key in clf.target_keys:
                col = proba[:, clf.classes.index(key)]
                i = int(col.argmax())
                c_best, c_at, c_n = float(col[i]), float(starts[i]), int((col >= THRESHOLD).sum())
            s_best = s_n = None
            if stock_cols.get(key):
                col = scores[:, stock_cols[key]].max(axis=1)
                s_best, s_n = float(col.max()), int((col >= THRESHOLD).sum())
            rows.append((c_best or 0, key, spec, c_best, c_at, c_n, s_best, s_n))
        for _, key, target, c_best, c_at, c_n, s_best, s_n in sorted(rows, key=lambda r: -r[0])[:5]:
            at = None if c_at is None else f"{c_at:.1f}s"
            print(f"  {target.common_name:<24}{fmt(c_best, 12, '.3f')}{fmt(at, 8)}"
                  f"{fmt(c_n, 8)}{fmt(s_best, 12, '.3f')}{fmt(s_n, 8)}")
        bg = proba[:, clf.classes.index(clf.background)]
        print(f"  (background: median {np.median(bg):.2f}; top 5 targets by custom score shown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
