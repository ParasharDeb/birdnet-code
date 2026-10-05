"""Train the target-species classifier on BirdNET embeddings.

    .\\.venv\\Scripts\\python.exe -m training.train

The model is a softmax logistic regression over the targets plus a
"background" class. Because it only has to choose between eleven options, a
clear call can score far higher than it does among BirdNET's 6,522 species.

Validation splits by recording (GroupKFold), never by window: neighbouring
windows of one recording are near-duplicates, and splitting them across train
and test would report accuracy the model doesn't have. Noise-augmented copies
are trained on but never scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import BACKGROUND, FEATURES_PATH, MODEL_PATH, MODELS_DIR

THRESHOLD = 0.70
MIN_RECORDINGS = 4


def make_model(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=3000),
    )


def cross_validate(X, y, groups, aug, c: float, folds: int) -> np.ndarray:
    """Out-of-fold probabilities for every non-augmented window."""
    classes = np.unique(y)
    proba = np.full((len(y), len(classes)), np.nan)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_model(c).fit(X[train_idx], y[train_idx])
        test_idx = test_idx[~aug[test_idx]]
        p = model.predict_proba(X[test_idx])
        # A fold can miss a rare class entirely; align columns by name.
        col = {cls: i for i, cls in enumerate(model.classes_)}
        for j, cls in enumerate(classes):
            proba[test_idx, j] = p[:, col[cls]] if cls in col else 0.0
    return proba


def summarise(y, groups, aug, proba, classes) -> dict:
    scored = ~aug & ~np.isnan(proba[:, 0])
    y_s, p_s, g_s = y[scored], proba[scored], groups[scored]
    col = {cls: i for i, cls in enumerate(classes)}
    targets = [c for c in classes if c != BACKGROUND]

    per_class = {}
    for cls in targets:
        mine = y_s == cls
        p = p_s[mine, col[cls]]
        # Recording-level: does the best window of each recording clear 0.70?
        rec_best = {}
        for g, v in zip(g_s[mine], p):
            rec_best[g] = max(rec_best.get(g, 0.0), v)
        per_class[cls] = {
            "windows": int(mine.sum()),
            "recordings": len(rec_best),
            "window_recall_at_0.70": float((p >= THRESHOLD).mean()) if len(p) else 0.0,
            "window_recall_at_0.50": float((p >= 0.5).mean()) if len(p) else 0.0,
            "recording_recall_at_0.70": float(np.mean([v >= THRESHOLD for v in rec_best.values()])) if rec_best else 0.0,
            "median_score": float(np.median(p)) if len(p) else 0.0,
        }

    bg = y_s == BACKGROUND
    target_cols = [col[c] for c in targets]
    bg_best = p_s[bg][:, target_cols].max(axis=1)
    # Which target wrongly fires on background, and how often.
    false_alarms = {}
    for cls in targets:
        false_alarms[cls] = int((p_s[bg][:, col[cls]] >= THRESHOLD).sum())
    return {
        "per_class": per_class,
        "background_windows": int(bg.sum()),
        "background_false_alarm_rate_at_0.70": float((bg_best >= THRESHOLD).mean()),
        "false_alarms_by_target": false_alarms,
        "accuracy": float((np.asarray(classes)[p_s.argmax(axis=1)] == y_s).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--c", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1])
    args = ap.parse_args()

    if not FEATURES_PATH.exists():
        sys.exit("No features.npz — run `python -m training.extract` first.")
    data = np.load(FEATURES_PATH)
    X, y, groups, aug = data["X"].astype(np.float32), data["y"], data["groups"], data["aug"]

    # A class needs enough separate recordings to be learned *and* validated.
    dropped = []
    for cls in np.unique(y):
        if cls != BACKGROUND and len(np.unique(groups[y == cls])) < MIN_RECORDINGS:
            dropped.append(cls)
    if dropped:
        print(f"Not enough recordings to train (need {MIN_RECORDINGS}+), left out: {', '.join(dropped)}")
        keep = ~np.isin(y, dropped)
        X, y, groups, aug = X[keep], y[keep], groups[keep], aug[keep]
    classes = list(np.unique(y))

    best = None
    for c in args.c:
        proba = cross_validate(X, y, groups, aug, c, args.folds)
        s = summarise(y, groups, aug, proba, classes)
        recalls = [v["window_recall_at_0.70"] for v in s["per_class"].values()]
        # Reward recall at 0.70, penalise false alarms on background 3x as hard.
        objective = float(np.mean(recalls)) - 3 * s["background_false_alarm_rate_at_0.70"]
        print(f"C={c:<6} mean recall@0.70 {np.mean(recalls):.3f}  "
              f"background false alarms@0.70 {s['background_false_alarm_rate_at_0.70']:.3f}  "
              f"accuracy {s['accuracy']:.3f}")
        if best is None or objective > best[0]:
            best = (objective, c, s)

    _, c, summary = best
    print(f"\nChosen C={c}. Held-out results (recordings the model never saw):\n")
    print(f"  {'species':<24}{'recs':>5}{'windows':>9}{'>=0.70':>9}{'>=0.50':>9}{'median':>8}{'recs >=0.70':>13}")
    for cls, m in summary["per_class"].items():
        print(f"  {cls:<24}{m['recordings']:>5}{m['windows']:>9}"
              f"{m['window_recall_at_0.70']:>9.0%}{m['window_recall_at_0.50']:>9.0%}"
              f"{m['median_score']:>8.2f}{m['recording_recall_at_0.70']:>13.0%}")
    print(f"\n  background windows wrongly scored >= 0.70 for some target: "
          f"{summary['background_false_alarm_rate_at_0.70']:.1%} of {summary['background_windows']}")

    model = make_model(c).fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "classes": list(model.classes_),
        "background": BACKGROUND,
        "embedding": "BirdNET_GLOBAL_6K_V2.4 penultimate layer (1024)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "C": c,
        "training_windows": int(len(y)),
        "validation": summary,
        "left_out": dropped,
    }
    joblib.dump(bundle, MODEL_PATH)
    (MODEL_PATH.with_suffix(".json")).write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "model"}, indent=1), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
