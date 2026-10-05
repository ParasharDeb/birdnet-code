"""Turn downloaded recordings into labelled BirdNET embeddings.

    .\\.venv\\Scripts\\python.exe -m training.extract

The hard part is the labels. A xeno-canto "Black Drongo" recording is mostly
silence and other birds, so labelling every 3 s window as Black Drongo would
teach the model that silence is a drongo. Instead each window is kept as a
positive only when there is evidence the bird is calling in it:

* species BirdNET already knows: the stock score for that species is at least
  ``--min-stock`` (deliberately low — BirdNET under-scores these birds, which is
  why we are training), or the window is clearly louder than the rest of the
  file and BirdNET doesn't confidently call it some other bird;
* species BirdNET doesn't know (White-winged Duck, Greater Adjutant): the
  loudness test alone.

Background examples come from recordings of other common local birds, from the
quiet stretches of target recordings, and from synthetic noise. Positives are
also copied with added noise, since the INMP441 in the field will be noisier
than a xeno-canto recordist's shotgun mic.

Per-recording results are cached, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import (
    BACKGROUND,
    DATA_DIR,
    FEATURES_PATH,
    SAMPLE_RATE,
    WINDOW,
    BirdNetRunner,
    band_energy,
    load_audio,
    windows,
)

CACHE_DIR = DATA_DIR / "cache"

_runner: BirdNetRunner | None = None
_target_labels: dict[str, list[int]] = {}


def _init_worker() -> None:
    global _runner, _target_labels
    from app.species import resolve_targets

    _runner = BirdNetRunner(threads=1)
    index = {label: i for i, label in enumerate(_runner.labels)}
    _target_labels = {
        key: [index[l] for l in res.all_labels]
        for key, res in resolve_targets(_runner.labels).items()
    }


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n)
    return (noise / (np.abs(noise).max() + 1e-9)).astype(np.float32)


def _add_noise(chunks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mix pink noise in at a random 3-15 dB SNR — a rough stand-in for a cheap mic outdoors."""
    out = np.empty_like(chunks)
    for i, c in enumerate(chunks):
        noise = _pink_noise(WINDOW, rng)
        snr_db = rng.uniform(3, 15)
        p_sig = np.mean(c ** 2) + 1e-12
        p_noise = np.mean(noise ** 2) + 1e-12
        out[i] = c + noise * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return out


def process(entry: dict, args: dict) -> dict:
    """Embed one recording and pick its labelled windows."""
    cache = CACHE_DIR / f"XC{entry['xc_id']}_{entry['target_key']}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    rng = np.random.default_rng(int(entry["xc_id"]))
    y = load_audio(DATA_DIR / entry["path"], max_seconds=args["max_seconds"])
    starts, chunks = windows(y, args["step"])
    empty = {"X": np.zeros((0, 1024), np.float16), "y": np.zeros(0, "U32"),
             "start": np.zeros(0, np.float32), "aug": np.zeros(0, bool)}
    if len(chunks) == 0:
        np.savez(cache, **empty)
        return empty

    emb, scores = _runner.run(chunks)
    energy = band_energy(chunks)
    loud = energy >= np.median(energy) + args["loud_db"]
    key = entry["target_key"]

    if key == BACKGROUND:
        any_target = np.zeros(len(chunks))
        for idx in _target_labels.values():
            if idx:
                any_target = np.maximum(any_target, scores[:, idx].max(axis=1))
        keep = np.flatnonzero(any_target < 0.05)
        keep = rng.permutation(keep)[: args["background_per_recording"]]
        X, labels, st, aug = emb[keep], [BACKGROUND] * len(keep), starts[keep], [False] * len(keep)
    else:
        idx = _target_labels.get(key, [])
        target = scores[:, idx].max(axis=1) if idx else np.zeros(len(chunks))
        others = scores.copy()
        if idx:
            others[:, idx] = 0
        other_best = others.max(axis=1)
        if idx:
            pos = (target >= args["min_stock"]) | (loud & (target >= 0.02) & (other_best < 0.3))
        else:
            pos = loud & (other_best < 0.5)
        pos_idx = np.flatnonzero(pos)

        # Quiet stretches of the same recording are good "nothing here" examples.
        quiet = np.flatnonzero((energy < np.median(energy)) & (target < 0.02) & ~pos)
        quiet = rng.permutation(quiet)[: args["quiet_per_recording"]]

        X = [emb[pos_idx], emb[quiet]]
        labels = [key] * len(pos_idx) + [BACKGROUND] * len(quiet)
        st = [starts[pos_idx], starts[quiet]]
        aug = [False] * (len(pos_idx) + len(quiet))
        if len(pos_idx) and args["augment"]:
            noisy, _ = _runner.run(_add_noise(chunks[pos_idx], rng))
            X.append(noisy)
            labels += [key] * len(pos_idx)
            st.append(starts[pos_idx])
            aug += [True] * len(pos_idx)
        X, st = np.concatenate(X), np.concatenate(st)

    result = {"X": np.asarray(X, np.float16), "y": np.asarray(labels, "U32"),
              "start": np.asarray(st, np.float32), "aug": np.asarray(aug, bool)}
    np.savez(cache, **result)
    return result


def synthetic_background(n: int, seed: int = 0) -> np.ndarray:
    """Silence, hiss, pink noise and low rumble/wind at several levels."""
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(n):
        kind = i % 4
        level = 10 ** rng.uniform(-4, -0.5)
        if kind == 0:
            c = rng.normal(0, 1e-5, WINDOW)
        elif kind == 1:
            c = rng.normal(0, level, WINDOW)
        elif kind == 2:
            c = _pink_noise(WINDOW, rng) * level * 3
        else:  # wind: strongly low-passed noise with slow gusts
            from scipy.signal import butter, sosfilt
            sos = butter(2, rng.uniform(80, 400), "lowpass", fs=SAMPLE_RATE, output="sos")
            gust = 1 + 0.8 * np.sin(np.linspace(0, rng.uniform(1, 6), WINDOW))
            c = sosfilt(sos, rng.normal(0, 1, WINDOW)) * gust
            c = c / (np.abs(c).max() + 1e-9) * level * 5
        chunks.append(np.clip(c, -1, 1))
    return np.asarray(chunks, np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--step", type=float, default=1.0, help="seconds between window starts")
    ap.add_argument("--max-seconds", type=float, default=90, help="only use the start of long recordings")
    ap.add_argument("--min-stock", type=float, default=0.10, help="stock BirdNET score that marks a window as positive")
    ap.add_argument("--loud-db", type=float, default=6.0, help="dB above the file's median that counts as a sound event")
    ap.add_argument("--background-per-recording", type=int, default=25)
    ap.add_argument("--quiet-per-recording", type=int, default=4)
    ap.add_argument("--synthetic", type=int, default=400, help="synthetic noise windows")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    args = ap.parse_args()

    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit("No manifest.json — run `python -m training.download` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    opts = {k: getattr(args, k) for k in (
        "step", "max_seconds", "min_stock", "loud_db",
        "background_per_recording", "quiet_per_recording", "augment")}
    X, y, groups, aug = [], [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        futures = {pool.submit(process, entry, opts): entry for entry in manifest}
        for i, fut in enumerate(as_completed(futures), 1):
            entry = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # a corrupt download shouldn't sink the run
                print(f"  skipped XC{entry['xc_id']}: {type(exc).__name__}: {exc}")
                continue
            X.append(r["X"]); y.append(r["y"]); aug.append(r["aug"])
            groups.append(np.full(len(r["y"]), entry["xc_id"], "U16"))
            if i % 25 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} recordings, {time.perf_counter() - started:.0f}s")

    if args.synthetic:
        _init_worker()
        emb, _ = _runner.run(synthetic_background(args.synthetic))
        X.append(emb.astype(np.float16)); y.append(np.full(len(emb), BACKGROUND, "U32"))
        aug.append(np.zeros(len(emb), bool))
        groups.append(np.asarray([f"synthetic{i % 20}" for i in range(len(emb))], "U16"))

    X, y, groups, aug = map(np.concatenate, (X, y, groups, aug))
    np.savez_compressed(FEATURES_PATH, X=X, y=y, groups=groups, aug=aug)

    print(f"\nSaved {len(y)} windows to {FEATURES_PATH}")
    labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(labels, counts):
        recs = len(np.unique(groups[y == label]))
        print(f"  {label:<24} {count:>6} windows from {recs:>4} recordings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
