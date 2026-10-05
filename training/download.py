"""Download training recordings from xeno-canto (API v3).

Needs a free API key from https://xeno-canto.org/account:

    $env:XC_API_KEY = "..."
    .\\.venv\\Scripts\\python.exe -m training.download

Recordings are ranked by quality (A > B > C), then by whether they come from the
Indian subcontinent (closer to the birds this system will hear), then by length
(short recordings are mostly the bird; long ones are mostly everything else).
Re-running skips files that are already on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .common import AUDIO_DIR, DATA_DIR, HELD_OUT_XC_IDS, Source, background_sources, target_sources

API = "https://xeno-canto.org/api/3/recordings"
REGION = {"India", "Nepal", "Bangladesh", "Bhutan", "Sri Lanka", "Myanmar", "Pakistan"}
QUALITY_RANK = {"A": 0, "B": 1, "C": 2}
USER_AGENT = "birdnet-target-classifier/1.0 (research prototype)"


def _seconds(length: str) -> float:
    parts = [float(p) for p in str(length).split(":") if p]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def _query(session: requests.Session, key: str, query: str) -> list[dict]:
    recordings, page = [], 1
    while True:
        r = session.get(API, params={"query": query, "key": key, "page": page}, timeout=60)
        if r.status_code == 401:
            sys.exit(f"xeno-canto rejected the API key: {r.text[:200]}")
        r.raise_for_status()
        body = r.json()
        recordings += body.get("recordings", [])
        if page >= int(body.get("numPages", 1)) or page >= 5:
            return recordings
        page += 1


def find_recordings(session: requests.Session, key: str, sci: str) -> list[dict]:
    genus, _, species = sci.partition(" ")
    # v3 takes genus and species epithet as separate tags.
    recs = _query(session, key, f'gen:{genus} sp:{species}')
    # The domestic form of a goose may be filed under a subspecies; keep it.
    return [r for r in recs if f"{r.get('gen', '')} {r.get('sp', '')}".lower() == sci.lower()]


def pick(recs: list[dict], limit: int, max_seconds: float) -> list[dict]:
    usable = [
        r for r in recs
        if r.get("q") in QUALITY_RANK
        and str(r.get("id")) not in HELD_OUT_XC_IDS
        and 2 <= _seconds(r.get("length", "0")) <= max_seconds
        and r.get("file")
    ]
    usable.sort(key=lambda r: (
        QUALITY_RANK[r["q"]],
        r.get("cnt") not in REGION,
        _seconds(r["length"]),
    ))
    return usable[:limit]


def download(session: requests.Session, rec: dict, folder: Path) -> tuple[Path, bool]:
    path = folder / f"XC{rec['id']}.mp3"
    if path.exists() and path.stat().st_size > 0:
        return path, False
    url = rec["file"] if rec["file"].startswith("http") else "https:" + rec["file"]
    for attempt in range(3):
        try:
            with session.get(url, timeout=120, stream=True) as r:
                r.raise_for_status()
                tmp = path.with_suffix(".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                tmp.replace(path)
            return path, True
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download XC{rec['id']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-target", type=int, default=60, help="recordings per target species")
    ap.add_argument("--per-background", type=int, default=12, help="recordings per background species")
    ap.add_argument("--max-seconds", type=float, default=120, help="skip recordings longer than this")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    key = os.environ.get("XC_API_KEY", "").strip()
    if not key:
        sys.exit("Set XC_API_KEY to your xeno-canto API key (https://xeno-canto.org/account).")

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    manifest: list[dict] = []
    jobs: list[tuple[dict, Path, Source]] = []

    sources = [(s, args.per_target) for s in target_sources()]
    sources += [(s, args.per_background) for s in background_sources()]
    for source, limit in sources:
        recs = find_recordings(session, key, source.scientific_name)
        chosen = pick(recs, limit, args.max_seconds)
        folder = AUDIO_DIR / source.scientific_name.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        print(f"{source.scientific_name:<28} -> {source.target_key:<22} "
              f"{len(recs):>4} on xeno-canto, using {len(chosen)}")
        for rec in chosen:
            jobs.append((rec, folder, source))

    print(f"\nDownloading {len(jobs)} recordings...")
    new = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [(pool.submit(download, session, rec, folder), rec, source) for rec, folder, source in jobs]
        for i, (fut, rec, source) in enumerate(futures, 1):
            try:
                path, fresh = fut.result()
            except RuntimeError as exc:
                print(f"  {exc}")
                continue
            new += fresh
            manifest.append({
                "path": str(path.relative_to(DATA_DIR)),
                "xc_id": str(rec["id"]),
                "scientific_name": source.scientific_name,
                "target_key": source.target_key,
                "quality": rec.get("q"),
                "country": rec.get("cnt"),
                "type": rec.get("type"),
                "length": rec.get("length"),
                "recordist": rec.get("rec"),
                "license": rec.get("lic"),
                "background_species": rec.get("also", []),
            })
            if i % 50 == 0:
                print(f"  {i}/{len(futures)}")

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nDone: {len(manifest)} recordings ({new} new). Manifest: {DATA_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
