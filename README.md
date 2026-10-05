# BirdNET Species API

A FastAPI server that runs [BirdNET](https://github.com/kahst/BirdNET-Analyzer) over an
audio recording and reports which of ten target species it heard, with timestamps
and confidence scores.

| | Species | Scientific name | BirdNET v2.4 label |
|---|---|---|---|
| 🦚 | Indian Peafowl | *Pavo cristatus* | ✅ `Pavo cristatus_Indian Peafowl` |
| 🦜 | Alexandrine Parakeet | *Psittacula eupatria* | ✅ `Psittacula eupatria_Alexandrine Parakeet` |
| 🐦 | Greater Adjutant | *Leptoptilos dubius* | ❌ **not in the model** |
| 🦅 | Crested Serpent-Eagle | *Spilornis cheela* | ✅ `Spilornis cheela_Crested Serpent-Eagle` |
| 🦉 | Brown Fish-Owl | *Ketupa zeylonensis* | ✅ `Ketupa zeylonensis_Brown Fish-Owl` |
| 🐦 | Black Drongo | *Dicrurus macrocercus* | ✅ `Dicrurus macrocercus_Black Drongo` |
| 🐦 | Black Kite | *Milvus migrans* | ✅ `Milvus migrans_Black Kite` |
| 🐦 | House Sparrow | *Passer domesticus* | ✅ `Passer domesticus_House Sparrow` |
| 🦆 | White-winged Duck (Deohah) | *Asarcornis scutulata* | ❌ **not in the model** |
| 🪿 | Domestic Goose | *Anser anser* / *A. cygnoides* (domestic) | ✅ `Anser anser_Graylag Goose` + `Anser cygnoides_Swan Goose` |

BirdNET always classifies against its full 6,522-species label set; this server
filters those results down to the targets. Non-target detections are still
available via `include_non_target=true`, which is useful for sanity-checking a
recording.

### Domestic geese

BirdNET has no domestic-goose class. Domestic geese descend from the Graylag
Goose (European and Indian breeds) and the Swan Goose (Chinese and African
breeds) and sound like them, so a detection of either label is reported as
`domestic_goose`. A wild Graylag will also be reported as a domestic goose.

### The range filter never drops a target

BirdNET's location filter is built from sightings of wild birds. For Assam it
removes Graylag Goose from about April to September, Swan Goose all year, and
Indian Peafowl too. Target species are therefore always kept eligible, while
every other species is still filtered as usual. Set
`BIRDNET_RANGE_FILTER_KEEPS_TARGETS=false` to restore the stock behaviour.

### White-winged Duck is not in BirdNET

*Asarcornis scutulata* (formerly *Cairina scutulata*) has no label in BirdNET
v2.4. It is listed with `available_in_model: false`, the same as Greater
Adjutant below, and detecting it would need a custom classifier.

### Greater Adjutant is not in BirdNET

Verified against the loaded model: *Leptoptilos dubius* has no label in BirdNET
v2.4. The only bird in the genus is *Leptoptilos crumenifer* (Marabou Stork), a
different African species — matching against it would produce wrong answers, so
the server does not.

The API still lists Greater Adjutant in every response, with
`available_in_model: false`, so a client shows a truthful "cannot detect" rather
than a silent absence that looks like "not heard". To actually detect it you would
need to train a custom classifier on BirdNET's embeddings
(`BirdNET-Analyzer`'s `train.py`) from labelled Greater Adjutant recordings, then
point this server at the resulting model. Worth knowing before investing in that:
the species is largely non-vocal, communicating mostly by bill-clattering,
grunts and hisses at the colony — audio is a weak detection channel for it.

## Setup

```powershell
uv venv --python 3.12 .venv
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip install -r requirements.txt
```

Or with plain pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Verify the species are in the model

Species names shift between taxonomy versions, and not every bird is in BirdNET.
Run this first — it tells you exactly which label each target resolved to, and
flags any that the model does not contain:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_species
```

A species reported as `MISSING` can never be detected. The script prints similar
labels from the model so you can add the right spelling to `scientific_aliases` /
`common_aliases` in `app/species.py`.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

The first startup downloads the BirdNET model, so it takes a while; after that it
is cached. Interactive docs: <http://127.0.0.1:8000/docs>

## Endpoints

### `POST /analyze`

Multipart upload. Form fields:

| Field | Default | Meaning |
|---|---|---|
| `file` | — | Audio: wav, mp3, flac, ogg, m4a, aac, aiff |
| `min_confidence` | `0.25` | Drop detections below this score (0–1) |
| `sensitivity` | `1.0` | 0.5–1.5; higher finds more, with more false positives |
| `overlap` | `0.0` | 0–2.9 s overlap between BirdNET's 3 s windows |
| `latitude`, `longitude` | unset | Location for BirdNET's range filter |
| `recorded_on` | unset | `YYYY-MM-DD`, for the seasonal range filter |
| `include_non_target` | `false` | Also return other species heard |

The range filter only engages when latitude, longitude **and** date are all supplied.
It suppresses species that are implausible for that place and time of year, which
cuts false positives considerably — worth passing whenever you know them.

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze `
  -F "file=@recording.wav" `
  -F "min_confidence=0.15" `
  -F "latitude=26.14" -F "longitude=91.79" `
  -F "recorded_on=2026-01-20"
```

Response:

```json
{
  "filename": "recording.wav",
  "duration_seconds": 60.0,
  "analysis_seconds": 2.41,
  "min_confidence": 0.15,
  "species_summary": [
    {
      "key": "indian_peafowl",
      "common_name": "Indian Peafowl",
      "emoji": "🦚",
      "available_in_model": true,
      "detected": true,
      "detection_count": 3,
      "max_confidence": 0.8712,
      "first_detection_time": 12.0
    }
  ],
  "target_detections": [
    {
      "common_name": "Indian Peafowl",
      "scientific_name": "Pavo cristatus",
      "start_time": 12.0,
      "end_time": 15.0,
      "confidence": 0.8712,
      "species_key": "indian_peafowl",
      "emoji": "🦚"
    }
  ],
  "total_detections": 9
}
```

`species_summary` always lists every target, detected or not, so a client can
render a fixed set of rows without special-casing absences.

### `POST /analyze/path`

Analyzes a file already on the server instead of re-uploading it. Disabled unless
`BIRDNET_LOCAL_AUDIO_DIR` is set; reads are confined to that directory.

```powershell
$env:BIRDNET_LOCAL_AUDIO_DIR = "D:\recordings"
curl.exe -X POST "http://127.0.0.1:8000/analyze/path?path=night/2026-01-20.wav"
```

### `GET /species`

Which BirdNET label each target resolved to, and which targets the model lacks.

### `GET /health`

Model load state plus how many of the targets resolved. Reports `degraded`
rather than failing if the model could not load.

## Custom target classifier

Stock BirdNET rarely scores these birds above 0.70: it has to pick among 6,522
species, and xeno-canto test recordings of Black Kite and Black Drongo peaked at
0.43 and 0.51. The `training/` pipeline trains a small classifier on BirdNET's
embeddings that only chooses between the targets and "background", so clear
calls score much higher. It can also learn White-winged Duck and Greater
Adjutant, which stock BirdNET cannot detect at all, if xeno-canto has enough
recordings of them.

```powershell
$env:XC_API_KEY = "<your key from https://xeno-canto.org/account>"
.\.venv\Scripts\python.exe -m training.download   # ~1 GB of recordings into training\data\
.\.venv\Scripts\python.exe -m training.extract    # BirdNET embeddings, ~30-60 min on 12 cores
.\.venv\Scripts\python.exe -m training.train      # fits and validates, writes models\target_classifier.joblib
.\.venv\Scripts\python.exe -m training.evaluate   # compares with stock BirdNET on the files in public\
```

* **What becomes a training example.** Only 3 s windows with evidence the bird
  is calling: a stock BirdNET score of at least 0.10 for that species, or a
  window clearly louder than the rest of the recording that BirdNET doesn't
  confidently assign to another bird. Background examples come from 25 other
  common local birds (including look-alikes such as Ashy Drongo, Shikra, Tree
  Sparrow and Rose-ringed Parakeet), quiet stretches, and synthetic wind/noise.
  Every positive is also copied with added noise to mimic a cheap mic outdoors.
* **Validation** splits by recording, so the reported scores come from
  recordings the model never trained on. `train` prints, per species, how many
  held-out calls reach 0.70 and how often background is wrongly scored that high.
* **The files in `public/` are never used for training**, so `evaluate` is an
  honest check.
* **The server loads the model automatically** if `models/target_classifier.joblib`
  exists (override with `BIRDNET_CUSTOM_MODEL_PATH`). Targets the model covers are
  scored by it (`"source": "custom"`); others fall back to stock BirdNET. Every
  `species_summary` row has `scored_by` and `confirmed` (best score >=
  `BIRDNET_CONFIRM_THRESHOLD`, default 0.70). Restart the server after retraining.
* A species with fewer than 4 usable recordings is left out of the model and
  stays on stock BirdNET; `train` says which.
* **Recordings from your own INMP441 help the most.** Put labelled clips in
  `training\data\audio\<folder>\`, add them to `manifest.json` with the
  right `target_key` (or `background`), and re-run extract and train.

xeno-canto recordings are under Creative Commons licences, mostly
non-commercial; `manifest.json` records each one's recordist and licence.

## Configuration

Every field in `app/config.py` reads a `BIRDNET_`-prefixed env var, or a `.env` file:

```ini
BIRDNET_MIN_CONFIDENCE=0.2
BIRDNET_SENSITIVITY=1.25
BIRDNET_LATITUDE=26.14
BIRDNET_LONGITUDE=91.79
BIRDNET_MAX_UPLOAD_MB=500
BIRDNET_LOCAL_AUDIO_DIR=D:\recordings
```

## Notes on accuracy and performance

- Measured on this machine: ~2–3 s to analyse 60 s of audio once warm (~25× realtime,
  CPU only). The first request after startup takes ~8 s while TensorFlow warms up.
- BirdNET works on 3-second windows of 48 kHz audio; anything shorter than 3 s is
  padded and scores poorly. Prefer clips of 10 s or more.
- Confidence is not probability. Treat `< 0.3` as a lead worth listening to rather
  than a detection, and verify by ear.
- **Pass latitude, longitude and date if you know them.** Without the range filter,
  10 s of pure synthetic noise scored *Lesser Spotted Woodpecker* at 0.52 in testing —
  BirdNET has no "none of the above" class, so it distributes probability over
  whatever it has. The range filter removes species that are geographically
  impossible and is the single biggest false-positive reduction available.
- Crested Serpent-Eagle is best detected by its distinctive flight whistle; expect
  lower recall on perched birds.
- The analyzer holds mutable interpreter state, so requests are serialised through
  a lock. Run several worker processes (`--workers N`) for parallel throughput;
  each loads its own copy of the model.
