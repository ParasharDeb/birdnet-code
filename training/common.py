"""Shared pieces of the custom-classifier pipeline: paths and species lists.
Audio windowing and the BirdNET runner live in ``app.embeddings`` so the API and
the pipeline compute identical features.

The classifier is a small model trained on BirdNET's 1024-dim embeddings. That is
the same idea as BirdNET-Analyzer's ``train.py``, but kept in-process so the API
can still use the stock model for every non-target species and the range filter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import (  # noqa: F401  (re-exported for the pipeline)
    SAMPLE_RATE,
    WINDOW,
    WINDOW_SECONDS,
    BirdNetRunner,
    band_energy,
    load_audio,
    windows,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BIRDNET_TRAINING_DATA", ROOT / "training" / "data"))
AUDIO_DIR = DATA_DIR / "audio"
FEATURES_PATH = DATA_DIR / "features.npz"
MODELS_DIR = ROOT / "models"
# Same env var the API reads, so both agree on where the model lives.
MODEL_PATH = Path(os.environ.get("BIRDNET_CUSTOM_MODEL_PATH", MODELS_DIR / "target_classifier.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH

BACKGROUND = "background"

# xeno-canto IDs of the files in public/ — never train on what we test with.
HELD_OUT_XC_IDS = {"1015740", "1175047", "1180528", "1069298", "1087518", "1088310"}


@dataclass(frozen=True)
class Source:
    """One xeno-canto species query that feeds a training class."""

    scientific_name: str
    target_key: str  # a key from app.species, or BACKGROUND


def target_sources() -> list[Source]:
    """Every target species, plus the extra species that count as it."""
    from app.species import TARGET_SPECIES

    sources = []
    for spec in TARGET_SPECIES:
        sources.append(Source(spec.scientific_name, spec.key))
        for sci in spec.also_counts:
            sources.append(Source(sci, spec.key))
    return sources


# Common birds of North-East India that are not targets. They teach the model
# what "some other bird" sounds like. Several are deliberate near-misses: Ashy
# Drongo vs Black Drongo, Shikra vs Black Kite, Tree Sparrow vs House Sparrow,
# Rose-ringed vs Alexandrine Parakeet, Lesser Whistling-Duck vs White-winged Duck.
BACKGROUND_SPECIES = (
    "Dicrurus leucophaeus",      # Ashy Drongo
    "Dicrurus hottentottus",     # Hair-crested Drongo
    "Accipiter badius",          # Shikra
    "Haliastur indus",           # Brahminy Kite
    "Passer montanus",           # Eurasian Tree Sparrow
    "Psittacula krameri",        # Rose-ringed Parakeet
    "Dendrocygna javanica",      # Lesser Whistling-Duck
    "Nettapus coromandelianus",  # Cotton Pygmy-Goose
    "Anas poecilorhyncha",       # Indian Spot-billed Duck
    "Athene brama",              # Spotted Owlet
    "Pycnonotus cafer",          # Red-vented Bulbul
    "Acridotheres tristis",      # Common Myna
    "Gracupica contra",          # Indian Pied Starling
    "Eudynamys scolopaceus",     # Asian Koel
    "Copsychus saularis",        # Oriental Magpie-Robin
    "Corvus splendens",          # House Crow
    "Corvus macrorhynchos",      # Large-billed Crow
    "Spilopelia chinensis",      # Spotted Dove
    "Orthotomus sutorius",       # Common Tailorbird
    "Psilopogon haemacephalus",  # Coppersmith Barbet
    "Centropus sinensis",        # Greater Coucal
    "Vanellus indicus",          # Red-wattled Lapwing
    "Halcyon smyrnensis",        # White-breasted Kingfisher
    "Ardeola grayii",            # Indian Pond-Heron
    "Gallus gallus",             # Red Junglefowl (also domestic chickens)
)


def background_sources() -> list[Source]:
    return [Source(sci, BACKGROUND) for sci in BACKGROUND_SPECIES]
