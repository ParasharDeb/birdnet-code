"""A process-wide BirdNET analyzer.

The TFLite interpreter behind birdnetlib holds mutable tensor state, so it is
not safe to run two analyses through it concurrently. Everything here funnels
through one lock; FastAPI's threadpool keeps the event loop responsive while a
request waits for it.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from pathlib import Path

from .config import settings
from .custom_model import TargetClassifier, load_classifier
from .species import BY_KEY, Resolution, resolve_targets

logger = logging.getLogger(__name__)


class ModelNotReady(RuntimeError):
    pass


class BirdNetService:
    def __init__(self) -> None:
        self._analyzer = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self._resolutions: dict[str, Resolution] = {}
        self._labels: list[str] = []
        self._classifier: TargetClassifier | None = None
        self._runner = None  # BirdNetRunner, only built when a classifier exists

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        """Build the analyzer once at startup so the first request isn't slow."""
        from birdnetlib.analyzer import Analyzer

        started = time.perf_counter()
        try:
            with self._lock:
                self._analyzer = Analyzer()
                self._labels = list(self._analyzer.labels)
            self._resolutions = resolve_targets(self._labels)
            if settings.range_filter_keeps_targets:
                self._keep_targets_through_range_filter()
            self._load_error = None
        except Exception as exc:  # pragma: no cover - depends on model download
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("BirdNET model failed to load")
            return

        self.load_classifier()

        missing = [
            r.species.common_name for k, r in self._resolutions.items() if not self.available(k)
        ]
        logger.info(
            "BirdNET loaded in %.1fs with %d labels; %d/%d target species detectable",
            time.perf_counter() - started,
            len(self._labels),
            len(self._resolutions) - len(missing),
            len(self._resolutions),
        )
        if missing:
            logger.warning(
                "Not in BirdNET or the custom classifier, so they can never be detected: %s",
                ", ".join(missing),
            )

    def _keep_targets_through_range_filter(self) -> None:
        """Never let BirdNET's location filter drop a target species.

        The filter is built from eBird sightings of *wild* birds, so it removes
        species we know are here: domestic geese (Graylag is only "expected" in
        Assam in winter, Swan Goose never) and even Indian Peafowl. Targets stay
        eligible year-round; everything else is still filtered as before.
        """
        analyzer = self._analyzer
        original = analyzer.return_predicted_species_list
        target_labels = [l for r in self._resolutions.values() for l in r.all_labels]

        def with_targets(*args, **kwargs):
            species = list(original(*args, **kwargs))
            return species + [l for l in target_labels if l not in species]

        analyzer.return_predicted_species_list = with_targets
        analyzer.cached_species_lists.clear()

    def load_classifier(self) -> None:
        """(Re)load the custom target classifier, if one has been trained."""
        path = Path(settings.custom_model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        classifier = load_classifier(path)
        with self._lock:
            self._classifier = classifier
            if classifier and self._runner is None:
                from .embeddings import BirdNetRunner

                self._runner = BirdNetRunner(threads=settings.embedding_threads)

    # ----------------------------------------------------------- properties

    @property
    def ready(self) -> bool:
        return self._analyzer is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def resolutions(self) -> dict[str, Resolution]:
        return self._resolutions

    @property
    def classifier(self) -> TargetClassifier | None:
        return self._classifier

    def scored_by(self, key: str) -> str | None:
        """Which model scores this target: "custom", "birdnet", or None if neither can."""
        if self._classifier and key in self._classifier.target_keys:
            return "custom"
        res = self._resolutions.get(key)
        return "birdnet" if res and res.resolved else None

    def available(self, key: str) -> bool:
        return self.scored_by(key) is not None

    def label_to_key(self) -> dict[str, str]:
        return {
            label: key for key, r in self._resolutions.items() for label in r.all_labels
        }

    # ------------------------------------------------------------- analysis

    def analyze(
        self,
        path: Path,
        *,
        min_confidence: float,
        sensitivity: float,
        overlap: float,
        latitude: float | None,
        longitude: float | None,
        on_date: date | None,
    ) -> tuple[list[dict], float | None]:
        """Run BirdNET over one file. Returns (detections, duration_seconds)."""
        if self._analyzer is None:
            raise ModelNotReady(self._load_error or "model not loaded")

        from birdnetlib import Recording

        kwargs: dict = {
            "min_conf": min_confidence,
            "sensitivity": sensitivity,
            "overlap": overlap,
        }
        # BirdNET's range filter needs latitude, longitude and a date together;
        # passing a partial set silently disables it, so require all three.
        if latitude is not None and longitude is not None and on_date is not None:
            kwargs.update(lat=latitude, lon=longitude, date=on_date)

        with self._lock:
            recording = Recording(self._analyzer, str(path), **kwargs)
            recording.analyze()
            detections = list(recording.detections)
            duration = getattr(recording, "duration", None)
            classifier, runner = self._classifier, self._runner
            if classifier is not None:
                custom = self._custom_detections(path, classifier, runner, overlap, min_confidence)

        if classifier is not None:
            # Targets the classifier covers are scored by it alone; the stock
            # model's (lower) scores for them would only be duplicates.
            label_to_key = self.label_to_key()
            detections = [
                d for d in detections
                if label_to_key.get(f"{d['scientific_name']}_{d['common_name']}")
                not in classifier.target_keys
            ] + custom

        return detections, duration

    @staticmethod
    def _custom_detections(path, classifier, runner, overlap, min_confidence) -> list[dict]:
        from .embeddings import WINDOW_SECONDS, load_audio, windows

        starts, chunks = windows(load_audio(path), WINDOW_SECONDS - overlap)
        embeddings, _ = runner.run(chunks)
        proba = classifier.predict(embeddings)
        out = []
        for col, key in enumerate(classifier.classes):
            if key == classifier.background:
                continue
            spec = BY_KEY[key]
            for row in (proba[:, col] >= min_confidence).nonzero()[0]:
                out.append({
                    "common_name": spec.common_name,
                    "scientific_name": spec.scientific_name,
                    "label": f"custom:{key}",
                    "species_key": key,
                    "source": "custom",
                    "start_time": float(starts[row]),
                    "end_time": float(starts[row]) + WINDOW_SECONDS,
                    "confidence": float(proba[row, col]),
                })
        return out


service = BirdNetService()
