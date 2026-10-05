"""The custom target-species classifier trained by ``training/train.py``.

It scores BirdNET embeddings against just the targets plus a "background"
class, so a clear call scores far higher than it does among BirdNET's 6,522
species. Targets the classifier covers are scored by it; any it left out
(too little training data) fall back to stock BirdNET.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TargetClassifier:
    def __init__(self, path: Path) -> None:
        import joblib

        # joblib unpickles: only load a model file you trained yourself.
        bundle = joblib.load(path)
        self.path = path
        self.model = bundle["model"]
        self.classes: list[str] = list(bundle["classes"])
        self.background: str = bundle["background"]
        self.trained_at: str | None = bundle.get("trained_at")
        self.target_keys: set[str] = {c for c in self.classes if c != self.background}

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Class probabilities, columns in ``self.classes`` order."""
        if len(embeddings) == 0:
            return np.zeros((0, len(self.classes)))
        return self.model.predict_proba(embeddings.astype(np.float32))


def load_classifier(path: Path) -> TargetClassifier | None:
    if not path.is_file():
        logger.info("No custom classifier at %s; targets use stock BirdNET scores", path)
        return None
    try:
        clf = TargetClassifier(path)
    except Exception:
        logger.exception("Custom classifier at %s failed to load; using stock BirdNET", path)
        return None
    logger.info(
        "Custom classifier loaded (%s, trained %s): %s",
        path.name, clf.trained_at, ", ".join(sorted(clf.target_keys)),
    )
    return clf
