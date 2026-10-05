"""Audio windowing and a batched BirdNET runner that exposes embeddings.

Shared by the API (to score windows with the custom classifier) and by the
training pipeline (to build its dataset), so both see identical features.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

SAMPLE_RATE = 48_000
WINDOW_SECONDS = 3.0
WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS)


# --------------------------------------------------------------------- audio


def load_audio(path, max_seconds: float | None = None) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=max_seconds)
    return y.astype(np.float32)


def windows(y: np.ndarray, step_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Cut audio into 3 s windows every ``step_seconds``, padding a short tail.

    Mirrors birdnetlib's chunking (a tail shorter than 1.5 s is dropped), so
    window start times line up with the API's detections.
    """
    step = int(step_seconds * SAMPLE_RATE)
    starts, chunks = [], []
    for i in range(0, max(len(y), 1), step):
        chunk = y[i : i + WINDOW]
        if len(chunk) < int(1.5 * SAMPLE_RATE):
            break
        if len(chunk) < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
        starts.append(i / SAMPLE_RATE)
        chunks.append(chunk)
    if not chunks:
        return np.zeros(0, dtype=np.float32), np.zeros((0, WINDOW), dtype=np.float32)
    return np.asarray(starts, dtype=np.float32), np.stack(chunks).astype(np.float32)


def band_energy(chunks: np.ndarray, low_hz: float = 1000, high_hz: float = 10000) -> np.ndarray:
    """Per-window energy in the band where most bird calls sit (dB)."""
    spec = np.abs(np.fft.rfft(chunks, axis=1)) ** 2
    freqs = np.fft.rfftfreq(chunks.shape[1], 1 / SAMPLE_RATE)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return 10 * np.log10(spec[:, band].mean(axis=1) + 1e-12)


# -------------------------------------------------------------------- BirdNET


class BirdNetRunner:
    """Runs the stock BirdNET TFLite model on batches of 3 s windows.

    One invoke yields both the 1024-dim embedding (the layer before the
    classifier) and BirdNET's own 6,522 species scores.
    """

    def __init__(self, threads: int = 1, batch_size: int = 16) -> None:
        from birdnetlib.analyzer import LABEL_PATH, MODEL_PATH as BN_MODEL_PATH

        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            from tensorflow import lite as tflite

        self.batch_size = batch_size
        self.interpreter = tflite.Interpreter(
            model_path=BN_MODEL_PATH,
            num_threads=threads,
            # The embedding is an intermediate tensor. The default XNNPACK
            # delegate frees it after invoke; the builtin kernels keep it.
            # Throughput scales better with one thread per process than with
            # more threads, so the pipeline parallelises across processes.
            experimental_op_resolver_type=tflite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.embedding_index = self.output_index - 1
        with open(LABEL_PATH, encoding="utf-8") as fh:
            self.labels = [line.rstrip("\n") for line in fh]
        self._batch = None

    def _resize(self, n: int) -> None:
        if self._batch != n:
            self.interpreter.resize_tensor_input(self.input_index, [n, WINDOW])
            self.interpreter.allocate_tensors()
            self._batch = n

    def run(self, chunks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (embeddings [n, 1024], scores [n, labels]) for [n, 144000] audio."""
        embs, scores = [], []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            self._resize(len(batch))
            self.interpreter.set_tensor(self.input_index, batch)
            self.interpreter.invoke()
            embs.append(self.interpreter.get_tensor(self.embedding_index).copy())
            logits = self.interpreter.get_tensor(self.output_index)
            scores.append(1 / (1 + np.exp(-np.clip(logits, -15, 15))))
        if not embs:
            return np.zeros((0, 1024), np.float32), np.zeros((0, len(self.labels)), np.float32)
        return np.concatenate(embs), np.concatenate(scores)
