"""Runtime configuration. Every field can be overridden with a BIRDNET_* env var."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BIRDNET_", env_file=".env", extra="ignore"
    )

    # --- analysis defaults (each is overridable per request) ---
    min_confidence: float = 0.25          # BirdNET's own default
    sensitivity: float = 1.0              # 0.5 - 1.5, higher = more detections
    overlap: float = 0.0                  # seconds of overlap between 3s windows, 0 - 2.9

    # Default location used for BirdNET's range filter when a request omits it.
    # None disables the range filter entirely (all 6k species stay eligible).
    latitude: float | None = None
    longitude: float | None = None
    # Keep target species eligible even when the range filter would drop them
    # (it is built from wild-bird sightings, so it drops domestic geese).
    range_filter_keeps_targets: bool = True

    # --- custom target classifier (built by training/train.py) ---
    # Relative paths resolve against the project root. Missing file = targets
    # are scored by stock BirdNET only.
    custom_model_path: str = "models/target_classifier.joblib"
    embedding_threads: int = 4
    # A target scoring at least this is reported as "confirmed".
    confirm_threshold: float = 0.70

    # --- upload limits ---
    max_upload_mb: int = 200
    allowed_extensions: str = "wav,wave,mp3,flac,ogg,oga,m4a,aac,mp4,aiff,aif"

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # Directory that /analyze/path is allowed to read from. Empty = endpoint disabled.
    local_audio_dir: str = ""

    @property
    def extension_set(self) -> set[str]:
        return {
            "." + e.strip().lower().lstrip(".")
            for e in self.allowed_extensions.split(",")
            if e.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
