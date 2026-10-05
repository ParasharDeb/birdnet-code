"""Request and response models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class Detection(BaseModel):
    common_name: str
    scientific_name: str
    label: str
    start_time: float = Field(description="Seconds from the start of the recording.")
    end_time: float
    confidence: float
    species_key: str | None = Field(
        default=None, description="Target key, or null for a non-target species."
    )
    emoji: str | None = None
    source: str = Field(
        default="birdnet", description='"custom" (target classifier) or "birdnet" (stock model).'
    )


class SpeciesSummary(BaseModel):
    """One row per target species — present whether or not it was heard."""

    key: str
    common_name: str
    scientific_name: str
    emoji: str
    available_in_model: bool = Field(
        description="False means neither BirdNET nor the custom classifier can detect it."
    )
    scored_by: str | None = Field(default=None, description='"custom", "birdnet" or null.')
    model_label: str | None = None
    detected: bool
    confirmed: bool = Field(
        default=False, description="max_confidence reached the confirm threshold (default 0.70)."
    )
    detection_count: int
    max_confidence: float | None = None
    first_detection_time: float | None = None


class AnalysisResponse(BaseModel):
    filename: str
    duration_seconds: float | None
    analysis_seconds: float
    min_confidence: float
    sensitivity: float
    overlap: float
    latitude: float | None
    longitude: float | None
    date: date | None
    species_summary: list[SpeciesSummary]
    target_detections: list[Detection]
    other_detections: list[Detection] | None = Field(
        default=None,
        description="Non-target species; only populated when include_non_target=true.",
    )
    total_detections: int


class SpeciesInfo(BaseModel):
    key: str
    common_name: str
    scientific_name: str
    emoji: str
    available_in_model: bool
    scored_by: str | None = None
    model_label: str | None
    model_extra_labels: list[str] = Field(
        default_factory=list, description="Other model labels that also count as this target."
    )
    model_common_name: str | None
    model_scientific_name: str | None
    matched_on: str | None


class SpeciesResponse(BaseModel):
    model_labels_total: int
    targets: list[SpeciesInfo]
    unavailable: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_labels_total: int | None
    targets_resolved: int
    targets_total: int
    custom_model_loaded: bool = False
    custom_model_trained_at: str | None = None
    detail: str | None = None
