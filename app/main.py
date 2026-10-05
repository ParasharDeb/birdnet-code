"""FastAPI server that runs BirdNET and reports the target species."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse

from .analyzer import ModelNotReady, service
from .config import settings
from .schemas import (
    AnalysisResponse,
    Detection,
    HealthResponse,
    SpeciesInfo,
    SpeciesResponse,
    SpeciesSummary,
)
from .species import TARGET_SPECIES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("birdnet-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading BirdNET model (the first run downloads it)...")
    service.load()
    yield


app = FastAPI(
    title="BirdNET Species API",
    version="1.0.0",
    summary="Detect a fixed set of target bird species in audio recordings.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------- utils


def _build_response(
    *,
    filename: str,
    detections: list[dict],
    duration: float | None,
    analysis_seconds: float,
    min_confidence: float,
    sensitivity: float,
    overlap: float,
    latitude: float | None,
    longitude: float | None,
    on_date: date | None,
    include_non_target: bool,
) -> AnalysisResponse:
    label_to_key = service.label_to_key()
    resolutions = service.resolutions

    targets: list[Detection] = []
    others: list[Detection] = []

    for det in detections:
        label = det.get("label") or f"{det['scientific_name']}_{det['common_name']}"
        key = det.get("species_key") or label_to_key.get(label)
        item = Detection(
            common_name=det["common_name"],
            scientific_name=det["scientific_name"],
            label=label,
            start_time=round(float(det["start_time"]), 2),
            end_time=round(float(det["end_time"]), 2),
            confidence=round(float(det["confidence"]), 4),
            species_key=key,
            emoji=resolutions[key].species.emoji if key else None,
            source=det.get("source", "birdnet"),
        )
        (targets if key else others).append(item)

    targets.sort(key=lambda d: (d.start_time, -d.confidence))
    others.sort(key=lambda d: -d.confidence)

    summary: list[SpeciesSummary] = []
    for spec in TARGET_SPECIES:
        res = resolutions.get(spec.key)
        hits = [d for d in targets if d.species_key == spec.key]
        best = max((d.confidence for d in hits), default=None)
        summary.append(
            SpeciesSummary(
                key=spec.key,
                common_name=spec.common_name,
                scientific_name=spec.scientific_name,
                emoji=spec.emoji,
                available_in_model=service.available(spec.key),
                scored_by=service.scored_by(spec.key),
                model_label=res.label if res else None,
                detected=bool(hits),
                confirmed=best is not None and best >= settings.confirm_threshold,
                detection_count=len(hits),
                max_confidence=best,
                first_detection_time=min((d.start_time for d in hits), default=None),
            )
        )

    return AnalysisResponse(
        filename=filename,
        duration_seconds=round(duration, 2) if duration else None,
        analysis_seconds=round(analysis_seconds, 2),
        min_confidence=min_confidence,
        sensitivity=sensitivity,
        overlap=overlap,
        latitude=latitude,
        longitude=longitude,
        date=on_date,
        species_summary=summary,
        target_detections=targets,
        other_detections=others if include_non_target else None,
        total_detections=len(detections),
    )


def _validate_params(min_confidence: float, sensitivity: float, overlap: float) -> None:
    if not 0.0 <= min_confidence <= 1.0:
        raise HTTPException(422, "min_confidence must be between 0.0 and 1.0")
    if not 0.5 <= sensitivity <= 1.5:
        raise HTTPException(422, "sensitivity must be between 0.5 and 1.5")
    if not 0.0 <= overlap <= 2.9:
        raise HTTPException(422, "overlap must be between 0.0 and 2.9 seconds")


def _run(path: Path, filename: str, **kw) -> AnalysisResponse:
    started = time.perf_counter()
    try:
        detections, duration = service.analyze(
            path,
            min_confidence=kw["min_confidence"],
            sensitivity=kw["sensitivity"],
            overlap=kw["overlap"],
            latitude=kw["latitude"],
            longitude=kw["longitude"],
            on_date=kw["on_date"],
        )
    except ModelNotReady as exc:
        raise HTTPException(503, f"BirdNET model is not available: {exc}") from exc
    except Exception as exc:
        logger.exception("Analysis failed for %s", filename)
        raise HTTPException(
            400, f"Could not analyze this audio ({type(exc).__name__}: {exc})"
        ) from exc

    return _build_response(
        filename=filename,
        detections=detections,
        duration=duration,
        analysis_seconds=time.perf_counter() - started,
        **kw,
    )


# -------------------------------------------------------------------- routes


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    resolved = sum(1 for key in service.resolutions if service.available(key))
    clf = service.classifier
    return HealthResponse(
        status="ok" if service.ready else "degraded",
        model_loaded=service.ready,
        model_labels_total=len(service.labels) or None,
        targets_resolved=resolved,
        targets_total=len(TARGET_SPECIES),
        custom_model_loaded=clf is not None,
        custom_model_trained_at=clf.trained_at if clf else None,
        detail=service.load_error,
    )


@app.get("/species", response_model=SpeciesResponse)
def species() -> SpeciesResponse:
    """The target species and how each maps onto the loaded model."""
    if not service.ready:
        raise HTTPException(503, f"BirdNET model is not available: {service.load_error}")

    infos = []
    for spec in TARGET_SPECIES:
        res = service.resolutions[spec.key]
        infos.append(
            SpeciesInfo(
                key=spec.key,
                common_name=spec.common_name,
                scientific_name=spec.scientific_name,
                emoji=spec.emoji,
                available_in_model=service.available(spec.key),
                scored_by=service.scored_by(spec.key),
                model_label=res.label,
                model_extra_labels=res.extra_labels,
                model_common_name=res.model_common_name,
                model_scientific_name=res.model_scientific_name,
                matched_on=res.matched_on,
            )
        )
    return SpeciesResponse(
        model_labels_total=len(service.labels),
        targets=infos,
        unavailable=[i.common_name for i in infos if not i.available_in_model],
    )


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(
    file: UploadFile = File(..., description="Audio recording (wav, mp3, flac, ogg, m4a...)."),
    min_confidence: float = Form(settings.min_confidence),
    sensitivity: float = Form(settings.sensitivity),
    overlap: float = Form(settings.overlap),
    latitude: float | None = Form(settings.latitude),
    longitude: float | None = Form(settings.longitude),
    recorded_on: date | None = Form(
        None,
        description="YYYY-MM-DD. With lat/lon, enables BirdNET's seasonal range filter.",
    ),
    include_non_target: bool = Form(False, description="Also return other species heard."),
) -> AnalysisResponse:
    """Upload a recording and get back which of the target species are in it."""
    _validate_params(min_confidence, sensitivity, overlap)

    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in settings.extension_set:
        allowed = ", ".join(sorted(settings.extension_set))
        raise HTTPException(
            415,
            f"Unsupported audio format {suffix or '(none)'}. Allowed: {allowed}",
        )

    tmp_dir = tempfile.mkdtemp(prefix="birdnet-")
    tmp_path = Path(tmp_dir) / f"audio{suffix}"
    try:
        size = 0
        with tmp_path.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        413, f"File exceeds the {settings.max_upload_mb} MB limit."
                    )
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty.")

        return _run(
            tmp_path,
            name,
            min_confidence=min_confidence,
            sensitivity=sensitivity,
            overlap=overlap,
            latitude=latitude,
            longitude=longitude,
            on_date=recorded_on,
            include_non_target=include_non_target,
        )
    finally:
        file.file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/analyze/path", response_model=AnalysisResponse)
def analyze_path(
    path: str = Query(..., description="Path relative to BIRDNET_LOCAL_AUDIO_DIR."),
    min_confidence: float = Query(settings.min_confidence),
    sensitivity: float = Query(settings.sensitivity),
    overlap: float = Query(settings.overlap),
    latitude: float | None = Query(settings.latitude),
    longitude: float | None = Query(settings.longitude),
    recorded_on: date | None = Query(None),
    include_non_target: bool = Query(False),
) -> AnalysisResponse:
    """Analyze a file already on the server, avoiding a re-upload of a large recording.

    Disabled unless BIRDNET_LOCAL_AUDIO_DIR is set; reads are confined to it.
    """
    if not settings.local_audio_dir:
        raise HTTPException(404, "Set BIRDNET_LOCAL_AUDIO_DIR to enable this endpoint.")
    _validate_params(min_confidence, sensitivity, overlap)

    root = Path(settings.local_audio_dir).resolve()
    target = (root / path).resolve()
    # Containment check, so `path` cannot escape the configured directory.
    if not target.is_relative_to(root):
        raise HTTPException(403, "Path escapes the configured audio directory.")
    if not target.is_file():
        raise HTTPException(404, f"No such file: {path}")
    if target.suffix.lower() not in settings.extension_set:
        raise HTTPException(415, f"Unsupported audio format {target.suffix}.")

    return _run(
        target,
        target.name,
        min_confidence=min_confidence,
        sensitivity=sensitivity,
        overlap=overlap,
        latitude=latitude,
        longitude=longitude,
        on_date=recorded_on,
        include_non_target=include_non_target,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
