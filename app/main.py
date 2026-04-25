from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

from .model_arch import EffKANSeg
from .processor import (
    CLASS_NAMES,
    COLOR_MAP,
    build_color_mask,
    compute_area_stats,
    compute_change_analysis,
    compute_confidence_summary,
    compute_stats,
    generate_operational_insights,
    run_segmentation,
)
from .reporting import build_pdf_report
from .satellite import SatelliteFetchError, fetch_recent_scene_by_coordinates
from .schemas import CoordinateAnalysisRequest, ReportRequest

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

WEIGHTS_DIR = os.environ.get("SEGIMBUD_WEIGHTS_DIR", "weights")
WEIGHTS_FILE = os.environ.get("SEGIMBUD_WEIGHTS_FILE", "best_model_loveda_EffKANSeg_v9_1024.pth")
HF_REPO_ID = os.environ.get("SEGIMBUD_HF_REPO_ID", "Mazbah1/effkanseg")
AUTO_DOWNLOAD_WEIGHTS = os.environ.get("SEGIMBUD_AUTO_DOWNLOAD_WEIGHTS", "true").lower() not in {
    "0",
    "false",
    "no",
}
DESTINATION = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)
APP_VERSION = "segimbud-ui-2026-04-25-v4"

NAV_ITEMS = [
    {"id": "scene", "label": "Scene", "href": "/scene"},
    {"id": "coordinate", "label": "Coordinate", "href": "/coordinate"},
    {"id": "batch", "label": "Batch", "href": "/batch"},
    {"id": "review", "label": "Review", "href": "/review"},
    {"id": "change", "label": "Change", "href": "/change"},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: EffKANSeg | None = None
model_error: str | None = None

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def ensure_weights() -> str:
    if os.path.exists(DESTINATION):
        logger.info("Using weights from %s", DESTINATION)
        return DESTINATION

    if not AUTO_DOWNLOAD_WEIGHTS:
        raise FileNotFoundError(
            f"Model weights were not found at '{DESTINATION}'. "
            "Place the checkpoint there or enable auto-download."
        )

    logger.info("Weights not found. Downloading '%s' from %s...", WEIGHTS_FILE, HF_REPO_ID)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=WEIGHTS_FILE,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
    )

    if not os.path.exists(DESTINATION):
        raise FileNotFoundError(f"Weights download completed but '{DESTINATION}' is still missing.")

    return DESTINATION


def load_model() -> EffKANSeg:
    checkpoint_path = ensure_weights()
    network = EffKANSeg(num_classes=len(CLASS_NAMES))
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    network.load_state_dict(state_dict)
    network.to(device).eval()
    return network


def parse_uploaded_image(file: UploadFile) -> Image.Image:
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported image file.") from exc


def encode_png(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def encode_pil_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def confidence_to_heatmap(conf_map: np.ndarray) -> np.ndarray:
    normalized = np.clip(conf_map, 0.0, 1.0)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def build_health_payload() -> dict:
    ready = model is not None and model_error is None
    return {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "app_version": APP_VERSION,
        "device": str(device),
        "model_name": "EffKANSeg",
        "weights_path": DESTINATION,
        "error": model_error,
    }


def _require_model() -> EffKANSeg:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "The model is not ready yet. Check /api/health for startup details.",
        )
    return model


def predict_image(
    image: Image.Image,
    resolution_m_per_px: float | None = None,
    *,
    tile_size: int = 1024,
    tile_overlap: int = 160,
    force_tiled: bool = False,
    include_original_image: bool = False,
    source_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    network = _require_model()
    started_at = time.time()
    class_mask, conf_map, processing = run_segmentation(
        image,
        network,
        device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        force_tiled=force_tiled,
    )
    inference_time = round(time.time() - started_at, 3)

    color_mask = build_color_mask(class_mask)
    heatmap = confidence_to_heatmap(conf_map)
    stats = compute_area_stats(compute_stats(class_mask), resolution_m_per_px)
    confidence_summary = compute_confidence_summary(conf_map)
    insights = generate_operational_insights(stats, confidence_summary)

    payload: dict[str, Any] = {
        "mask_image": encode_png(color_mask),
        "class_mask_image": encode_png(class_mask),
        "confidence_map_image": encode_png((np.clip(conf_map, 0.0, 1.0) * 255).astype(np.uint8)),
        "heatmap_image": encode_png(heatmap),
        "stats": stats,
        "confidence_summary": confidence_summary,
        "operational_insights": insights,
        "legend": [
            {
                "class_id": class_id,
                "name": class_name,
                "color": COLOR_MAP[class_id],
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
        "inference_time": inference_time,
        "device": str(device),
        "resolution_m_per_px": resolution_m_per_px,
        "input_size": [image.width, image.height],
        "output_size": [image.width, image.height],
        "processing": processing,
    }

    if include_original_image:
        payload["original_image"] = encode_pil_image(image)
    if source_metadata is not None:
        payload["source_metadata"] = source_metadata

    return payload, class_mask, conf_map


def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid report image payload.") from exc


def build_report_response(report_request: ReportRequest) -> Response:
    pdf_bytes = build_pdf_report(
        report_title=report_request.report_title,
        mode_label=report_request.mode_label,
        summary_metrics=[(metric.label, metric.value) for metric in report_request.summary_metrics],
        image_panels=[
            (panel.title, decode_base64_image(panel.image_base64))
            for panel in report_request.image_panels
        ],
        insights=report_request.insights,
        stats_rows=report_request.stats_rows,
        extra_notes=report_request.extra_notes,
    )
    filename = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in report_request.report_title.lower()
    ).strip("-") or "segimbud-report"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
    )


def build_page_context(
    request: Request,
    *,
    active_page: str,
    page_title: str,
    page_description: str,
    page_styles: list[str],
    page_scripts: list[str],
) -> dict:
    return {
        "request": request,
        "page_title": page_title,
        "page_description": page_description,
        "active_page": active_page,
        "nav_items": NAV_ITEMS,
        "page_styles": page_styles,
        "page_scripts": page_scripts,
        "page_config": {
            "apiBase": "/api",
            "legend": [
                {"class_id": class_id, "name": class_name, "color": COLOR_MAP[class_id]}
                for class_id, class_name in enumerate(CLASS_NAMES)
            ],
            "classes": CLASS_NAMES,
            "appVersion": APP_VERSION,
        },
    }


def build_batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in results if item.get("ok")]
    total_inference_time = round(
        sum(float(item["prediction"]["inference_time"]) for item in successful),
        3,
    )
    total_tiles = sum(int(item["prediction"]["processing"]["tile_count"]) for item in successful)
    return {
        "requested_count": len(results),
        "successful_count": len(successful),
        "failed_count": len(results) - len(successful),
        "total_inference_time": total_inference_time,
        "total_tile_count": total_tiles,
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, model_error

    try:
        model = load_model()
        model_error = None
        logger.info("SegImBud model loaded on %s", device)
    except Exception as exc:
        model = None
        model_error = str(exc)
        logger.exception("Failed to initialize model")

    yield


app = FastAPI(title="SegImBud", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def home_page():
    return RedirectResponse(url="/scene", status_code=307)


@app.get("/scene", response_class=HTMLResponse, include_in_schema=False)
def scene_page(request: Request):
    context = build_page_context(
        request,
        active_page="scene",
        page_title="Scene Analysis",
        page_description="Single-scene segmentation workspace",
        page_styles=["css/pages/scene.css"],
        page_scripts=["js/pages/scene.js"],
    )
    return templates.TemplateResponse("pages/scene.html", context)


@app.get("/coordinate", response_class=HTMLResponse, include_in_schema=False)
def coordinate_page(request: Request):
    context = build_page_context(
        request,
        active_page="coordinate",
        page_title="Coordinate Analysis",
        page_description="Fetch recent satellite imagery by coordinates and run segmentation",
        page_styles=["css/pages/coordinate.css"],
        page_scripts=["js/pages/coordinate.js"],
    )
    return templates.TemplateResponse("pages/coordinate.html", context)


@app.get("/batch", response_class=HTMLResponse, include_in_schema=False)
def batch_page(request: Request):
    context = build_page_context(
        request,
        active_page="batch",
        page_title="Batch Processing",
        page_description="Process multiple scenes with tiled inference",
        page_styles=["css/pages/batch.css"],
        page_scripts=["js/pages/batch.js"],
    )
    return templates.TemplateResponse("pages/batch.html", context)


@app.get("/review", response_class=HTMLResponse, include_in_schema=False)
def review_page(request: Request):
    context = build_page_context(
        request,
        active_page="review",
        page_title="Review and QA",
        page_description="Manual review, correction, and QA export workspace",
        page_styles=["css/pages/review.css"],
        page_scripts=["js/pages/review.js"],
    )
    return templates.TemplateResponse("pages/review.html", context)


@app.get("/change", response_class=HTMLResponse, include_in_schema=False)
def change_page(request: Request):
    context = build_page_context(
        request,
        active_page="change",
        page_title="Change Detection",
        page_description="Two-scene change intelligence workspace",
        page_styles=["css/pages/change.css"],
        page_scripts=["js/pages/change.js"],
    )
    return templates.TemplateResponse("pages/change.html", context)


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
@app.get("/healthz", include_in_schema=False)
def health():
    return build_health_payload()


@app.get("/api/legend")
def legend():
    return {
        "legend": [
            {"class_id": class_id, "name": class_name, "color": COLOR_MAP[class_id]}
            for class_id, class_name in enumerate(CLASS_NAMES)
        ]
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.post("/api/predict")
@app.post("/predict", include_in_schema=False)
async def predict(
    file: UploadFile = File(...),
    resolution_m_per_px: float | None = Form(default=None),
    tile_size: int = Form(default=1024),
    tile_overlap: int = Form(default=160),
    force_tiled: bool = Form(default=False),
):
    image = parse_uploaded_image(file)
    payload, _, _ = predict_image(
        image,
        resolution_m_per_px,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        force_tiled=force_tiled,
    )
    return payload


@app.post("/api/coordinate-predict")
async def coordinate_predict(request_body: CoordinateAnalysisRequest):
    try:
        image, source_metadata = fetch_recent_scene_by_coordinates(
            request_body.latitude,
            request_body.longitude,
            area_km=request_body.area_km,
            search_days=request_body.search_days,
            max_cloud_cover=request_body.max_cloud_cover,
            output_size_px=request_body.output_size_px,
        )
    except SatelliteFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    payload, _, _ = predict_image(
        image,
        resolution_m_per_px=source_metadata["resolution_m_per_px"],
        include_original_image=True,
        source_metadata=source_metadata,
    )
    return payload


@app.post("/api/batch-predict")
async def batch_predict(
    files: list[UploadFile] = File(...),
    resolution_m_per_px: float | None = Form(default=None),
    tile_size: int = Form(default=1024),
    tile_overlap: int = Form(default=160),
    force_tiled: bool = Form(default=True),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required for batch processing.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Batch processing currently supports up to 10 images at a time.")

    results: list[dict[str, Any]] = []
    for upload in files:
        try:
            image = parse_uploaded_image(upload)
            payload, _, _ = predict_image(
                image,
                resolution_m_per_px=resolution_m_per_px,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                force_tiled=force_tiled,
            )
            results.append(
                {
                    "ok": True,
                    "filename": upload.filename or "scene",
                    "prediction": payload,
                }
            )
        except Exception as exc:
            logger.exception("Batch prediction failed for %s", upload.filename)
            results.append(
                {
                    "ok": False,
                    "filename": upload.filename or "scene",
                    "error": str(exc),
                }
            )

    return {
        "results": results,
        "summary": build_batch_summary(results),
    }


@app.post("/api/change-detection")
async def change_detection(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    resolution_m_per_px: float | None = Form(default=None),
    tile_size: int = Form(default=1024),
    tile_overlap: int = Form(default=160),
    force_tiled: bool = Form(default=False),
):
    before_image = parse_uploaded_image(before_file)
    after_image = parse_uploaded_image(after_file)

    before_payload, before_mask, _ = predict_image(
        before_image,
        resolution_m_per_px,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        force_tiled=force_tiled,
    )
    after_payload, after_mask, after_conf_map = predict_image(
        after_image,
        resolution_m_per_px,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        force_tiled=force_tiled,
    )

    change_summary = compute_change_analysis(
        before_mask,
        after_mask,
        resolution_m_per_px=resolution_m_per_px,
    )
    change_insights = generate_operational_insights(
        after_payload["stats"],
        compute_confidence_summary(after_conf_map),
        change_summary,
    )

    return {
        "before": before_payload,
        "after": after_payload,
        "change_mask_image": encode_png(change_summary["change_mask"]),
        "change_summary": {
            key: value for key, value in change_summary.items() if key != "change_mask"
        },
        "operational_insights": change_insights,
        "legend": [
            {"class_id": class_id, "name": class_name, "color": COLOR_MAP[class_id]}
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
    }


@app.post("/api/reports/pdf")
async def create_pdf_report(report_request: ReportRequest):
    return build_report_response(report_request)
