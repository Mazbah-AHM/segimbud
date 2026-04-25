from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

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
    postprocess,
    preprocess,
    resize_prediction_maps,
)
from .reporting import build_pdf_report
from .schemas import ReportRequest

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
APP_VERSION = "segimbud-ui-2026-04-25-v3"

NAV_ITEMS = [
    {"id": "scene", "label": "Scene Analysis", "href": "/scene"},
    {"id": "change", "label": "Change Detection", "href": "/change"},
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


def build_prediction_payload(
    image: Image.Image,
    resolution_m_per_px: float | None = None,
) -> dict:
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=model_error or "The model is not ready yet. Check /api/health for startup details.",
        )

    original_width, original_height = image.size
    input_tensor = preprocess(image).to(device)

    started_at = time.time()
    with torch.inference_mode():
        output = model(input_tensor)
    inference_time = round(time.time() - started_at, 3)

    _, class_mask, conf_map = postprocess(output)
    class_mask, conf_map = resize_prediction_maps(
        class_mask,
        conf_map,
        (original_width, original_height),
    )

    color_mask = build_color_mask(class_mask)
    heatmap = confidence_to_heatmap(conf_map)
    stats = compute_area_stats(compute_stats(class_mask), resolution_m_per_px)
    confidence_summary = compute_confidence_summary(conf_map)
    insights = generate_operational_insights(stats, confidence_summary)

    return {
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
        "input_size": [original_width, original_height],
        "output_size": [original_width, original_height],
    }


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
        },
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
):
    image = parse_uploaded_image(file)
    return build_prediction_payload(image, resolution_m_per_px)


@app.post("/api/change-detection")
async def change_detection(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    resolution_m_per_px: float | None = Form(default=None),
):
    before_image = parse_uploaded_image(before_file)
    after_image = parse_uploaded_image(after_file)

    before_payload = build_prediction_payload(before_image, resolution_m_per_px)
    after_payload = build_prediction_payload(after_image, resolution_m_per_px)

    before_mask = np.array(Image.open(BytesIO(base64.b64decode(before_payload["class_mask_image"]))).convert("L"))
    after_mask = np.array(Image.open(BytesIO(base64.b64decode(after_payload["class_mask_image"]))).convert("L"))

    change_summary = compute_change_analysis(
        before_mask,
        after_mask,
        resolution_m_per_px=resolution_m_per_px,
    )
    after_conf_map = np.array(
        Image.open(BytesIO(base64.b64decode(after_payload["confidence_map_image"]))).convert("L"),
        dtype=np.float32,
    ) / 255.0
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

