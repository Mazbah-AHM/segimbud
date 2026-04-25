from __future__ import annotations

from pydantic import BaseModel, Field


class ReportMetric(BaseModel):
    label: str
    value: str


class ReportImagePanel(BaseModel):
    title: str
    image_base64: str


class ReportRequest(BaseModel):
    report_title: str
    mode_label: str
    summary_metrics: list[ReportMetric]
    image_panels: list[ReportImagePanel]
    insights: list[str]
    stats_rows: list[dict]
    extra_notes: list[str] = Field(default_factory=list)


class CoordinateAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    area_km: float = Field(default=5.0, ge=0.2, le=50.0)
    search_days: int = Field(default=21, ge=1, le=180)
    max_cloud_cover: float = Field(default=20.0, ge=0.0, le=100.0)
    output_size_px: int = Field(default=1024, ge=256, le=2048)
