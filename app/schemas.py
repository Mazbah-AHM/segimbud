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
