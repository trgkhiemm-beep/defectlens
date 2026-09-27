"""REST API (versioned under /api/v1). OpenAPI docs at /docs."""
from __future__ import annotations

import base64
from typing import Annotated

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.service import InspectionService, render_overlay

router = APIRouter(prefix="/api/v1")


class Timing(BaseModel):
    preprocess: float
    embed: float
    score: float
    total: float


class InspectionResult(BaseModel):
    category: str
    verdict: str = Field(description="DEFECT or OK")
    is_defect: bool
    score: float = Field(description="max anomaly-map value (nearest-neighbour distance)")
    threshold: float
    score_ratio: float = Field(description="score / threshold; >= 1 means DEFECT")
    warnings: list[str]
    timing_ms: Timing
    heatmap_png_base64: str | None = None


def service(request: Request) -> InspectionService:
    return request.app.state.service


@router.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", **service(request).info()}


@router.get("/categories")
def categories(request: Request) -> dict:
    return service(request).info()["categories"]


@router.get("/metrics")
def metrics(request: Request) -> dict:
    return service(request).metrics()


@router.post("/inspect", response_model=InspectionResult)
async def inspect(request: Request, file: Annotated[UploadFile, File(description="PNG/JPEG image of one part")],
                  category: Annotated[str, Form()], heatmap: Annotated[bool, Form()] = False) -> InspectionResult:
    svc = service(request)
    if category not in svc.categories:
        raise HTTPException(404, f"unknown category {category!r}; available: {svc.categories}")
    try:
        img = svc.decode(await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    r = svc.inspect(img, category)
    png = None
    if heatmap:
        ok, buf = cv2.imencode(".png", render_overlay(img, r["anomaly_map"], r["threshold"]))
        png = base64.b64encode(buf.tobytes()).decode() if ok else None
    return InspectionResult(category=category, verdict="DEFECT" if r["is_defect"] else "OK",
                            is_defect=r["is_defect"], score=r["score"], threshold=r["threshold"],
                            score_ratio=r["score_ratio"], warnings=r["warnings"], timing_ms=Timing(**r["timing_ms"]),
                            heatmap_png_base64=png)
