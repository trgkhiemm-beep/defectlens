"""ASGI entry point:  uvicorn app.main:app --host 0.0.0.0 --port 7860"""
from __future__ import annotations

import logging
import os

import gradio as gr
from fastapi import FastAPI

from app.api import router
from app.service import InspectionService
from app.ui import build_ui

log = logging.getLogger("defectlens")


def create_app(model_dir: str | None = None, with_ui: bool = True) -> FastAPI:
    svc = InspectionService(model_dir)  # load + compile once at start-up, not per request
    log.info("model loaded: %s", svc.info())
    api = FastAPI(title="DefectLens", version="1.0",
                  description="Unsupervised industrial defect detection (PatchCore, OpenVINO INT8).")
    api.state.service = svc
    api.include_router(router)
    if with_ui:
        api = gr.mount_gradio_app(api, build_ui(svc), path="/")
    return api


app = create_app() if os.environ.get("DEFECTLENS_NO_AUTOLOAD") != "1" else None
