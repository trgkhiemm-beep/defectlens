"""ASGI app factory:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 7860

A factory rather than a module-level `app = create_app()`: importing this module must not
load a model (a module-level app made every test that imported it depend on a local
deploy/model/ bundle, which exists on a dev machine but not in CI).
"""
from __future__ import annotations

import logging

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
