"""Gradio demo, mounted on the same FastAPI app (one container, one port)."""
from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from app.service import InspectionService, render_overlay

INTRO = """# DefectLens: unsupervised visual inspection on a CPU
Trained **only on defect-free parts** (PatchCore, MVTec AD). INT8 OpenVINO model, runs on the CPU of this Space.
Pick a category, upload a photo of that part (or click an example), and see where the model thinks the defect is.
REST API: [`/docs`](/docs) · Monitoring: [`/api/v1/metrics`](/api/v1/metrics)"""


def build_ui(svc: InspectionService) -> gr.Blocks:
    samples = Path(svc.model_dir) / "samples"
    examples = [[str(p), p.stem.rsplit("_", 1)[0]] for p in sorted(samples.glob("*.png"))
                if p.stem.rsplit("_", 1)[0] in svc.categories]

    def run(image_rgb: np.ndarray | None, category: str):
        if image_rgb is None:
            raise gr.Error("Upload an image or pick an example first.")
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        r = svc.inspect(bgr, category)
        overlay = cv2.cvtColor(render_overlay(bgr, r["anomaly_map"], r["threshold"]), cv2.COLOR_BGR2RGB)
        # Not a probability: PatchCore scores are distances, so show them against the threshold.
        verdict = f"{'DEFECT' if r['is_defect'] else 'OK'}  (score = {r['score_ratio']:.2f} x threshold)"
        details = {"score": round(r["score"], 4), "threshold": round(r["threshold"], 4),
                   "score / threshold": round(r["score_ratio"], 3),
                   "latency_ms": {k: round(v, 1) for k, v in r["timing_ms"].items()}, "warnings": r["warnings"]}
        return overlay, verdict, details

    with gr.Blocks(title="DefectLens") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Part image", type="numpy")
                category = gr.Dropdown(svc.categories, value=svc.categories[0], label="Category")
                button = gr.Button("Inspect", variant="primary")
            with gr.Column():
                overlay = gr.Image(label="Anomaly map (white contour = above threshold)")
                verdict = gr.Label(label="Verdict")
                details = gr.JSON(label="Details")
        if examples:
            gr.Examples(examples, inputs=[image, category], label="Examples from the MVTec AD test set")
        with gr.Accordion("Production monitoring (drift, defect rate, latency)", open=False):
            monitor = gr.JSON()
            gr.Button("Refresh").click(svc.metrics, outputs=monitor)
        button.click(run, inputs=[image, category], outputs=[overlay, verdict, details])
    return demo
