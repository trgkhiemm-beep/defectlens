"""Inspection service: model + production monitoring, independent of the web layer.

Shared by the REST API and the Gradio UI. Everything here is CPU/OpenVINO only
(no torch), so the serving image stays small.
"""
from __future__ import annotations

import os
import threading
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from src.edge.runtime import EdgeInspector

MAX_IMAGE_BYTES = 10 * 2**20
OOD_FACTOR = 3.0  # score this many times the threshold -> probably not the selected part at all


class DriftMonitor:
    """Watches the scores of parts judged OK. If their rolling median rises by more than
    `alert_ratio` over the reference (val-normal median from training, or else the first
    `window` accepted parts after start-up), the line has probably changed: new material
    lot, lighting, camera. Lesson from `carpet`: test normals scored x1.09 higher than
    val normals and the fixed threshold rejected 50% of good parts."""

    def __init__(self, reference: float | None, window: int = 100, alert_ratio: float = 1.05, min_samples: int = 30):
        self.reference, self.window, self.alert_ratio, self.min_samples = reference, window, alert_ratio, min_samples
        self.reference_source = "validation" if reference else "warm-up"
        self.scores: deque[float] = deque(maxlen=window)
        self._warmup: list[float] = []

    def update(self, score: float) -> None:
        if self.reference is None:
            self._warmup.append(score)
            if len(self._warmup) >= self.window:
                self.reference = float(np.median(self._warmup))
            return
        self.scores.append(score)

    def status(self) -> dict:
        if self.reference is None:
            return {"state": "warming_up", "samples": len(self._warmup), "needed": self.window}
        if len(self.scores) < self.min_samples:
            return {"state": "collecting", "samples": len(self.scores), "reference": self.reference,
                    "reference_source": self.reference_source}
        ratio = float(np.median(self.scores)) / self.reference
        return {"state": "drift" if ratio > self.alert_ratio else "ok", "ratio": round(ratio, 4),
                "alert_ratio": self.alert_ratio, "samples": len(self.scores), "reference": self.reference,
                "reference_source": self.reference_source,
                "action": "collect new normal samples and recalibrate the threshold" if ratio > self.alert_ratio else None}


class InspectionService:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        model_dir = Path(model_dir or os.environ.get("DEFECTLENS_MODEL_DIR", "deploy/model"))
        device = device or os.environ.get("DEFECTLENS_DEVICE", "CPU")
        self.model_dir = model_dir
        inspector = EdgeInspector.from_bundle(model_dir, device, cache_dir=os.environ.get("DEFECTLENS_CACHE_DIR"))
        self.inspector, self.device = inspector, device
        self._lock = threading.Lock()  # an OpenVINO CompiledModel call is not re-entrant
        refs = inspector.reference_medians
        self.drift = {c: DriftMonitor(refs.get(c)) for c in inspector.scorers}
        self.counts: dict[str, Counter] = {c: Counter() for c in inspector.scorers}
        self.latency: deque[float] = deque(maxlen=500)

    @property
    def categories(self) -> list[str]:
        return sorted(self.inspector.scorers)

    def info(self) -> dict:
        m = self.inspector.manifest
        return {"categories": {c: {"threshold": self.inspector.thresholds[c]} for c in self.categories},
                "device": self.device, "variant": m.get("default"), "backbone": m["backbone"],
                "image_size": m["image_size"], "compile_ms": round(self.inspector.compile_ms, 1),
                "lineage": m.get("lineage", {})}

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("empty upload")
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"image larger than {MAX_IMAGE_BYTES // 2**20} MB")
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # drops alpha, expands grey
        except cv2.error:
            img = None
        if img is None:
            raise ValueError("not a decodable image (PNG/JPEG/BMP/WebP expected)")
        return img

    def inspect(self, image_bgr: np.ndarray, category: str) -> dict:
        if category not in self.inspector.scorers:
            raise KeyError(category)
        with self._lock:
            r = self.inspector.predict(image_bgr, category)
        ratio = r["score"] / r["threshold"]
        warnings = []
        if ratio >= OOD_FACTOR:
            warnings.append(f"score is {ratio:.1f}x the threshold: the image may not show a {category} "
                            "(wrong part, wrong camera or not an inspection image)")
        self.counts[category]["defect" if r["is_defect"] else "ok"] += 1
        if not r["is_defect"]:
            self.drift[category].update(r["score"])
        self.latency.append(r["timing_ms"]["total"])
        return {**r, "score_ratio": ratio, "warnings": warnings}

    def metrics(self) -> dict:
        lat = np.array(self.latency) if self.latency else np.array([np.nan])
        return {"device": self.device,
                "latency_ms": {"p50": float(np.nanpercentile(lat, 50)), "p95": float(np.nanpercentile(lat, 95)),
                               "window": len(self.latency)},
                "categories": {c: {"inspected": sum(self.counts[c].values()), "defects": self.counts[c]["defect"],
                                   "defect_rate": self.counts[c]["defect"] / max(1, sum(self.counts[c].values())),
                                   "drift": self.drift[c].status()} for c in self.categories}}


def render_overlay(image_bgr: np.ndarray, amap: np.ndarray, threshold: float) -> np.ndarray:
    """Heatmap over the image, coloured relative to the threshold (mid-scale == threshold);
    contours mark regions above the threshold. Returns BGR at model resolution."""
    size = amap.shape[0]
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap((np.clip(amap / (2 * threshold), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    cnts, _ = cv2.findContours((amap >= threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out
