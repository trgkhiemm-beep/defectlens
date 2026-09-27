"""Production inference on OpenVINO: shared embedder + one scorer per category.

  inspector = EdgeInspector("edge", precision="int8", bank="r0.01", device="CPU")
  result = inspector.predict(image_bgr, "transistor")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """Same transform as training (prepare_data INTER_AREA resize + ImageNet normalisation)."""
    if image_bgr.shape[:2] != (size, size):
        image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy()


class EdgeInspector:
    def __init__(self, edge_dir: str | Path, precision: str = "int8", bank: str = "r0.01", device: str = "CPU",
                 categories: list[str] | None = None, cache_dir: str | Path | None = None):
        self.dir = Path(edge_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.size = self.manifest["image_size"]
        variant = self.manifest["variants"][f"{precision}/{bank}"]
        core = ov.Core()
        if cache_dir:  # compiled-blob cache: big cold-start win on iGPU
            core.set_property({"CACHE_DIR": str(cache_dir)})
        hint = {"PERFORMANCE_HINT": "LATENCY"}
        t0 = time.perf_counter()
        self.embedder = core.compile_model(str(self.dir / self.manifest["embedders"][precision]["xml"]), device, hint)
        self.scorers, self.thresholds = {}, {}
        for cat, info in variant["categories"].items():
            if info.get("scorer") and (categories is None or cat in categories):
                self.scorers[cat] = core.compile_model(str(self.dir / info["scorer"]), device, hint)
                self.thresholds[cat] = info["threshold"]
        self.compile_ms = (time.perf_counter() - t0) * 1000
        if not self.scorers:
            raise ValueError(f"no scorer for {precision}/{bank} {categories or ''} in {self.dir}")

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embedder(x)[0]

    def score(self, category: str, emb: np.ndarray) -> tuple[float, np.ndarray]:
        out = self.scorers[category](emb)
        return float(out[0][0]), out[1][0, 0]

    def predict(self, image_bgr: np.ndarray, category: str) -> dict:
        t0 = time.perf_counter()
        x = preprocess(image_bgr, self.size)
        t1 = time.perf_counter()
        emb = self.embed(x)
        t2 = time.perf_counter()
        score, amap = self.score(category, emb)
        t3 = time.perf_counter()
        thr = self.thresholds[category]
        return {"category": category, "score": score, "threshold": thr, "is_defect": score >= thr,
                "anomaly_map": amap,
                "timing_ms": {"preprocess": (t1 - t0) * 1000, "embed": (t2 - t1) * 1000,
                              "score": (t3 - t2) * 1000, "total": (t3 - t0) * 1000}}
