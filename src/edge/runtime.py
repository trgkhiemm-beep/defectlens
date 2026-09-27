"""Production inference on OpenVINO: shared embedder + one scorer per category.

  inspector = EdgeInspector("edge", precision="int8", bank="r0.01", device="CPU")
  result = inspector.predict(image_bgr, "transistor")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import openvino as ov

from src.edge.preprocess import preprocess  # noqa: F401  (re-exported)


class EdgeInspector:
    def __init__(self, edge_dir: str | Path, precision: str = "int8", bank: str = "r0.01", device: str = "CPU",
                 categories: list[str] | None = None, cache_dir: str | Path | None = None):
        self.dir = Path(edge_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.size = self.manifest["image_size"]
        variant = self.manifest["variants"][f"{precision}/{bank}"]
        self.reference_medians = {c: i["val_normal_median"] for c, i in variant["categories"].items()
                                  if i.get("val_normal_median")}
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

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path, device: str = "CPU", **kw) -> "EdgeInspector":
        """Load the deployment bundle written by scripts/export_bundle.py (its default variant)."""
        d = json.loads((Path(bundle_dir) / "manifest.json").read_text())["default"]
        return cls(bundle_dir, d["precision"], d["bank"], device, **kw)

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
