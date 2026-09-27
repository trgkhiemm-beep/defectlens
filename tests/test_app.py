"""Serving layer: bundle export, REST API, Gradio mount, drift monitor, torch-free imports."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.service import DriftMonitor  # noqa: E402


@pytest.fixture(scope="module")
def bundle(edge, tmp_path_factory):
    out = tmp_path_factory.mktemp("bundle") / "model"
    subprocess.run([sys.executable, "scripts/export_bundle.py", "--edge", str(edge), "--precision", "int8",
                    "--bank", "r0.05", "--out", str(out)], cwd=ROOT, check=True)
    return out


@pytest.fixture(scope="module")
def client(bundle):
    from app.main import create_app
    return TestClient(create_app(str(bundle)))


def test_bundle_contains_only_one_variant(bundle):
    import json
    m = json.loads((bundle / "manifest.json").read_text())
    assert m["default"] == {"precision": "int8", "bank": "r0.05"}
    assert list(m["variants"]) == ["int8/r0.05"] and list(m["embedders"]) == ["int8"]
    assert not (bundle / "embedder_fp32.bin").exists() and (bundle / "MODEL_CARD.md").exists()


def test_health_and_categories(client):
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and set(h["categories"]) == {"metal_nut", "transistor"}
    assert client.get("/api/v1/categories").json()["transistor"]["threshold"] > 0


def test_inspect_returns_verdict_and_heatmap(client, bundle):
    with open(bundle / "samples" / "transistor_defect.png", "rb") as f:
        r = client.post("/api/v1/inspect", files={"file": f}, data={"category": "transistor", "heatmap": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in ("OK", "DEFECT") and body["is_defect"] == (body["score"] >= body["threshold"])
    assert body["heatmap_png_base64"] and body["timing_ms"]["total"] > 0
    m = client.get("/api/v1/metrics").json()
    assert m["categories"]["transistor"]["inspected"] >= 1 and m["latency_ms"]["window"] >= 1


@pytest.mark.parametrize("files,category,status", [
    ({"file": ("notes.txt", b"not an image")}, "transistor", 400),
    ({"file": ("empty.png", b"")}, "transistor", 400),
    (None, "pizza", 404),
])
def test_inspect_rejects_bad_input(client, bundle, files, category, status):
    files = files or {"file": open(bundle / "samples" / "transistor_good.png", "rb")}
    assert client.post("/api/v1/inspect", files=files, data={"category": category}).status_code == status


def test_ui_and_openapi_are_served(client):
    assert client.get("/").status_code == 200
    assert "/api/v1/inspect" in client.get("/openapi.json").json()["paths"]


def test_serving_code_does_not_import_torch():
    code = "import sys, app.service, app.api, app.ui; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_drift_monitor_states():
    d = DriftMonitor(reference=1.0, window=50, alert_ratio=1.05, min_samples=10)
    assert d.status()["state"] == "collecting"
    for _ in range(20):
        d.update(1.01)
    assert d.status()["state"] == "ok"
    for _ in range(50):
        d.update(1.10)  # new lot scores 10% higher
    s = d.status()
    assert s["state"] == "drift" and s["ratio"] == pytest.approx(1.10) and s["action"]


def test_drift_monitor_warm_up_reference():
    d = DriftMonitor(reference=None, window=5, min_samples=3)
    for _ in range(5):
        d.update(2.0)
    assert d.reference == 2.0 and d.reference_source == "warm-up"
    for _ in range(3):
        d.update(2.0)
    assert d.status()["state"] == "ok"
