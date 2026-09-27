"""Edge (OpenVINO) export, INT8 quantisation, runtime and benchmark tests (CPU, resnet18, no downloads)."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openvino as ov  # noqa: E402

from src.edge.export import quantize_int8, to_openvino  # noqa: E402
from src.edge.runtime import EdgeInspector, preprocess  # noqa: E402
from src.models.patchcore import Embedder, PatchCore, Scorer  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    torch.manual_seed(0)
    return Embedder("resnet18", pretrained=False).eval()


def test_openvino_embedder_and_scorer_match_pytorch(embedder):
    x = torch.randn(1, 3, 128, 128)
    ref = embedder(x).detach()
    got = ov.Core().compile_model(to_openvino(embedder, x), "CPU")(x.numpy())[0]
    assert np.abs(ref.numpy() - got).max() < 1e-3 * np.abs(ref.numpy()).max()

    scorer = Scorer(torch.randn(200, embedder.embed_dim), sigma=4.0, out_size=128).eval()
    s_ref, m_ref = scorer(ref)
    out = ov.Core().compile_model(to_openvino(scorer, ref), "CPU")(ref.numpy())
    assert out[0][0] == pytest.approx(s_ref.item(), rel=1e-4)
    assert out[1].shape == (1, 1, 128, 128)
    assert np.abs(out[1] - m_ref.detach().numpy()).max() < 1e-3 * float(s_ref)


def test_separable_blur_equals_2d_gaussian():
    import torch.nn.functional as F
    from src.models.patchcore import gaussian_kernel
    sc = Scorer(torch.randn(10, 4), sigma=4.0, out_size=64)
    x = torch.rand(1, 1, 64, 64)
    p = sc.blur_x.shape[-1] // 2
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    assert torch.allclose(F.conv2d(F.conv2d(xp, sc.blur_x), sc.blur_y), F.conv2d(xp, gaussian_kernel(4.0)), atol=1e-6)


def test_scorer_equals_patchcore_forward(embedder):
    torch.manual_seed(1)
    pc = PatchCore("resnet18", pretrained=False)
    pc.memory_bank = torch.randn(300, pc.embed_dim)
    x = torch.randn(2, 3, 64, 64)
    s, a = pc(x)
    s2, a2 = pc.scorer(64)(pc.embed(x))
    assert torch.allclose(s, s2, atol=1e-4) and torch.allclose(a, a2, atol=1e-4)


def test_int8_quantisation_preserves_features(embedder):
    x = torch.randn(1, 3, 128, 128)
    fp32 = to_openvino(embedder, x)
    calib = [np.random.default_rng(i).standard_normal((1, 3, 128, 128)).astype(np.float32) for i in range(8)]
    int8 = ov.Core().compile_model(quantize_int8(fp32, calib, 8), "CPU")(x.numpy())[0]
    ref = embedder(x).detach().numpy()
    assert np.corrcoef(ref.ravel(), int8.ravel())[0, 1] > 0.98


def test_preprocess_matches_training_transform(built):
    from src.data.transforms import build_eval_transform
    path = next((built[0] / "transistor" / "test" / "good").glob("*.png"))
    bgr = cv2.imread(str(path))
    ref = build_eval_transform(256)(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))["image"].numpy()
    assert np.abs(preprocess(bgr, 256)[0] - ref).max() < 1e-5


@pytest.fixture(scope="module")
def edge(built, tmp_path_factory):
    out = tmp_path_factory.mktemp("edge") / "edge"
    subprocess.run([sys.executable, "scripts/build_edge.py", "--data", str(built[0]), "--out", str(out),
                    "--ratios", "0.5", "0.05", "--save-all-max-ratio", "0.1", "--calib-size", "8",
                    "--backbone", "resnet18", "--no-pretrained"], cwd=ROOT, check=True)
    return out


def test_build_edge_outputs(edge):
    m = json.loads((edge / "manifest.json").read_text())
    assert set(m["variants"]) == {f"{p}/{r}" for p in ("fp32", "int8", "int8mix") for r in ("r0.5", "r0.05")}
    assert m["embedders"]["int8mix"]["calibration_images"] == 8
    first = next(iter(m["variants"]["fp32/r0.5"]["categories"]))
    for key, v in m["variants"].items():
        for cat, info in v["categories"].items():
            assert 0 <= info["image_auroc"] <= 1 and info["threshold"] > 0
            expect_scorer = v["coreset_ratio"] <= 0.1 or cat == first
            assert bool(info["scorer"]) == expect_scorer, (key, cat)
            if info["scorer"]:
                assert (edge / info["scorer"]).exists() and info["export_rel_diff"] < 1e-2
    assert m["embedders"]["int8"]["mb"] < m["embedders"]["fp32"]["mb"] / 2
    assert m["embedders"]["int8mix"]["mb"] < m["embedders"]["fp32"]["mb"] / 2
    assert "Mean image AUROC" in (edge / "report.md").read_text()
    assert len(list((edge / "samples").glob("*.png"))) == 2 * len(m["variants"]["fp32/r0.5"]["categories"])


def test_edge_inspector_predict(edge):
    insp = EdgeInspector(edge, "int8", "r0.05", "CPU")
    img = cv2.imread(str(edge / "samples" / "transistor_defect.png"))
    r = insp.predict(img, "transistor")
    assert r["anomaly_map"].shape == (256, 256) and isinstance(r["is_defect"], bool)
    assert r["timing_ms"]["total"] > 0 and r["threshold"] == insp.thresholds["transistor"]


def test_benchmark_script(edge):
    subprocess.run([sys.executable, "scripts/benchmark.py", "--edge", str(edge), "--devices", "CPU",
                    "--runs", "2", "--warmup", "1", "--no-pretrained"], cwd=ROOT, check=True)
    md = (edge / "benchmark.md").read_text()
    assert "PyTorch eager" in md and "OpenVINO" in md and "INT8" in md
    res = json.loads((edge / "benchmark.json").read_text())["results"]
    assert all(r["p50_ms"] > 0 for r in res)
